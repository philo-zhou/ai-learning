# SGLang 分布式通信层导读（与 vLLM 对照）

> 层次：上层 · 来源：third_party/sglang@d2a4364、third_party/vllm@02653fa（对照）、ai-infra-book 第 9 章（`manuscripts/09-分布式推理.md` 9.2、9.4）

## 一句话

SGLang 的通信骨架（`GroupCoordinator`、pynccl、custom allreduce、shm_broadcast）源自 vLLM v0.6.4.post1，之后向“DeepSeek 式大规模 EP 推理”方向分叉：更细的进程组（attention-TP/DP、moe-EP/TP/DP、CP）、DP-Attention 的 token gather/scatter、层边界通信编排（`layer_boundary/`）、DeepEP normal/low_latency 按 batch 自动切换、双 batch 重叠（TBO），以及基于 Mooncake/NIXL 的 PD 分离。

## 在调用链中的位置

```text
模型层（DeepSeek/Qwen MoE 等）
  ├─ layer_boundary/ops.py：tp_reduce_scatter / attn_tp_gather_input / dp_gather / dp_reduce_scatterv ...
  │    └─ dp_attention.py：_dp_gather_via_all_reduce | _all_gather | _all_gatherv；dp_scatter
  ├─ communication_op.py：tensor_model_parallel_all_reduce / moe_tensor_model_parallel_all_reduce ...
  │    └─ GroupCoordinator.all_reduce（parallel_state.py）
  │         └─ "ca" | "pcie_ipc" | "qr" | "pymscclpp" | "torch_symm_mem" | "pynccl" | torch.distributed
  └─ MoE：token_dispatcher/deepep.py DeepEPDispatcher（dispatch_a/b, combine_a/b）
         └─ deep_ep.Buffer：get_dispatch_layout/dispatch/combine 或 low_latency_dispatch/combine
PD 分离：disaggregation/prefill.py ↔ decode.py ── KVSender/KVReceiver ── Mooncake/NIXL/Mori 传输引擎
```

## 核心概念

1. **同源**：`third_party/sglang/python/sglang/srt/distributed/parallel_state.py` 第 3 行注明 “Adapted from … vllm … v0.6.4.post1 … parallel_state.py”；`device_communicators/` 下 `pynccl.py`、`pynccl_wrapper.py`、`custom_all_reduce.py`、`shm_broadcast.py`、`cuda_wrapper.py` 等同样标注改编自 vLLM。`monkey_patch_vllm_parallel_state()` 还能把 vLLM 的 `get_tp_group` 等重定向到 SGLang 的实现（复用 vLLM 组件时用）。
2. **更多组**：为 DP-Attention + EP 的 MoE 推理拆出 `attention_tp`、`attn_cp`、`moe_ep`、`moe_tp`、`moe_dp`、`shared_experts_tp`、`dcp` 等组。
3. **DP-Attention**：attention 部分按请求做数据并行（每个 attention-DP rank 自己的 batch 与 KV），MoE 部分按专家 EP；层间要把各 DP rank 的 token 聚到 MoE 组再分回。
4. **AllReduce 后端解析**：与 vLLM 思路一致，但实现为一个“方法名解析器”，并增加了 mscclpp、PCIe IPC、V2 JIT custom AR。
5. **DeepEP 模式自适应**：`DeepEPMode.AUTO` 在 prefill(extend) 用 normal、decode 用 low_latency。
6. **通算重叠在调度层**：TBO 把一个 batch 拆成两个 micro-batch，让一个的 A2A 与另一个的计算重叠；dispatcher 的 `_a/_b` 两阶段 API 就是为此切出的“发起/等待”点。

## 源码导读（路径 + 函数 + 作用）

### 1. 并行组：`third_party/sglang/python/sglang/srt/distributed/parallel_state.py`

| 符号 | 作用 |
|---|---|
| `class GroupCoordinator`（L253） | 字段含 `cpu_group`、`device_group`、`use_pynccl`、`use_mscclpp`、`use_custom_allreduce`、`use_torch_symm_mem_all_reduce`；通信器 `pynccl_comm`、`pymscclpp_comm`、`ca_comm`、`qr_comm`（HIP）、`pcie_ipc_comm`、`torch_symm_mem_comm`、`mq_broadcaster` 及 NPU/HPU/XPU 通信器 |
| `GroupCoordinator.all_reduce`（L693） | world_size==1 直接返回 → CPU 张量走 `sgl_kernel.shm_allreduce` 或 `torch.distributed` → 各非 CUDA 设备通信器 → `torch.compiler.is_compiling()` 时走自定义 op → pynccl 对称内存原地路径 → FlashInfer → `_resolve_outplace_all_reduce_method` → 兜底 `_all_reduce_in_place` |
| `_resolve_outplace_all_reduce_method`（L957） | 依次返回 `"ca"`（且 mscclpp 不接管时）→ `"pcie_ipc"` → `"qr"` → `"pymscclpp"` → `"torch_symm_mem"` → `"pynccl"`（仅 piecewise CUDA graph） |
| `_all_reduce_in_place`（L1073） | pynccl → torch_symm_mem → `torch.distributed` |
| 其他方法 | `reduce_scatter_tensor`、`reduce_scatterv`、`all_to_all_single`、`all_gather_into_tensor`、`all_gatherv`、`broadcast_tensor_dict`、`send_tensor_dict`/`recv_tensor_dict`、`graph_capture`、`fused_allreduce_rmsnorm`（ROCm） |
| 模块级自定义 op | `inplace_all_reduce`、`outplace_all_reduce`、`reg_all_gather_into_tensor`、`reg_reduce_scatter_tensor`、`reg_all_to_all_single` 等，按组名查组 |
| 全局组 | `_WORLD, _TP, _ATTN_TP, _ATTN_CP, _DCP, _MOE_EP, _MOE_TP, _MOE_DP, _SHARED_EXPERTS_TP, _PP, _SELF_PP, _PDMUX_PREFILL_TP_GROUP`；getter 如 `get_moe_ep_group()`、`get_attn_tp_group()` |
| `initialize_model_parallel`（L2561） | 尺寸来自 `get_parallel()`（tp、moe_ep、pp、attn_dp、attn_cp、moe_dp、attn_dcp、shared_experts_tp） |
| `init_model_parallel_group`（L2156） | 组工厂；pynccl 默认开启（NPU/XPU/mooncake 后端除外）；只有 TP 组带 `mq_broadcaster`（`SGLANG_USE_MESSAGE_QUEUE_BROADCASTER`） |
| `"mooncake"` torch 后端 | `MooncakeBackendOptions`，用于弹性 EP 场景 |

启动链：`third_party/sglang/python/sglang/srt/distributed/bootstrap.py` `init_parallel_runtime` → `_set_all_reduce_flags`（`set_custom_all_reduce` / `set_mscclpp` / `set_torch_symm_mem_all_reduce`）→ `_init_parallel_groups` → `init_distributed_environment` + `initialize_model_parallel`。

### 2. `device_communicators/`（`third_party/sglang/python/sglang/srt/distributed/device_communicators/`）

| 文件 | 内容 | vLLM 对应 |
|---|---|---|
| `pynccl.py` / `pynccl_wrapper.py` | `PyNcclCommunicator`、ctypes `NCCLLibrary`；库路径可由 `SGLANG_NCCL_SO_PATH` 指定 | 同名文件 |
| `pynccl_allocator.py` | NCCL 对称内存池（`use_symmetric_memory`、`is_symmetric_memory_enabled`） | 同名文件 |
| `custom_all_reduce.py` | 旧版 `CustomAllreduce`（IPC）+ `dispatch_custom_allreduce()` | `custom_all_reduce.py` |
| `custom_all_reduce_v2.py` | `CustomAllReduceV2`：JIT kernel，PushPlane/PullPlane 两套对称 buffer，算法 `1shot_push / 1shot_pull / 2shot_pull`，可用 multicast 地址；CUDA Graph 输入在捕获后交换 IPC/VMM 句柄写入设备侧指针表 | 无 |
| `pymscclpp.py` | `PyMscclppCommunicator`（`should_mscclpp_allreduce`） | 无 |
| `torch_symm_mem.py` | `TorchSymmMemCommunicator` | `symm_mem.py` |
| `pcie_ipc_ar.py` | `PcieIpcCommunicator`（FlashInfer PCIe IPC） | `flashinfer_pcie_ipc_all_reduce.py` |
| `quick_all_reduce.py` | ROCm QuickReduce | 同名文件 |
| `shm_broadcast.py` | `ShmRingBuffer`、`MessageQueue` | 同名文件 |
| `triton_symm_mem_ag.py` | Triton multimem AllGather | 无 |
| `vocab_gather.py` | `NcclVocabGather`、`NVLinkVocabGather`（LM head 词表维 gather） | 无 |
| `mooncake_transfer_engine.py` | `MooncakeTransferEngine` 封装 | 无（vLLM 在 KV connector 内） |

`dispatch_custom_allreduce()`（`custom_all_reduce.py` L344）：CUDA 上若 `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2`（默认 True）且 `can_use_custom_all_reduce_v2` 通过则用 V2，否则旧版；HIP 上默认用 aiter 的实现（`SGLANG_USE_AITER_AR`）。kernel 源码：AOT 版在 `third_party/sglang/python/sglang/kernels/aot/csrc/allreduce/`（`custom_all_reduce.cu/.cuh`、`quick_all_reduce.*` 等），JIT 版在 `third_party/sglang/python/sglang/kernels/jit/csrc/distributed/`（`custom_all_reduce.cuh`、`all_reduce_fusion.cuh`、`ipc_a2a.cuh`、`nvlink_comm.cuh`）。注意：本 commit 没有顶层 `sgl-kernel/` 目录。

### 3. DP-Attention：`third_party/sglang/python/sglang/srt/layers/dp_attention.py`

- 初始化：`compute_dp_attention_world_info`、`initialize_dp_attention`、`is_dp_attention_enabled`、`get_dp_tp_group`。
- 缓冲：`DpPaddingMode`（`MAX_LEN` / `SUM_LEN`）、`get_global_dp_buffer`、`get_local_dp_buffer`。
- gather：`dp_gather_partial` / `dp_gather_replicate` → `_dp_gather`，三选一：
  - `_dp_gather_via_all_reduce`：每 rank 把自己的 token 写进零填充的全局 buffer 后做 AllReduce（实现简单，字节多 d 倍）；
  - `_dp_gather_via_all_gather`：按 MAX_LEN 填充后 AllGather；
  - `_dp_gather_via_all_gatherv`：变长 AllGatherV（另有 FP8 变体，`SGLANG_ENABLE_DP_GATHER_FP8`）。
- scatter：`dp_scatter` 本地切回自己那段；`dp_reduce_scatter_tensor`（`SGLANG_DP_USE_REDUCE_SCATTER`）以 ReduceScatter 合并“求和 + 分发”。
- TBO 异步版：`dp_gather_partial_async`、`dp_reduce_scatterv_async`、`get_dp_tbo_comm_stream`（独立通信 stream）。

### 4. 层边界通信编排：`third_party/sglang/python/sglang/srt/layers/layer_boundary/`

包注释：“Stage boundaries: layout transport, residual updates, reads, and fusion.” 早期的 `layers/communicator.py`（`LayerCommunicator`）在本 commit 不存在，职能迁到这里。
- `layout.py`：`TokenAxis{ATTN_DP, ATTN_CP, ATTN_TP}`、`Layout`（token 在哪些轴上被切分）、`SumGroup`（哪些组欠一次求和）。
- `ops.py`：实际的搬运原语 `tp_reduce_scatter`、`attn_tp_gather_input`、`attn_tp_reduce_scatter`、`dp_gather`、`dp_gather_sum`、`dp_reduce_scatterv`、`sum_output`、`moe_cp_gather`。
- `boundary.py`：`bind_entry` / `bind_exit` / `tbo_split_moves`；`fusions/allreduce.py`：AllReduce + Norm 融合。
- CLI `--boundary-reduction {auto,ar,rs,rsv,rs+rsv}` 选择 FFN 边界用 AllReduce 还是 ReduceScatter(V)。
- 思想：把“上一层输出的切分布局”与“下一层需要的布局”做差，自动推导需要的集合通信（AR / RS / AG / gatherv），而非每个模型手写。

### 5. MoE EP

- 后端枚举 `MoeA2ABackend`（`third_party/sglang/python/sglang/srt/layers/moe/utils.py`）：`none, deepep, deepep_v2, mooncake, nixl, mori, pplx, flashinfer, megamoe, flashinfer_megamoe, ascend_fuseep, ascend_tp` 等；CLI `--moe-a2a-backend`。
- `DeepEPMode`（同文件 L253）：`normal / low_latency / auto`，`resolve(is_extend_in_batch)`；CLI `--deepep-mode`。
- `layers/moe/token_dispatcher/deepep.py`：
  - `DeepEPBuffer`：包装 `deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)`；
  - `_DeepEPDispatcherImplNormal`：`get_dispatch_layout` → `dispatch` → `combine`；
  - `_DeepEPDispatcherImplLowLatency`：`low_latency_dispatch` / `low_latency_combine`，`return_recv_hook` 为真时用 hook 等待，否则 `event.current_stream_wait()`；
  - `DeepEPDispatcher`：AUTO 下两种实现都建，`dispatch_a/dispatch_b/combine_a/combine_b` 分两段，便于在中间插入另一 micro-batch 的计算。
  - `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（默认 128）决定 LL buffer 容量。
- 其他 dispatcher：`deepep_v2.py`、`mooncake.py`、`nixl.py`、`moriep.py`、`pplx.py`、`flashinfer.py`。
- 重叠：`third_party/sglang/python/sglang/srt/batch_overlap/two_batch_overlap.py`（`TboForwardBatchPreparer`、`MaybeTboDeepEPDispatcher`、`model_forward_stages`）、`operations.py`（`execute_overlapped_operations`、`YieldOperation`）、`single_batch_overlap.py`。CLI `--enable-two-batch-overlap`、`--enable-single-batch-overlap`。
- EPLB：`third_party/sglang/python/sglang/srt/eplb/`（`EPLBManager`、`expert_distribution.py` 统计各专家负载）。

### 6. PD 分离：`third_party/sglang/python/sglang/srt/disaggregation/`

- 抽象（`base/conn.py`）：`KVArgs`、`KVPoll`（Failed/Bootstrapping/WaitingForInput/Transferring/Success）、`BaseKVManager`、`BaseKVSender`（`init/send/poll/abort`）、`BaseKVReceiver`（`init/send_metadata/poll`）、`BaseKVBootstrapServer`。
- 公共层（`common/conn.py`）：`CommonKVManager`（ZMQ PULL socket）、`CommonKVBootstrapServer`（aiohttp，路由 `/route`、`/register_dp_rank`、`/query_dp_ranks`、`/health`）。
- 后端：`mooncake/`（`MooncakeKVManager`，通过 `engine.batch_transfer_sync` 批量 RDMA 传输）、`nixl/`、`mori/`、`ascend/`、`fake/`；选择 `--disaggregation-transfer-backend`（默认 mooncake），角色 `--disaggregation-mode {null,prefill,decode}`，`--disaggregation-bootstrap-port`（8998），`--disaggregation-ib-device`。
- 协作流程：prefill 侧 `PrefillBootstrapQueue` 建 sender，调度循环 `event_loop_normal_disagg_prefill` / `event_loop_overlap_disagg_prefill` 中 `send_kv_chunk`；decode 侧 `DecodePreallocQueue` 建 receiver 并预分配 KV，`DecodeTransferQueue.pop_transferred` 取完成的请求进入 decode。bootstrap server 负责让 D 找到对应 P 的 rank 与地址（`managers/disagg_service.py` `start_disagg_service`）。

### 7. 与 vLLM 的异同

| 方面 | vLLM | SGLang |
|---|---|---|
| 组抽象 | `GroupCoordinator` + 平台 `DeviceCommunicator`（`CudaCommunicator`） | `GroupCoordinator` 直接持有各通信器（无单独 CudaCommunicator 层） |
| rank 布局 | `ExternalDP x DP x PP x PCP x TP` | 由 `get_parallel()` 的多维尺寸生成 attention/MoE 两套组 |
| 小消息 AR | NCCL symm mem、custom AR（1stage/2stage）、torch symm mem、FlashInfer、QuickReduce | custom AR V2（1shot_push/1shot_pull/2shot_pull + multicast）、旧版 CA、mscclpp、PCIe IPC、torch symm mem、QuickReduce |
| AR 融合 | FlashInfer AR 融合 | `layer_boundary/fusions/allreduce.py`、`--flashinfer-allreduce-fusion-backend` |
| EP 选择 | `--all2all-backend`，manager + PrepareAndFinalize | `--moe-a2a-backend` + `--deepep-mode`，Dispatcher（`_a/_b` 两段） |
| HT/LL 切换 | 部署时固定选一种 | AUTO 按 batch 类型（extend/decode）切换 |
| 重叠 | MoE `prepare_async/finalize_async` | TBO / SBO（batch_overlap/） |
| DP-Attention | DP 组 + EP 组 | `dp_attention.py` 专门的 gather/scatter（AR / AG / AGV 三选一） |
| PD 分离 | `KVConnectorBase_V1`（scheduler/worker 双角色，按层 load/save） | `KVSender/KVReceiver` + bootstrap server + 调度循环 mixin |
| 传输引擎 | NIXL（pull/push）、Mooncake、LMCache、MoRIIO… | Mooncake（默认）、NIXL、Mori、Ascend |

## 性能要点与常见坑

- `_dp_gather_via_all_reduce` 的字节量是 AllGather 的约 d 倍（零填充 + 求和），只在消息很小或配合 CP 时合适；大 batch 下应走 all_gather(v)。
- AUTO 模式需要两套 DeepEP buffer（normal 的 NVL/RDMA buffer + LL 的 RDMA buffer），显存开销更大；LL 的 buffer 按 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 预留，decode batch 超过它会出错或需要拆分。
- TBO 需要两个 micro-batch 的 token 量足够均衡（`--tbo-token-distribution-threshold`，默认 0.48），否则重叠收益有限。
- custom AR V2 与 mscclpp 在判断上有优先级耦合（mscclpp 接管时 CA 让路），排查“走了哪个后端”要看 `_resolve_outplace_all_reduce_method` 的条件。
- PD 分离中 bootstrap 超时（`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`，默认 300）、心跳（`SGLANG_DISAGGREGATION_HEARTBEAT_*`）是常见故障点，与 NCCL 无关，问题通常在 RDMA 设备选择（`--disaggregation-ib-device`）或网络可达性。
- DeepEP 版本：SGLang 的 `deepep.py` 调用 V1 风格 `Buffer`/`low_latency_dispatch`，与本地 DeepEP V2.5 的 `EPBuffer` 不匹配；`deepep_v2.py` 对应 V2 接口（见 [04](04-deepep.md)）。

## 面试自测

1. 问：SGLang 和 vLLM 的通信层是什么关系？
   要点：SGLang 的 `parallel_state.py` 与多个 device communicator 改编自 vLLM v0.6.4.post1；之后各自演进：SGLang 加了 mscclpp、custom AR V2、DP-Attention、layer_boundary、TBO；vLLM 加了 CudaCommunicator 层、NCCL symm mem、多种 all2all manager、KVConnector v1。
2. 问：DP-Attention 为什么需要额外通信？有哪几种实现？
   要点：attention 各 DP rank 处理不同请求，而 MoE/dense FFN 需要组内全部 token；层间要 gather 再 scatter。实现：零填充后 AllReduce、按最大长度 AllGather、变长 AllGatherV；回程可用 ReduceScatter。
3. 问：为什么 DeepEP 要分 normal 和 low_latency 两种模式？SGLang 如何选？
   要点：prefill token 多，追求带宽，用分层 NVLink+RDMA 转发；decode token 少，追求延迟与 CUDA Graph，用固定 buffer 的直接 RDMA；SGLang `DeepEPMode.AUTO` 按 batch 是否 extend 自动切换。
4. 问：TBO（two-batch overlap）是怎么隐藏 A2A 的？
   要点：把 batch 切两半，算子序列切成 stage；A 的 dispatch 发出（`dispatch_a`）后去算 B 的 attention，再回来 `dispatch_b` 等待；通信 kernel 与计算 kernel 在不同 stream/SM 上并行。前提是通信占用 SM 少、两半负载均衡。
5. 问：SGLang PD 分离中，decode 如何知道向哪个 prefill 拿 KV？
   要点：prefill 实例运行 bootstrap server；请求带 bootstrap 地址/房间；decode 的 receiver 通过 bootstrap 查询 P 侧 rank 与地址并 `send_metadata`（告知自己的 KV 地址），P 侧 sender 用 Mooncake/NIXL 发起 RDMA 传输，双方 `poll` 状态到 Success。

## 待深入

- 读 `third_party/sglang/python/sglang/srt/layers/layer_boundary/ops.py` 与 `layout.py`，画出 DeepSeek 模型一层内各边界选择的集合通信。
- 读 `third_party/sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh`，与 vLLM `csrc/custom_all_reduce.cuh` 的 1stage/2stage 对比 push/pull 设计差异。
- 读 `third_party/sglang/python/sglang/srt/disaggregation/mooncake/` 中 `MooncakeKVManager`，确认按层还是按请求整体发送、是否与 prefill 计算重叠。
- 待确认：mscclpp 的 allreduce 算法与阈值；确认方式：读 `third_party/sglang/python/sglang/srt/distributed/device_communicators/pymscclpp.py` `should_mscclpp_allreduce`，并克隆 microsoft/mscclpp 查 kernel。
