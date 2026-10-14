# vLLM 分布式通信层导读

> 层次：上层 · 来源：third_party/vllm@02653fa、third_party/nccl@3e46c08、ai-infra-book 第 6 章（6.4.4）与 `case-studies/collective-paths-and-diagnosis.md`

## 一句话

vLLM 用 `GroupCoordinator` 把每个并行组包装成“gloo CPU 组 + NCCL 设备组 + 平台通信器”，TP AllReduce 按消息大小和拓扑在 NCCL symm mem / custom IPC allreduce / torch symm mem / pynccl / `torch.distributed` 之间动态挑选；EP 的 dispatch/combine 交给可插拔的 all2all manager（DeepEP、Mori、NIXL-EP、FlashInfer…）；PD 分离的 KV 传输则走独立的 KVConnector 体系。

## 在调用链中的位置

```text
RowParallelLinear.forward / MoE runner
  └─ tensor_model_parallel_all_reduce()          communication_op.py
      └─ get_tp_group().all_reduce()             parallel_state.py GroupCoordinator
          └─ torch.ops.vllm.all_reduce(...)      direct_register_custom_op 注册（torch.compile/CUDA Graph 友好）
              └─ _all_reduce_out_place → device_communicator.all_reduce
                  └─ CudaCommunicator.all_reduce  cuda_communicator.py：逐级判断后端
                      ├─ NCCL symm mem（pynccl + ncclMemAlloc/窗口注册）
                      ├─ QuickAllReduce（ROCm）/ FlashInfer / AITER
                      ├─ CustomAllreduce（IPC buffer + 自研 kernel）
                      ├─ SymmMemCommunicator（torch symm_mem multimem / two-shot）
                      ├─ PyNcclCommunicator.all_reduce（ctypes → ncclAllReduce）
                      └─ torch.distributed.all_reduce（兜底，ProcessGroupNCCL）
MoE: FusedMoEPrepareAndFinalize.prepare/finalize → all2all_manager.get_handle() → deep_ep.Buffer 等
PD:  KVConnectorBase_V1（scheduler 侧 + worker 侧）→ NIXL / Mooncake / LMCache ...
```

## 核心概念

1. **双组设计**：每个 `GroupCoordinator` 同时持有 `cpu_group`（gloo，用于元数据、barrier、对象广播、pynccl unique id 交换）和 `device_group`（NCCL，用于张量）。控制面与数据面分离，避免小对象走 NCCL、也避免 CPU 同步阻塞 GPU stream。
2. **rank 布局**：`ExternalDP x DP x PP x PCP x TP`，TP 最内层，保证 TP 组是连续 rank（通常同机 NVLink）。EP 组由 DP×PCP×TP 拼成（只对 MoE 模型创建）。
3. **自定义算子包装**：AllReduce/AllGather/ReduceScatter 被注册成 `torch.ops.vllm.*`，带 fake 实现，使 `torch.compile` 能把通信当作不透明节点、CUDA Graph 能捕获。
4. **多后端择优**：NCCL 是“正确性兜底 + 大消息最优”，自研/第三方 kernel 抢小消息和特殊拓扑。
5. **pynccl**：不经 PyTorch ProcessGroup，直接 ctypes 加载 `libnccl.so.2`，自己管理 `ncclComm_t`、自己选 stream。好处：可以用与 PyTorch 不同的 NCCL 版本、可被 CUDA Graph 捕获、可调用 PyTorch 未暴露的新 API（窗口注册、suspend/resume）。
6. **custom allreduce**：每个 rank 分配一块可 IPC 共享的 buffer，用 `cudaIpc*` 句柄映射对端地址，kernel 直接通过 NVLink P2P load 对端数据并归约；用 GPU 内存里的 `Signal` 做跨卡 barrier。没有 proxy 线程、没有 channel 建立，延迟主要是一次 kernel。

## 源码导读（路径 + 函数 + 作用）

### 1. 并行组：`third_party/vllm/vllm/distributed/parallel_state.py`

| 符号 | 作用 |
|---|---|
| `class GroupCoordinator`（L426） | 字段 `rank, ranks, world_size, local_rank, rank_in_group, cpu_group, device_group, device_communicator, mq_broadcaster, unique_name, use_custom_op_call` |
| `GroupCoordinator.__init__` | 对每组 rank 调 `torch.distributed.new_group(ranks, backend=...)` 建设备组，再以 `backend="gloo"` 建 CPU 组；`VLLM_DISTRIBUTED_USE_SPLIT_GROUP` 时改走 `_create_subgroups_split_group`；`world_size > 1` 时用 `current_platform.get_device_communicator_cls()` 实例化设备通信器（CUDA 上为 `CudaCommunicator`，见 `third_party/vllm/vllm/platforms/cuda.py`） |
| `all_reduce`（L724） | `world_size==1` 直接返回；否则 `torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)` 或直接 `_all_reduce_out_place` |
| `all_gather` / `all_gatherv` / `reduce_scatter` / `reduce_scatterv` / `gather` / `broadcast` | 其余集合通信，均委托 device communicator |
| `broadcast_tensor_dict`（L985） | 元数据经 CPU 组 `broadcast_object`，张量按所在设备分别走 `cpu_group` / `device_group` |
| `graph_capture`（L673） | 在旁路 stream 上进入捕获，并嵌套 `ca_comm.capture()` 等，让 custom AR 在退出时注册 graph buffer |
| `dispatch` / `combine` / `dispatch_router_logits`（L1410 起） | MoE 路由通信，委托 device communicator 的 all2all manager |
| `all_reduce`/`reduce_scatter`/`all_gather` 模块级函数 + `direct_register_custom_op`（L398/404/410） | 注册为 `torch.ops.vllm.*`；`direct_register_custom_op` 定义在 `third_party/vllm/vllm/utils/torch_utils.py` |
| 全局组 `_WORLD, _TP, _PP, _DP, _EP, _EPLB, _DCP, _PCP, _ETP, _ENGRAM_DP, _INNER_DP_WORLD` | 对应 `get_tp_group()` 等 getter |
| `init_distributed_environment`（L1814） | DP>1 或多节点时把 rank 偏移为 `dp_rank*world_size + rank`，调用 `torch.distributed.init_process_group`，建 `_WORLD` |
| `initialize_model_parallel`（L2056） | `torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)`，对各维 transpose 到末尾得到各组；TP/DCP 组带 `use_message_queue_broadcaster=True`；EP 组带 `use_all2all` |

封装函数：`third_party/vllm/vllm/distributed/communication_op.py` 的 `tensor_model_parallel_all_reduce` / `_all_gather` / `_reduce_scatter` / `_gather` / `broadcast_tensor_dict`，都是 `get_tp_group()` 的薄包装。

### 2. 后端选择：`third_party/vllm/vllm/distributed/device_communicators/cuda_communicator.py`

`CudaCommunicator.__init__`：只有 `unique_name` 前缀为 `tp` 的组才启用 custom AR / symm mem / FlashInfer 等 TP 专用后端；创建 `pynccl_comm`（传入的是 **CPU 组**）、`symm_mem_comm`、`fi_ar_comm`、`fi_pcie_ipc_ar_comm`、`aiter_ar_comm`、`ca_comm`、`qr_comm`（ROCm）。

`CudaCommunicator.all_reduce`（L362 起）判断顺序：

1. NCCL symmetric memory：`should_nccl_symm_mem_allreduce(...)` 通过时调 `torch.ops.vllm.all_reduce_symmetric_with_copy`；
2. `qr_comm.should_quick_allreduce`（ROCm QuickReduce，支持量化通信）；
3. FlashInfer PCIe IPC；4. FlashInfer AllReduce；5. AITER（ROCm）；
6. `ca_comm.should_custom_ar` → `custom_all_reduce`；
7. ROCm batch-invariant 路径；
8. `symm_mem_comm.should_use_symm_mem`；
9. `pynccl_comm.all_reduce`；
10. `torch.distributed.all_reduce(out, group=self.device_group)`。

阈值表在 `third_party/vllm/vllm/distributed/device_communicators/all_reduce_utils.py`：
- `CUSTOM_ALL_REDUCE_MAX_SIZES`（按 compute capability × world size，例如 CC 9.0 下 ws=8 为 256K、ws=2 为 64M）；
- `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`（`min_world_size` 4；ws=4 时 16K~512K、ws=8 时 16K~128K 区间优先 custom AR；`always_use_above_world_size` 8）；
- `should_nccl_symm_mem_allreduce`、`gpu_p2p_access_check`、`can_actually_p2p`。

### 3. pynccl：ctypes 直调 NCCL

`third_party/vllm/vllm/distributed/device_communicators/pynccl_wrapper.py`
- `class ncclUniqueId(ctypes.Structure)`：`internal` 为 128 字节，对应 NCCL 的 `ncclUniqueId`。
- `class NCCLLibrary`：`ctypes.CDLL(so_file or find_nccl_library())`，按路径缓存；`find_nccl_library` 在 `third_party/vllm/vllm/utils/nccl.py`，读 `VLLM_NCCL_SO_PATH`，否则用 `libnccl.so.2`（HIP 为 `librccl.so.1`）。
- 用 `Function(name, restype, argtypes)` 声明的导出：`ncclGetErrorString, ncclGetVersion, ncclGetUniqueId, ncclCommInitRank, ncclCommQueryProperties, ncclAllReduce, ncclReduce, ncclReduceScatter, ncclAllGather, ncclBroadcast, ncclSend, ncclRecv, ncclGroupStart, ncclGroupEnd, ncclCommDestroy, ncclCommAbort, ncclCommSuspend, ncclCommResume, ncclCommWindowRegister, ncclCommWindowDeregister`。这些都能在 `third_party/nccl/src/nccl.h.in` 找到对应声明。

`third_party/vllm/vllm/distributed/device_communicators/pynccl.py` `class PyNcclCommunicator`（L91）
- 断言传入的 group 不是 NCCL 组（必须是 gloo/StatelessProcessGroup）；`world_size==1` 或 `VLLM_DISABLE_PYNCCL` 时禁用。
- rank0 `ncclGetUniqueId`，经 `dist.broadcast(tensor, src=ranks[0], group=group)`（CPU 组）发给其他 rank；`_init_comm` 调 `ncclCommInitRank`，再做一次 warmup `all_reduce` 并同步（ROCm 可用 `defer_comm_warmup_on_rocm` 推迟）。
- `all_reduce(in_tensor, out_tensor=None, op=ReduceOp.SUM, stream=None)`：默认用 `current_stream()`——通信与计算在同一条 stream 上串行，天然被 CUDA Graph 捕获，不需要 event 同步。
- 其他：`all_gather, reduce_scatter, reduce, broadcast, send, recv, group_start/group_end, batch_isend_irecv, register_comm_window(_raw)`；`register_nccl_symmetric_ops`（L62）注册 `all_reduce_symmetric_with_copy`。

`third_party/vllm/vllm/distributed/device_communicators/pynccl_allocator.py`：内联 C 源码调 `ncclMemAlloc`，包成 `CUDAPluggableAllocator` + `torch.cuda.MemPool`；`nccl_symm_mem_context` 中对分配的内存做窗口注册（`register_comm_window_raw` → `ncclCommWindowRegister`），使 NCCL 能走对称内存（symmetric kernel）路径。开关 `VLLM_USE_NCCL_SYMM_MEM`（`is_symmetric_memory_enabled`）。

### 4. custom allreduce：何时替代 NCCL

`third_party/vllm/vllm/distributed/device_communicators/custom_all_reduce.py` `class CustomAllreduce`
- `_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]`，默认 `max_size` 8 MiB（启用 symm mem 时受 `CUSTOM_ALL_REDUCE_MAX_SIZES` 限制）。
- 初始化时禁用条件：库不可用、world size 不支持、超过 2 张且非全互联（`current_platform.is_fully_connected` 判定 NVLink 全连接）、P2P 不可达（`VLLM_SKIP_P2P_CHECK` 默认 `"1"`，信任 `torch.cuda.can_device_access_peer`；否则跑 `gpu_p2p_access_check`）、多节点组需要 MNNVL multicast。
- IPC 准备：`create_shared_buffer` 分配 `meta_ptrs`（`meta_size() + max_size`，放 `Signal` 与临时 buffer）和 `buffer_ptrs`，交换 IPC 句柄；`ops.init_custom_ar(...)`、`ops.register_buffer`。
- `should_custom_ar`（L499）：`world_size <= 8`；dtype ∈ {fp32, fp16, bf16}；字节数 16 对齐；weak contiguous；`world_size == 2 or fully_connected`；且 `inp_size < max_size`。
- CUDA Graph：`capture()` 退出时 `register_graph_buffers()`——收集图中用到的输入地址的 IPC 元数据（`ops.get_graph_buffer_ipc_meta`）、`broadcast_object_list` 交换、`ops.register_graph_buffers`。eager 模式则先拷进预注册 buffer 再归约。

kernel：`third_party/vllm/csrc/custom_all_reduce.cuh`
- `cross_device_reduce_1stage`：`barrier_at_start` 后每个 rank 直接读所有对端 buffer 求和写结果（one-shot，读量 = (n−1)·M，延迟最低）；注释强调不重排地址以保证各 rank 位级一致。
- `cross_device_reduce_2stage`：stage1 reduce-scatter 到各自 tmp buffer，barrier，stage2 allgather（two-shot，字节量更省，适合更大消息）。
- 选择（`REDUCE_CASE` 宏）：ws==2 用 1stage；全互联下 `(ws<=4 && bytes<512KiB) || (ws<=8 && bytes<256KiB)` 用 1stage，否则 2stage；可被 C++ 侧环境变量 `VLLM_CUSTOM_ALLREDUCE_ALGO`（`1stage|oneshot|2stage|twoshot`）强制。
- barrier/`Signal`/`RankData` 定义在 `third_party/vllm/csrc/custom_collective_common.cuh`；绑定在 `third_party/vllm/csrc/custom_all_reduce.cu`。

其他 AllReduce 后端：
- `symm_mem.py` `SymmMemCommunicator`：PyTorch symmetric memory rendezvous，按 CC 与 world size 用 `torch.ops.symm_mem.multimem_all_reduce_`（NVSwitch multicast/NVLS）或 `two_shot_all_reduce_`；开关 `VLLM_ALLREDUCE_USE_SYMM_MEM`（默认 1）。
- `quick_all_reduce.py` `QuickAllReduce`：仅 ROCm gfx94/95，支持 INT8/INT6/INT4 等量化通信（`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`）。
- `flashinfer_all_reduce.py`、`flashinfer_pcie_ipc_all_reduce.py`、`aiter_custom_all_reduce.py`：第三方 kernel 包装；`VLLM_ALLREDUCE_USE_FLASHINFER`、`VLLM_FLASHINFER_ALLREDUCE_BACKEND`。

### 5. 控制面广播：`shm_broadcast.py`

`third_party/vllm/vllm/distributed/device_communicators/shm_broadcast.py` `class MessageQueue`：一写多读；本机 reader 走共享内存环形 buffer（`ShmRingBuffer`），远端或超大消息走 zmq。用于 `GroupCoordinator.mq_broadcaster`（TP 组把调度输入从 driver 广播给 worker），也用于 `third_party/vllm/vllm/v1/executor/multiproc_executor.py` 的 `rpc_broadcast_mq`。意义：每个 decode step 的调度元数据广播不能走 NCCL（CPU 对象）也不能太慢。

### 6. EP all2all 后端

- 配置：`third_party/vllm/vllm/config/parallel.py` 中 `All2AllBackend` Literal 与 `all2all_backend` 字段，默认 `"allgather_reducescatter"`；`"pplx"`、`"naive"` 已移除并重映射。CLI 为 `--all2all-backend`（`third_party/vllm/vllm/engine/arg_utils.py`）。注意：不存在 `VLLM_ALL2ALL_BACKEND` 环境变量。
- 选择：`CudaCommunicator.__init__` 中按 `all2all_backend` 实例化：

| 后端字符串 | 类（`all2all.py`） | 说明 |
|---|---|---|
| `allgather_reducescatter` | `AgRsAll2AllManager` | dispatch=`all_gatherv`、combine=`reduce_scatterv`，纯 NCCL 实现，通用但浪费带宽 |
| `deepep_high_throughput` | `DeepEPHTAll2AllManager` | `deep_ep.Buffer(low_latency_mode=False)`，NVL buffer 由 `VLLM_DEEPEP_BUFFER_SIZE_MB` 决定 |
| `deepep_low_latency` | `DeepEPLLAll2AllManager` | `low_latency_mode=True`，RDMA buffer 用 `get_low_latency_rdma_size_hint` |
| `deepep_v2` | `DeepEPV2All2AllManager` | 基于 DeepEP v2 `ElasticBuffer` 统一接口 |
| `mori_high_throughput` / `mori_low_latency` | `MoriAll2AllManager` | AMD Mori |
| `nixl_ep` | `NixlEPAll2AllManager` | |
| `moonep` | `MoonEPAll2AllManager` | |
| `flashinfer_nvlink_two_sided` / `_one_sided` | `FlashInferNVLink{Two,One}SidedManager` | 单 NVLink 域（含 MNNVL） |
| `passthrough` | `PassThroughAll2AllManager` | |

- 基类 `All2AllManagerBase` 在 `third_party/vllm/vllm/distributed/device_communicators/base_device_communicator.py`；DeepEP 系列只提供 `get_handle(kwargs)` 返回缓存的 buffer，真正的 dispatch/combine 由 MoE 层的 PrepareAndFinalize 调用：
  - `third_party/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py` `maybe_make_prepare_finalize` 按 `moe.use_deepep_ht_kernels` 等标志构造对应类；
  - `prepare_finalize/deepep_ht.py`（`DeepEPHTPrepareAndFinalize`：`get_dispatch_layout` → `dispatch` → `combine`）；
  - `prepare_finalize/deepep_ll.py`（`DeepEPLLPrepareAndFinalize`：`low_latency_dispatch(..., return_recv_hook=True)` → `low_latency_combine`）；
  - `modular_kernel.py` `FusedMoEPrepareAndFinalize` 定义 `prepare / prepare_async / finalize / finalize_async`，`FusedMoEKernelModularImpl._prepare/_finalize` 调用它们——async 版本就是给通算重叠留的口子。

### 7. KV transfer / connector（PD 分离）

- `third_party/vllm/vllm/distributed/kv_transfer/README.md` 描述的 “KV pipe / lookup buffer / connector” 三层抽象是早期（V0）设计，现行接口以 v1 为准。
- 工厂：`kv_connector/factory.py` `KVConnectorFactory.register_connector` 懒加载注册，名称包括 `NixlConnector, NixlPullConnector, NixlPushConnector, MooncakeConnector, MooncakeStoreConnector, LMCacheConnectorV1, LMCacheMPConnector, MultiConnector, OffloadingConnector, MoRIIOConnector, FlexKVConnectorV1, HF3FSKVConnector` 等。本 commit 没有 `P2pNcclConnector`。
- 基类：`kv_connector/v1/base.py` `KVConnectorBase_V1`，`KVConnectorRole` 分 SCHEDULER / WORKER：
  - worker 侧：`start_load_kv`、`wait_for_layer_load(layer_name)`、`save_kv_layer`、`wait_for_save`——按层异步加载/保存，可与逐层前向流水；
  - scheduler 侧：`get_num_new_matched_tokens`（远端有多少 token 的 KV 可复用）、`update_state_after_alloc`、`build_connector_meta`；
  - 其他：`register_kv_caches`、`get_finished`、`get_handshake_metadata`。
- 角色配置：`third_party/vllm/vllm/config/kv_transfer.py` 的 `kv_role`：`kv_producer` / `kv_consumer` / `kv_both`。
- NIXL 实现：`kv_connector/v1/nixl/` 分 pull/push 两套 worker——`pull_worker.py` `_read_blocks` 中用 `make_prepped_xfer("READ", ...)`（D 端主动从 P 端 RDMA READ）、`push_worker.py` 用 `"WRITE"`；`base_worker.py` `add_remote_agent` 做对端元数据握手；`tp_mapping.py` 处理 P/D 两端 TP 不同的映射。
- 这条路径**不经过 NCCL**：对端是另一个 vLLM 实例，成员动态变化，传输按 block 寻址。

## 性能要点与常见坑

- custom AR 只在 NVLink 全互联（或 2 卡）时启用；PCIe 机器上会自动退回 NCCL。排查“为什么没走 custom AR”：看 `_log_all_reduce_backend_selection` 的日志、P2P 检查、world size 是否在支持列表、消息是否超过 `max_size`。
- custom AR 和 pynccl 都在当前 stream 上执行，没有独立通信 stream：好处是 CUDA Graph 与顺序语义简单，代价是不能自动与计算重叠（重叠要靠 MoE async prepare/finalize 或 DBO 等上层机制）。
- pynccl 与 `torch.distributed` 的 NCCL 是两个独立 communicator（不同 `ncclComm_t`），两者都会占用 NCCL 资源（channel buffer、proxy 线程）；`VLLM_NCCL_SO_PATH` 指定的库与 PyTorch 自带 NCCL 版本可能不同，排障时要确认实际加载的是哪一个。
- 多实例 DP + EP 时 `init_distributed_environment` 会重写 rank/world size，日志里的 rank 与单实例语义不同。
- NCCL symm mem 依赖 `ncclMemAlloc` 分配的内存和窗口注册，普通 `cudaMalloc` 张量需要 `all_reduce_symmetric_with_copy` 先拷贝。
- DeepEP 版本：vLLM 同时调用 V1 风格 `deep_ep.Buffer` / `low_latency_dispatch` 和 V2 的 `deep_ep.ElasticBuffer`；而本地 `third_party/DeepEP@93eb6eb` 已是 V2.5（`EPBuffer`），API 不匹配（见 [04](04-deepep.md)）。

## 面试自测

1. 问：vLLM 为什么在 gloo CPU 组上初始化 pynccl？
   要点：只需交换 128 字节 unique id，CPU 组可靠且不依赖已有 NCCL comm；pynccl 自己 `ncclCommInitRank`，与 PyTorch 的 NCCL comm 解耦，便于换版本、CUDA Graph 捕获、调用新 API。
2. 问：custom allreduce 比 NCCL 快在哪里？什么情况下不用它？
   要点：IPC 映射对端 buffer，一个 kernel 内用 NVLink load 直接归约，GPU 内存 flag barrier，无 proxy/协议开销；one-shot 读全部对端，延迟最低。不用：非全互联 PCIe、>8 卡、消息超过 `max_size`、dtype/对齐不满足；大消息 NCCL ring/NVLS 带宽更优。
3. 问：one-shot 与 two-shot 的权衡？
   要点：one-shot 每卡读 (n−1)·M，1 次 barrier；two-shot 先 RS 再 AG，每卡读写约 2(n−1)M/n，2 次 barrier。小消息 one-shot（延迟），大消息 two-shot（带宽）；vLLM 阈值 ws≤4 <512 KiB、ws≤8 <256 KiB。
4. 问：CUDA Graph 下 custom AR 的难点是什么？
   要点：图里的输入地址在捕获时才确定，必须事后收集这些地址的 IPC 句柄并交换注册（`register_graph_buffers`），重放时 kernel 才能访问对端对应地址；eager 下用拷贝到预注册 buffer 规避。
5. 问：vLLM 的 EP 后端怎么选？HT 和 LL 各适合什么？
   要点：`--all2all-backend` 配置；`allgather_reducescatter` 通用兜底；`deepep_high_throughput` 适合 prefill 大 batch（NVLink+RDMA 分层转发）；`deepep_low_latency` 适合 decode 小 batch（固定 buffer、可 CUDA Graph、recv hook 重叠）。
6. 问：PD 分离 NIXL connector 是 push 还是 pull？各有什么取舍？
   要点：两种都有；pull 由 D 端在分配好 KV block 后 RDMA READ，D 端掌握显存分配节奏；push 由 P 端 WRITE，P 端可尽早释放显存但需要先知道 D 端地址。

## 待深入

- 待确认：`torch.ops.symm_mem.multimem_all_reduce_` 的底层实现（是否依赖 NVLS multicast 对象）；确认方式：克隆 pytorch/pytorch 后查 `torch/csrc/distributed/c10d/symm_mem/`。
- 读 `third_party/vllm/vllm/distributed/device_communicators/all2all.py` 中 `DeepEPLLAll2AllManager` 的 buffer 尺寸计算，与 DeepEP 文档对照。
- 读 `third_party/ai-infra-book/case-studies/collective-paths-and-diagnosis.md`，对比书中记录的 vLLM custom AllReduce 演进与本 commit 的差异。
- 读 `third_party/vllm/vllm/distributed/eplb/` 与 `elastic_ep/`，理解专家重排和弹性扩缩时的通信重建。
