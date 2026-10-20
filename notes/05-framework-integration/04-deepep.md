# DeepEP：MoE dispatch/combine 专用通信库

> 层次：上层（跨中层） · 来源：third_party/DeepEP@b5a33ee（V2.5，`deep_ep/__init__.py` 中 `__version__ = '2.5.0'`）、上游 DeepEP `hybrid-ep` 分支@10d4dd73（历史版本，未克隆；含 V1 源码与 Hybrid-EP）、third_party/nccl@3e46c08（GIN / device API / `contrib/nccl_ep`）、third_party/vllm@02653fa 与 third_party/sglang@d2a4364（V1 API 调用点）、ai-infra-book 第 6、7、9 章

> **kernel 级细节已移到 [09-ep-kernels](../09-ep-kernels/README.md)**：V2.5 的调用链与公共设施见 [02](../09-ep-kernels/02-deepep-v25-overview.md)，direct 模式见 [03](../09-ep-kernels/03-deepep-v25-direct.md)，hybrid 模式见 [04](../09-ep-kernels/04-deepep-v25-hybrid.md)。本篇保留框架视角与版本演进。

## 一句话

DeepEP 把 MoE 的“按路由把 token 发给专家（dispatch）+ 把专家输出加权送回（combine）”做成 GPU 端发起的专用 All-to-All kernel：V1 用 NVSHMEM（normal：NVLink+RDMA 分层转发；low-latency：IBGDA 直连 + recv hook），本地的 V2.5 已改用 NCCL GIN 后端并把两种模式统一进 `EPBuffer`。

## 重要版本说明（先读）

- `third_party/DeepEP@b5a33ee` 是 **V2.5**。README 明确：“Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation. NVSHMEM is no longer a dependency”，并且 “EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported”。
- V1 源码只在上游 `hybrid-ep` 分支（@10d4dd73，https://github.com/deepseek-ai/DeepEP/tree/10d4dd7377d5bce900fbb4b80cce863764892b95）：`csrc/kernels/intranode.cu`、`internode.cu`、`internode_ll.cu`、`ibgda_device.cuh`、`deep_ep/buffer.py` 都在其中，另含 NVIDIA 的 Hybrid-EP（`csrc/hybrid_ep/`）。2026-10-20 起本仓库只保留最新主线，不再克隆该分支；下文 `hybrid-ep@10d4dd73/<路径>` 指该分支上的文件。V1 LL / normal 在 V2.5 中分别对应 direct / hybrid 模式，见 [09/05 的"V1 与 Hybrid-EP 在 V2.5 中变成了什么"](../09-ep-kernels/05-lineage-and-selection.md)。
- 框架侧版本错配：vLLM/SGLang 仍在调用 V1 `deep_ep.Buffer`（HT/LL）与 V2 早期的 `deep_ep.ElasticBuffer`（`third_party/vllm/vllm/distributed/device_communicators/all2all.py` `DeepEPV2All2AllManager`），而 V2.5 已改名为 `EPBuffer`。ai-infra-book 也记录了这一点（`third_party/ai-infra-book/manuscripts/09-分布式推理.md` 9.4.3：“V2 的公开接口使用 ElasticBuffer，底层采用 NCCL Gin……并移除了 V1 中不占用 SM、直接经 RDMA 收发的低延迟 EP 通信路径”）。

## 在调用链中的位置

```text
MoE 层（router 输出 topk_idx/topk_weights）
  vLLM:   fused_moe/prepare_finalize/deepep_ht.py | deepep_ll.py | deepep_v2.py
  SGLang: layers/moe/token_dispatcher/deepep.py | deepep_v2.py
        │  dispatch(x, topk_idx, ...) → 专家 GEMM（grouped GEMM）→ combine(expert_out, handle)
        ▼
DeepEP Buffer（V1: Buffer / V2: ElasticBuffer / V2.5: EPBuffer）
        │  复用或新建 NCCL communicator（V2.5：EP_REUSE_NCCL_COMM 时取 PyTorch backend._comm_ptr()）
        ▼
V1：NVSHMEM（对称堆、IBGDA put）+ CUDA IPC（NVLink buffer）
V2.5：NCCL device API —— LSA（NVLink load/store，ncclGetLsaDevicePointer）
                        GIN（GPU 发起 RDMA put/signal，GDAKI 类型）
        ▼
NVLink（scale-up） / InfiniBand（scale-out，推荐自适应路由、独立 VL）
```

## 核心概念

### 1. dispatch / combine 语义

- **dispatch**：每个 token 按 top-k 路由复制到若干目标专家所在 rank。先要算布局：每个 rank、每个 RDMA 节点、每个专家各收多少 token、某 token 是否发往某 rank（V1 `get_dispatch_layout` 的返回值在 vLLM 调用点可见：`num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank`，见 `third_party/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`）。接收端按专家整理成 grouped GEMM 需要的布局。
- **combine**：dispatch 的逆过程，专家输出按 handle 记录的来源送回原 rank 并按 top-k 权重求和。数据量矩阵是 dispatch 的转置（`third_party/ai-infra-book/manuscripts/07-数据中心网络.md` 7.2.3）。
- **handle**：dispatch 产生的路由元数据，combine 和反向复用（训练中 dispatch 的反向就是 combine，combine 的反向就是 dispatch——V2.5 README 示例 `dispatch_backward` / `combine_backward` 正是这样写的）。
- **精度**：dispatch 常用 FP8（带 scale），combine 用 BF16。
- **去重**：同一 token 发往同一目的卡上的多个专家时只发一次（按卡而非按专家发送），`06-超节点.md` 6.2.6 算例中流量减半。

### 2. V1 normal（高吞吐）kernel —— 概念

- 场景：训练与推理 prefill，batch 大（数千 token/rank），追求带宽。
- 节点内：通过 NVLink（CUDA IPC 映射的 buffer）直接写对端。
- 跨节点：**两跳转发**——token 先经 RDMA 发到远端节点上“同 local rank 号”的 GPU（rail 对齐，只用同号网卡、避免跨轨），再由它经 NVLink 转发给该节点内真正持有专家的 GPU。这样跨节点流量按节点聚合（同一 token 发往一个节点的多个 GPU 只走一次 RDMA），NVLink 带宽用于节点内扇出。
- 需要 CPU 同步拿到接收计数来分配输出张量（vLLM 注释：“the dispatch kernel will block the CPU”，见 `deepep_ht.py`），因此不适合 CUDA Graph。
- 占用一定数量 SM 作为通信 warp（`Buffer.set_num_sms`，V1；vLLM `DeepEPAll2AllManagerBase` 中 `num_sms = 20`）。
- warp 角色（已核实）：`hybrid-ep@10d4dd73/csrc/kernels/internode.cu` 的 `enum class WarpRole` 定义了 `kRDMASender`、`kRDMASenderCoordinator`、`kRDMAAndNVLForwarder`、`kForwarderCoordinator`、`kNVLReceivers` 五种角色；接收计数经 `notify_dispatch` 写入零拷贝 host 内存 `moe_recv_counter`，CPU 轮询后才分配输出（见 [09/01](../09-ep-kernels/01-design-space.md)）。

### 3. V1 low-latency kernel —— 概念

- 场景：推理 decode，每 rank 只有几十到一百多 token，追求延迟、要求可被 CUDA Graph 捕获。
- **直连**：每个 (token, top-k 专家) 一次传输，不经节点内转发；跨节点用 **IBGDA**（GPU 直接写 NIC doorbell，绕过 CPU proxy）发起 put。早期版本节点内也走 RDMA；PR #173 之后，对端可经 NVLink P2P 访问时改为直接 store（`nvshmemi_get_p2p_ptr`，由 `allow_nvlink_for_low_latency_mode` 控制）。
- **固定容量 buffer**：按 `num_max_dispatch_tokens_per_rank` 预分配（SGLang `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 默认 128），输出形状固定 → 无 CPU 同步、可 CUDA Graph。
- **hook 式通算重叠**：`low_latency_dispatch(..., return_recv_hook=True)` 只发出 RDMA 写就返回，返回一个 `hook`；调用方先去做其他计算（如另一个 micro-batch 的 attention），稍后调用 `hook()` 才执行接收端等待/整理。发出与接收之间**不占用 SM**（数据由 NIC 搬运）。调用点：`third_party/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py`（`return_recv_hook=True`）、`third_party/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py`（`hook() if self.return_recv_hook else event.current_stream_wait()`）。
- 代价：同节点流量也走网卡，带宽不如 NVLink；buffer 按最大 token 数预留，显存大。

两种模式跨节点的数据路径对比（2 节点 × 4 GPU；t 是 node0 GPU0 上的一个 token，top-k 选中的专家 E 在 node1 的 GPU1、GPU2）：

```text
 V1 normal：先 RDMA 到远端"同号"GPU（rail 对齐，同一节点只跨网 1 次），再 NVLink 扇出

   node0                          node1
   ┌──────┐                       ┌──────┐   NVLink   ┌──────┐
   │ G0 t │════ RDMA (x1) ═══════►│ G0   │───────────►│ G1 E │
   └──────┘                       └──┬───┘            └──────┘
                                     │      NVLink    ┌──────┐
                                     └───────────────►│ G2 E │
                                                      └──────┘

 V1 low-latency：每个 (token, 专家) 直接发到目标 GPU，跨节点 IBGDA put，无转发

   node0                                              node1
   ┌──────┐                                           ┌──────┐
   │ G0 t │═════════════ RDMA put (IBGDA) ═══════════►│ G1 E │
   │      │                                           └──────┘
   │      │                                           ┌──────┐
   │      │═════════════ RDMA put (IBGDA) ═══════════►│ G2 E │
   └──────┘                                           └──────┘

            normal                                   low-latency
 跨节点     RDMA 到同号 GPU，再 NVLink 转发          RDMA 直达目标 GPU
 节点内     NVLink（CUDA IPC buffer）                早期走 RDMA；PR #173 后可 NVLink P2P store
 接收大小   CPU 等计数后分配，不能 CUDA Graph        固定容量 buffer，可 CUDA Graph
```

V2.5 的 hybrid 模式保留了 normal 的两跳结构：scaleout warp 用 `gin.put<ncclTeamTagRail>` 跨节点同轨发送，forward warp 经 `ncclTeamTagLsa` 对称指针做 NVLink 转发（见下文源码导读）。

### 4. V2 / V2.5（本地代码）

- 统一接口：`EPBuffer` 同时覆盖高吞吐与低延迟；按拓扑分 **direct**（scaleout=1，所有 rank 在一个 scale-up 域，直接 NVLink/GIN）与 **hybrid**（scaleout>1，RDMA + NVLink 转发，对应 V1 normal 的分层思想）。
- 后端从 NVSHMEM 换成 **NCCL GIN**：可复用框架已有的 NCCL communicator，不再单独初始化 NVSHMEM。
- SM 与 QP 数由解析模型计算（`get_theoretical_num_sms` / `get_theoretical_num_qps`），不再 auto-tune。
- 重叠改为 **deferred epilogue**：`defer_epilogue=True` + `async_with_compute_stream=True` 时 dispatch/combine 返回 `EventOverlap`，`.wait()`/`current_stream_wait()` 时在当前 stream 上执行收尾（拷贝/整理），中间可插入独立计算。
- 不再支持 0-SM 的 RDMA LL 路径（hook 期间不占 SM 的能力被移除）。

同样是"发起通信 → 插入别的计算 → 收尾"，V1 LL 的 recv hook 与 V2.5 的 deferred epilogue 在时间线上的区别：

```text
 V1 LL：low_latency_dispatch(..., return_recv_hook=True)

   compute SM : [dispatch: post RDMA writes][ other micro-batch compute ][ hook(): recv ][ experts ]
   NIC        :                             [ RDMA writes in flight ═►]
                                            ↑ 这段时间通信不占 SM，数据由网卡搬运

 V2.5：dispatch(..., async_with_compute_stream=True, defer_epilogue=True) → EventOverlap

   comm stream    : [ dispatch kernel（占 SM）]
   current stream :        [ independent compute ][ wait(): CPU 等接收计数 + copy epilogue ][ experts ]
                     ↑ current stream 不等通信 kernel        ↑ event.current_stream_wait() / .wait()
```
- 扩展：`lb_prefetch_weights` / `lb_reduce_grads`（冗余专家权重预取、梯度回收）、`BucketBuffer`（CP/DP 的 AG/RS/AR）、`EngramBuffer`、`PPBuffer`——DeepEP 更名为 “DeepEveryParallel”。

### 5. 与 NVSHMEM 的关系

- V1：NVSHMEM 提供对称堆（每个 rank 相同偏移的 buffer）、GPU 端 put/get/signal 与 IBGDA 传输，DeepEP 在其上写 kernel。`hybrid-ep` 分支的 `hybrid-ep@10d4dd73/third-party/README.md` 要求 NVSHMEM 3.3.9 及以上（上游版本即可），目录中仍保留 `nvshmem.patch`；IBGDA 的设备侧代码改编自 NVSHMEM，放在 `csrc/kernels/ibgda_device.cuh`。
- V2.5：不依赖 NVSHMEM。对称内存由 `ncclMemAlloc` 或 `cuMemCreate/cuMemMap` 分配并 `ncclCommWindowRegister` 注册；NVLink 对端指针来自 `ncclGetLsaDevicePointer`，NVLS multimem 指针来自 `ncclGetLsaMultimemDevicePointer`；RDMA 用 GIN 的 put / signal。等价地说，NCCL 2.3x 的 device API 吸收了“NVSHMEM 式”的 GPU 发起通信能力。

### 6. 为什么 MoE All2All 是 CCL 的新挑战

1. **动态、不均匀、事先未知**：每对 rank 的消息大小由当次路由决定，需要先交换计数（多一次同步或 GPU 端计数），无法像 AllReduce 那样静态切 chunk；热点专家造成 skew（`09-分布式推理.md` 9.4.1：EP8 约 8%、EP256 可达 4×）。
2. **数据重排与通信耦合**：发送前按目的地打包、接收后按专家排布给 grouped GEMM，最好在通信 kernel 内完成（避免额外的 permute kernel 和显存往返）。
3. **两种极端工况**：prefill/训练要带宽（MiB~GiB 级），decode 要延迟（每 rank 百 token 级、KiB 级消息、CUDA Graph），一种算法难以兼顾 → HT/LL 两套。
4. **分层拓扑**：NVLink 与 RDMA 带宽差一个数量级，要按节点聚合跨节点流量、用 NVLink 扇出；rail 对齐避免跨轨。
5. **SM 争用与重叠**：通信 kernel 占 SM 会拖慢专家 GEMM；需要 hook/deferred epilogue、双 batch 重叠、尽量少 SM。
6. **网络侧**：大量小消息、多 QP（V2.5 hybrid 默认 65/129 个 QP）、对 RDMA atomic 性能敏感（README 推荐 `PCI_ATOMIC_MODE=4`），对拥塞控制与自适应路由敏感，需要用 VL/SL 与其他流量隔离（`EP_OVERRIDE_RDMA_SL` > `sl_idx` > `EP_DEFAULT_RDMA_SL` > NCCL 默认）。
7. **通用库在追赶**：NCCL 自身在 `third_party/nccl/contrib/nccl_ep/` 提供 NCCL EP（`ncclEpDispatch` / `ncclEpCombine`，LL 与 HT 两种算法，基于 LSA 与 GIN；README 注明已迁至 NVIDIA/nccl-extensions），NCCL 核心也新增了 `ncclAlltoAll`、device API 与 `src/gin/`——说明 EP 通信正从“框架外专用库”回流到 CCL。

## 源码导读（路径 + 函数 + 作用）

均为 `third_party/DeepEP/` 下的 V2.5 代码。

| 路径 | 符号 | 作用 |
|---|---|---|
| `deep_ep/buffers/ep.py` | `class EPBuffer(BufferBase)` | 构造参数含 `group, num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch, allow_hybrid_mode, sl_idx, num_allocated_qps, num_cpu_timeout_secs, num_gpu_timeout_secs`；`num_allocated_qps=0` 时 hybrid 模式取 65（支持快速 RDMA atomic）或 129，否则 17 |
| 同上 | `EPBuffer.get_buffer_size_hint` / `get_theoretical_num_sms` / `get_theoretical_num_qps` | 解析式估算 buffer、SM、QP；direct 模式 QP=`min(num_sms, 9)`，hybrid 为 `num_sms*16+1` |
| 同上 | `EPBuffer.dispatch(...)` | 参数 `do_cpu_sync`（训练/prefill 拿精确大小；decode 用 GPU 侧计数）、`do_expand`、`expert_alignment`、`defer_epilogue`、`previous_event`、`async_with_compute_stream`；返回 `recv_x, recv_topk_idx, recv_topk_weights, EPHandle, EventOverlap` |
| 同上 | `EPBuffer.combine(x, handle, topk_weights, bias, ...)` | 逆向发送并求和，`bias` 可在 epilogue 中加 shared expert 输出或残差 |
| 同上 | `class EPHandle` | 路由元数据（`psum_num_recv_tokens_per_expert`、`recv_src_metadata` 等） |
| `deep_ep/utils/event.py` | `EventOverlap` | `current_stream_wait`、`wait`、`register_hook_after_wait` |
| `deep_ep/comm/handle.py` | `get_nccl_comm_handle` | 默认复用 PyTorch NCCL comm（`backend._comm_ptr()`，`EP_REUSE_NCCL_COMM`）；否则 all-gather unique id 自建 |
| `csrc/kernels/comm/context.cpp` | `comm::Context` | 查询 `ncclCommQueryProperties`，要求 `NCCL_GIN_TYPE_GDAKI`；`ncclDevCommCreate`；`ncclCommWindowRegister(..., NCCL_WIN_STRICT_ORDERING)`（集合调用，内部 bootstrap barrier）；`ncclGetLsaDevicePointer` / `ncclGetLsaMultimemDevicePointer` |
| `csrc/kernels/comm/symmetric.hpp` | `GPUSymmetricMemory`、`ElasticSymmetricMemory`、`HybridElasticSymmetricMemory` | 对称内存（`ncclMemAlloc` 或 `cuMemCreate/cuMemMap` + POSIX-FD/fabric 句柄） |
| `csrc/kernels/ep/dispatch.hpp` | `launch_dispatch` | `num_scaleout_ranks == 1` 走 direct（`dispatch_impl`），否则 hybrid（`hybrid_dispatch_impl`） |
| `deep_ep/include/deep_ep/impls/ep/dispatch.cuh` | `dispatch_impl` | notify warp + dispatch warp；`gin.put` 或经 `gin.get_sym_ptr` 直接 NVLink 写 |
| `deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh` | `hybrid_dispatch_impl` | notify / scaleout / forward 三类 warp，`kNumScaleoutWarps == kNumForwardWarps`，“a warp is a channel”；scaleout warp 用 `gin.put<ncclTeamTagRail>` 跨节点同轨 RDMA，forward warp 经 `ncclTeamTagLsa` 对称指针 NVLink 转发——V1 normal 两跳转发思想的 V2 版 |
| `deep_ep/include/deep_ep/impls/ep/combine.cuh` / `hybrid_combine.cuh` | `combine_impl` / `hybrid_combine_impl` | combine；`gin.is_nvlink_accessible` 判断能否 NVLink 直达 |
| `deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh` / `combine_reduce_epilogue.cuh` | epilogue kernel | deferred epilogue 的实体 |
| `deep_ep/include/deep_ep/comm/handle.cuh` | `struct NCCLGin` | 设备侧封装：`put`、`put_value`、`red_add_rel`、`get`、`flush`、`wait`、`get_sym_ptr`；team 标签 `ncclTeamTag{World,Lsa,Rail}` |
| `tests/ep/test_ep.py` | `test_dispatch_combine`、`test_loop` | 默认参数 `--num-tokens 4096 --hidden 7168 --num-topk 6 --num-experts 256` |

NCCL 侧对应：`third_party/nccl/src/include/nccl_device/core.h`（`NCCL_GIN_TYPE_PROXY = 2`、`NCCL_GIN_TYPE_GDAKI = 3`）、`third_party/nccl/src/transport/net_ib/gdaki/`（GDAKI 的 IB 实现）、`third_party/nccl/src/gin/`（GIN host 侧与 proxy）。

## 性能要点与常见坑

- V2.5 README 无本地性能表；ai-infra-book 的研究快照（`third_party/ai-infra-book/research/ub-ep-integration-2026-09-10/sources/deepep.md`，V2 早期版）记录：SM90+CX7、EP 8×2、8K token/batch、hidden 7168、top-8、FP8 dispatch/BF16 combine 时 dispatch 瓶颈带宽约 90 GB/s（RDMA）、12 个 SM；SM100 单节点 EP8 NVLink 约 726 GB/s（64 SM）。注明是“逻辑带宽”（含本地 rank 流量）。
- 所有 rank 必须对 `num_max_tokens_per_rank` 取一致值，并覆盖训练/prefill/decode 的最大 batch。
- `do_cpu_sync=False`（decode）时输出按容量分配，专家 GEMM 必须用 `handle.psum_num_recv_tokens_per_expert` 识别有效区间；错误使用会把填充数据当有效 token。
- 网络：推荐 IB 自适应路由、为 EP 流量单独 VL；最大带宽场景建议关闭拥塞控制；RoCE“理论兼容”但未充分测试——RoCE 集群部署需要额外验证。
- 超时：`num_cpu_timeout_secs=300`、`num_gpu_timeout_secs=100`，GPU 端超时用于检测对端掉线导致的死等。
- 框架版本错配（V1/V2/V2.5 API）是集成时最常见的问题，先确认安装的 `deep_ep.__version__`。

## 面试自测

1. 问：DeepEP normal 模式跨节点为什么要“先 RDMA 到同号 GPU 再 NVLink 转发”？
   要点：按节点聚合，一个 token 去同一节点多个专家只跨网一次；rail 对齐只用同号网卡、避免跨轨/PCIe 交换；节点内扇出用高带宽 NVLink；把 RDMA 消息变大、次数变少。
2. 问：low-latency 模式为什么能 CUDA Graph？为什么要纯 RDMA？
   要点：固定最大 token 容量、输出形状固定、无 CPU 同步拿计数；纯 RDMA 用 IBGDA 由 GPU 直接发起，省去节点内转发的同步阶段，路径最短；代价是同节点也走网卡、显存预留大。
3. 问：recv hook 如何做到通信不占 SM 的重叠？
   要点：dispatch 只发出 RDMA 写立即返回 hook；数据由 NIC DMA 写入接收端 buffer，期间 GPU 执行别的 kernel；需要结果时调用 hook 做等待和整理。V2.5 已移除此 0-SM 路径，改为 deferred epilogue（仍需 SM）。
4. 问：DeepEP V2 为什么从 NVSHMEM 换到 NCCL GIN？
   要点：复用框架已有 NCCL communicator，免去独立初始化与补丁版 NVSHMEM 依赖；继承 NCCL 的拓扑探测、网络插件和窗口注册；NCCL device API（LSA+GIN）已提供 GPU 发起 put/signal 能力。
5. 问：MoE All2All 相比传统集合通信新在哪？如果让你在 CCL 里支持它要做什么？
   要点：动态不均匀消息、需先交换计数、通信与数据重排融合、HT/LL 两种工况、分层拓扑聚合、SM 争用、网络小消息与多 QP。CCL 侧：GPU 发起的 device API、对称内存/窗口注册、按节点分层的 A2A 算法、可 CUDA Graph 的固定容量模式、低 SM 占用、与路由元数据（handle）复用——NCCL 的 `contrib/nccl_ep` 就是这条路线。
6. 问：训练中 dispatch 的反向是什么？
   要点：combine（把梯度按原路由送回并对 top-k 副本求和）；combine 的反向是 dispatch；复用前向的 handle，避免重新计算布局与 CPU 同步。

## 待深入

- 已完成：V1 normal 的 warp 角色与 `internode_ll.cu` 的 IBGDA 调用，见 [09-ep-kernels](../09-ep-kernels/README.md)；NCCL EP 与 DeepEP 的对应关系见 [09/05](../09-ep-kernels/05-lineage-and-selection.md)；V2.5 kernel 精读见 [09/02~04](../09-ep-kernels/README.md)。
- 读 `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh` 全文，画出 scaleout/forward warp 的数据流与 signal 时序。
- 读 `third_party/nccl/contrib/nccl_ep/README.md` 与 `device/ll_ep.cuh`、`device/hybrid_ep.cuh`，对比 NCCL EP 与 DeepEP 的 API 与算法。
- 读 `third_party/nccl/src/transport/net_ib/gdaki/` 理解 GIN 的 GDAKI 实现与 IBGDA 的对应关系（可并入 `06-comm-compute-fusion/`）。
