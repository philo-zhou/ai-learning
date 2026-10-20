# DeepEP V2.5 源码总览：一次 dispatch 从 Python 走到 kernel

> 层次：中层（贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP@b5a33eef（`deep_ep/buffers/ep.py`、`csrc/buffers/ep.hpp`、`csrc/kernels/ep/dispatch.hpp`、`csrc/kernels/ep/combine.hpp`、`csrc/kernels/comm/context.cpp`、`csrc/runtime/jit.hpp`、`deep_ep/include/deep_ep/{comm,layout,common}/`）、third_party/nccl@3e46c089（`src/include/nccl_device/gin.h`）、third_party/DeepInfer@774ce244

## 一句话

DeepEP V2.5 的 EP 部分是四层：Python `EPBuffer` 负责参数、handle 和事件；C++ `EPBuffer` 负责分配输出、管理 stream、决定 direct 还是 hybrid；launcher 把**所有形状都当成模板参数**拼成源码串交给 DeepJIT 运行时编译；kernel 建在 NCCL GIN（GPU 直驱网卡）和 LSA（NVLink 对称地址）之上，用 TMA 搬数据。读懂本篇的公共设施（对称内存、token 布局、workspace 信号、`NCCLGin` 封装、barrier、计数编码、stream 控制），后面 [03 direct](03-deepep-v25-direct.md) 和 [04 hybrid](04-deepep-v25-hybrid.md) 的 kernel 就只剩算法本身。

## 在调用链中的位置

```text
框架（DeepInfer v4_modules/mlp.py、vLLM DeepEPV2All2AllManager）
   │  buffer.dispatch(x, topk_idx, topk_weights, ..., do_cpu_sync, do_expand, defer_epilogue)
   ▼
deep_ep/buffers/ep.py        EPBuffer.dispatch                  估 SM/QP → 调 runtime → 包 EPHandle / EventOverlap
   │  self.runtime.dispatch(...)
   ▼
csrc/buffers/ep.hpp          EPBuffer::dispatch                 检查张量、stream 切换、分配 handle 张量、清 host workspace
   │  launch_dispatch(...)                                        ↓ 返回后（或 deferred 到 .wait()）
   │                                                             epilogue lambda：CPU 等计数 → 分配 recv_x → launch_dispatch_copy_epilogue
   ▼
csrc/kernels/ep/dispatch.hpp launch_dispatch                    选 dispatch_impl / hybrid_dispatch_impl，拼模板实参
   │  jit->compile(...) → jit->launch(cooperative, cluster=2)
   ▼
deep_ep/include/deep_ep/impls/ep/dispatch.cuh | hybrid_dispatch.cuh   主 kernel
   │  NCCLGin::put / put_value / red_add_rel / get_sym_ptr      comm/handle.cuh
   │  gpu_barrier                                                comm/barrier.cuh
   ▼
NCCL device API：ncclGin（GDAKI，GPU 直接写 WQE 敲门铃） / ncclGetLsaPointer（NVLink 对称地址）
```

## 目录地图

| 路径 | 内容 | 读到什么程度 |
|---|---|---|
| `deep_ep/buffers/ep.py` | `EPHandle`、`EPBuffer.__init__/dispatch/combine`、`get_theoretical_num_sms/qps` | 精读 |
| `csrc/buffers/ep.hpp` | C++ `EPBuffer`：缓冲大小公式、`dispatch`/`combine` 主流程、stream 控制 | 精读 |
| `csrc/kernels/ep/dispatch.hpp`、`combine.hpp` | launcher：warp 数、模板实参、launch 选项 | 通读 |
| `csrc/kernels/comm/context.cpp` | `Context`：`ncclDevCommCreate`、对称内存、窗口注册、逻辑域 | 通读 |
| `deep_ep/include/deep_ep/layout/ep/token.cuh`、`workspace.cuh` | 一个 token 在缓冲里的字节布局；workspace 里的计数与信号 | 精读 |
| `deep_ep/include/deep_ep/comm/handle.cuh`、`barrier.cuh` | `NCCLGin` 封装、QP 分配、`gpu_barrier` | 精读 |
| `deep_ep/include/deep_ep/common/ptx.cuh`、`math.cuh` | TMA、mbarrier、warp 原语、计数编码 | 遇到再查 |
| `deep_ep/include/deep_ep/impls/ep/*.cuh` | 7 个 kernel：dispatch / hybrid_dispatch / dispatch_copy_epilogue / combine / hybrid_combine / combine_reduce_epilogue + LB 两个 | 见 03、04 |
| `third-party/deep_jit`（子模块，未拉取） | DeepJIT：运行时编译与缓存 | 可跳过 |

## 核心概念

### 1. 构造 EPBuffer：通信器、缓冲大小、QP、对称内存

`EPBuffer.__init__`（`deep_ep/buffers/ep.py`）按顺序做五件事：

1. **复用 PyTorch 的 NCCL 通信器**：`comm.get_nccl_comm_handle(group)` 在 backend 有 `_comm_ptr` 且 `EP_REUSE_NCCL_COMM=1`（默认）时直接拿 PyTorch 的 `ncclComm_t`，否则自己建一个（`deep_ep/comm/handle.py`）。
2. **算缓冲大小**：`_C.calculate_ep_buffer_size` → C++ `calculate_buffer_size` 取 dispatch 与 combine 两种布局的较大者，再按 2 MB 对齐。`num_topk` 传 0 时按 32 估算（代码里大量 `kNumTopk <= 32` 限制）。
3. **定 QP 上限**：`num_allocated_qps` 为 0 时，hybrid 默认 65（网卡支持快速 RDMA 原子，即 `ibstat` 显示 MT4131 或更新）或 129，direct 默认 17。多出来的 1 个给 notify warp。
4. **建 C++ runtime**：`_C.EPBuffer(...)` 构造 `comm::Context`（`csrc/kernels/comm/context.cpp`）：
   - `ncclDevCommCreate`：只接受 GIN 的 GDAKI 类型（GPU 直接写 WQE，对应 V1 的 IBGDA）；`ginContextCount = num_allocated_qps` 且独占；`ginSignalCount = num_ranks + 4`（给 barrier 用）；连接类型 hybrid 用 `NCCL_GIN_CONNECTION_RAIL`（只和各节点上同号 GPU 建连接），direct 用 `NCCL_GIN_CONNECTION_FULL`（和所有 rank 建连接）。
   - 对称内存：`symmetric::alloc` 用 `cuMemCreate` 申请（优先 FABRIC handle，支持多节点 NVLink 域），布局为 `[workspace | EP buffer | LB 区]`，再 `ncclCommWindowRegister(..., NCCL_WIN_STRICT_ORDERING)` 注册成一个窗口。
   - 另外 `cudaMallocHost(..., cudaHostAllocMapped)` 分配一份 host workspace，GPU 通过映射指针直接写，CPU 轮询它拿接收计数。
5. **全组 barrier**：`torch.cuda.synchronize(); group.barrier()`，保证所有 rank 的窗口都注册好。

### 2. 逻辑域：scale-out × scale-up

`Context` 构造时从 `ncclTeamLsa(comm)` 得到 NVLink 域大小 `num_nvl_ranks`，于是 `num_rdma_ranks = num_ranks / num_nvl_ranks`。逻辑域由 `allow_hybrid_mode` 决定：

| | `num_scaleout_ranks` | `num_scaleup_ranks` | kernel |
|---|---|---|---|
| `allow_hybrid_mode=True`（默认） | `num_rdma_ranks` | `num_nvl_ranks` | 多节点时 `hybrid_*_impl`，单节点时退化为 direct |
| `allow_hybrid_mode=False` | 1 | 全部 rank | 永远 `dispatch_impl` / `combine_impl` |

`is_scaleup_nvlink = (num_scaleup_ranks == num_nvl_ranks)`：scale-up 域正好是 NVLink 域时，direct kernel 走纯 NVLink 分支（`team_t = ncclTeamTagLsa`）；否则 scale-up 域跨节点，走 `ncclTeamTagWorld`，NVLink 可达的用 TMA 直写，不可达的走 RDMA。launcher 里的判断就是 `num_scaleout_ranks == 1` → direct，否则 hybrid。

### 3. JIT：所有形状都是编译期常量

`launch_dispatch`（`csrc/kernels/ep/dispatch.hpp`）把 kernel 名和模板实参格式化成一段源码：

```text
#include <deep_ep/impls/ep/dispatch.cuh>
static void __instantiate_kernel() { auto ptr = &deep_ep::ep::dispatch_impl<is_scaleup_nvlink, do_cpu_sync,
    reuse_slot_indices, num_sms, num_notify_warps, num_dispatch_warps, num_ranks, hidden_bytes, num_sf_packs,
    num_max_tokens_per_rank, num_experts, num_topk, expert_alignment, num_qps, timeout_cycles>; }
```

再 `jit->compile` + `jit->launch`。后果：

- rank 数、hidden、top-k、专家数、`num_max_tokens_per_rank`、SM 数、QP 数都是 `constexpr`，kernel 里的除法、循环上界、共享内存切分都在编译期定死；`num_tokens` 是运行时参数，batch 大小变化不触发重编译。
- 第一次遇到新配置会调用 NVCC 编译，结果缓存在 `EP_JIT_CACHE_DIR`（默认 `$HOME/.dj`）。prefill 与 decode 用不同的 `num_max_tokens_per_rank`（DeepInfer 就是这样）会编译出两份 kernel。
- launch 选项：`cooperative = true`（kernel 内要 `this_grid().sync()`）、`cluster_dim = 2 - (num_sms % 2)`、动态共享内存开到设备上限；epilogue kernel 用 `enable_pdl = true`（Programmatic Dependent Launch），主 kernel 末尾 `cudaTriggerProgrammaticLaunchCompletion()`，epilogue 开头 `cudaGridDependencySynchronize()`，两个 kernel 之间不经过 CPU。
- warp 数由共享内存决定：direct dispatch 的数据 warp 数 = `min((smem − notify 部分) / token 字节数, 32 − 4)`，每个 warp 在共享内存里独占一个 token 缓冲。

### 4. token 布局：一个 token 在缓冲里长什么样

`layout::TokenLayout`（`layout/ep/token.cuh`）定义 dispatch 时一个 token 的字节布局，各段 32 字节对齐（TMA 要求），总长 64 字节对齐（`kNumRDMAAlignmentBytes`）：

```text
dispatch token（with_metadata = true）：各段 32 B 对齐，总长 64 B 对齐
 ├─ hidden                    hidden × elem_size（BF16 2 B / FP8 1 B）
 ├─ SF                        FP8 时 ⌈hidden/32⌉ 个 sf_pack_t（4 B）
 └─ metadata
     ├─ topk_idx              int32 × k
     ├─ topk_weights          float × k
     ├─ src_token_global_idx  int32，= rank × M + token
     └─ linked_list_idx       int32 × k，仅 hybrid 用
 共享内存里的副本末尾再加一个 mbarrier（get_num_bytes<true>）

combine token（with_metadata = false）：hidden（BF16） → topk_idx × k → topk_weights × k
```

要点：**路由信息跟着数据走**。接收端拿到一个 token 就知道它要去本地哪些专家（`topk_idx`）、权重是多少、来自哪个 rank 的第几个 token（`src_token_global_idx = rank_idx × num_max_tokens_per_rank + token_idx`），不需要另外传一份路由表。

`layout::BufferLayout<kWithMBarrier>` 是 `[num_ranks][num_max_tokens_per_rank]` 个 token 的二维数组，提供 `get_rank_buffer(r)`、`get_token_buffer(i)`、`get_channel_buffer<N>(c)` 三种切片。

### 5. 缓冲区按最坏情况划分

`get_dispatch_buffer_size` / `get_combine_buffer_size`（`csrc/buffers/ep.hpp`），记 R = rank 数，M = `num_max_tokens_per_rank`，S/U = scale-out / scale-up rank 数，k = top-k，C = `kNumMaxChannels` = 1280：

| 阶段 | direct | hybrid |
|---|---|---|
| dispatch 接收 | `[R][M]`：每个源 rank 一段，按发送端分配的槽位写入 | scale-up 接收 `[U][S×M]` + scale-out 接收 `[S][M + C]` |
| dispatch 发送 | `[1][M]`（RDMA 源必须在已注册窗口里）；纯 NVLink 时为 0 | scale-out 发送 `[1][M]` |
| combine 接收 | `[min(R,k)][M]`（允许多次归约）或 `[k][M]` | scale-up `[min(U,k)][S×M]` + scale-out `[min(S,k)][M]` |
| combine 发送 | 纯 NVLink 为 0，否则 `[R][M]`（不允许多次归约时 `×k`） | `[1 或 k][S×(M + C)]` |

最终大小取 dispatch 与 combine 的较大者——两个阶段**共用同一块缓冲**，这也是每个 kernel 开头都要先做一次全组 barrier 的原因（确认对端已经用完上一阶段的数据）。combine 接收只要 `min(R,k)` 行而不是 R 行，原因见 [03](03-deepep-v25-direct.md) 的 combine 一节。

### 6. workspace：计数与信号

`layout::EPSignals`（`layout/ep/workspace.cuh`）放在对称窗口最前面，所有 rank 偏移相同，所以可以直接远程写：

| 字段 | 用途 |
|---|---|
| `barrier_signals` | `gpu_barrier` 的 NVLink 计数器与相位 |
| `notify_reduction[R + E]` | 多个 SM 的计数做网格内归约：高 32 位数"到了几个 SM"，低 32 位累加计数 |
| `scaleup_rank_expert_count[2][R + E]` | `[0]` 发送区、`[1]` 接收区：每个源 rank 发给我的 token 数和每个专家的 token 数 |
| `scaleup_atomic_sender_count[R]` | 发送端按目标 rank 分配槽位的原子计数器（direct dispatch、hybrid forward） |
| `scaleout_rank_expert_count[2][R + E]` | hybrid 跨节点那一级的计数 |
| `scaleout_channel_signaled_tail[1280][1024]` | hybrid dispatch 每个 (channel, 源节点) 的 tail（含完成标志） |
| `channel_scaleup_tail[1280][1024]` | hybrid 中每个 (channel, NVLink peer) 的 tail / linked list 末尾 |

上限：`kNumMaxRanks = 1024`、`kNumMaxExperts = 2048`、`kNumMaxExpertsPerRank = 512`、`kNumMaxChannels = 1280`（`common/compiled.cuh`、`workspace.cuh`）。

### 7. 计数编码：0 表示"还没到"

`math::encode_decode_positive(v) = -v - 1`，`is_decoded_positive_ready(v) = v >= 0`（`common/math.cuh`）。写方写入 `-(count+1)`，读方再取一次 `-x-1` 还原。workspace 平时是 0，解码后是 −1，表示未就绪；计数为 0 时写入的是 −1，解码后是 0，表示"就绪，0 个"。和 V1 LL 写 `-(n+1)` 是同一个技巧（见 [01](01-design-space.md) 决定 2），这样读方用完把位置清零即可，不需要额外的有效位。

### 8. `NCCLGin`：一层很薄的封装

`comm::NCCLGin`（`comm/handle.cuh`）在每个 warp 里按 `qp_idx` 构造一个 `ncclGin`，同时持有三个 team：

- `team_world`：全部 rank；`team_lsa`：本 NVLink 域；`team_rail`：各节点上 local rank 相同的 GPU。

| 方法 | NVLink 可达 | 不可达 |
|---|---|---|
| `get_sym_ptr<team>(ptr, dst)` | 返回 `ncclGetLsaPointer(window, offset, dst_nvl_rank)`，即对端同一偏移的可写地址；自己则返回原指针 | 返回 `nullptr` |
| `put<team>(recv_ptr, send_ptr, bytes, dst)` | 也走网卡（注释："local or NVLink put will also go through NIC"），所以 kernel 里 NVLink 路径都改用 TMA 直写 | `ncclGin::put`，源和目的都按窗口偏移给出 |
| `put_value<team>(ptr, v, dst)` | `st.relaxed.sys` 直接写 | `ncclGin::putValue`（inline 写一个值） |
| `red_add_rel<team>(ptr, v, dst)` | `red.release.gpu/sys.add` | `ncclGin::signal` + `ncclGin_VASignalAdd`（远端原子加到窗口偏移处） |

`NCCLGin` 对 `ncclTeamTagRail` 认为只有自己可经 NVLink 访问（`get_sym_ptr` 只在 `dst == team_rail.rank` 时返回指针），所以 hybrid 的跨节点写全部走 GIN。

**QP 分配**：`comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>`（`comm/barrier.cuh`）。有 notify warp 时 QP 0 留给它；剩下的 QP 如果比 SM 多，每个 SM 独占若干个（`NCCL_GIN_RESOURCE_SHARING_CTA`），否则所有 channel 轮流共享（`..._GPU`）。`get_theoretical_num_qps` 给 direct 的默认值是 `min(num_sms, 9)`（"less QPs to reduce DB ringing overhead"），hybrid 是 `num_sms × 16 + 1`（每个 channel 一个独立 QP），都再被 `num_allocated_qps` 截断。

**批量门铃**：kernel 里很多 `gin.put` 带 `ncclGinOptFlagsAggregateRequests`。在 GDAKI 后端它被翻译成 `DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING`（`third_party/nccl/src/include/nccl_device/gin/gdaki/gin_gdaki.h`）：只写 WQE、不敲门铃，等后面一个不带该标志的请求一次性敲。这也说明 NCCL GIN 的 GDAKI 后端底层就是 DOCA GPUNetIO verbs，和 Hybrid-EP 用的是同一套 GPU 直驱接口。

**信号强弱**：DeepEP 用的 `ncclGin_VASignalAdd` 在本地 nccl@3e46c089 中已标为 deprecated，新接口分 `Strong`（"visibility implies all preceding puts are settled"）和 `Weak`（只保证捆绑的那个 put），旧名的强弱取决于 `comm.ginStrongLegacySignals`（`src/include/nccl_device/gin.h`、`impl/gin__funcs.h`）。hybrid 的 tail 信号依赖强语义，见 [04](04-deepep-v25-hybrid.md)。

### 9. `gpu_barrier`：完成靠 flush，不靠同 QP 顺序

`comm::gpu_barrier`（`comm/barrier.cuh`）是每个 kernel 开头、结尾都会调用的全组 barrier，模板参数 `kFlushStores / kSyncAtStart / kSyncAtEnd` 控制三段：

```text
gpu_barrier
 ├─ kFlushStores：tma_store_commit + tma_store_wait（本 warp 的 TMA 写完成）
 ├─ kSyncAtStart：this_grid().sync()
 ├─ scale-up 与 scale-out 两级（hybrid 时 SM 0 做 scale-up，其余 SM 做 scale-out，并行）
 │    NVLink 域：nvlink_barrier_wo_local_sync
 │       只用 SM 0；对每个 peer 的对称计数器 red.add.release.sys(±1)，再自旋等自己的计数器到目标值
 │       计数器低 2 位 = (sign, phase)：两个相位槽交替，+1 到 N、下一轮 −1 回到 0，无需清零
 │    RDMA：gin_barrier_wo_local_sync
 │       ① 所有 SM 分摊 flush 全部 QP（ncclGin::flush：等本 QP 已发出的请求在本地完成）
 │       ② World team 时再加 fence.acq_rel.sys（NVLink 直写与 GIN 信号混用）
 │       ③ this_grid().sync()
 │       ④ SM 0 用 QP 0 向每个 rank 发 ncclGin_SignalInc{signal = 我的 rank}，再等每个 rank 的信号
 └─ kSyncAtEnd：this_grid().sync()
```

和 V1 LL 的区别很关键：V1 LL 靠"数据和计数走同一个 RC QP、计数在后"来保证"计数可见 ⇒ 数据可见"，不需要等待；V2.5 的 direct 模式在 barrier 里**先 flush 所有数据 QP**（RC 下本地完成意味着收到了对端 ACK，数据已落到对端内存），再走 QP 0 发信号，所以数据和信号可以在不同 QP 上。代价是 flush 要等一个 RTT，换来的是 QP 可以按 channel 任意分配。

### 10. stream、事件与 deferred epilogue

`EPBuffer::dispatch`（`csrc/buffers/ep.hpp`）的 stream 控制：

- `stream_control_prologue`：通信 stream 先等 `previous_event`（没有就等当前计算 stream）。有 `previous_event` 时必须 `allocate_on_comm_stream=True`——计算已经排在前面了，新张量要在通信 stream 上分配，避免和计算争用内存。
- 主 kernel 在通信 stream 上发出（`comm::get_comm_stream()`）。
- epilogue 被包成一个 lambda：
  - `do_cpu_sync=True`：释放 GIL 后自旋读 host workspace 里每个 scale-up rank 和每个本地专家的计数（编码过，`is_decoded_positive_ready` 判就绪），超过 `num_cpu_timeout_secs` 抛异常；得到精确的 `num_recv_tokens` 和每专家数量再分配输出。
  - `do_cpu_sync=False`：按最坏情况分配，`num_recv_tokens = M × R`，展开模式 `num_expanded_tokens = R × M × min(k, 本地专家数) + (alignment − 1) × 本地专家数` 再对齐；真实数量留在 GPU 上的 `psum_num_recv_tokens_per_scaleup_rank` / `psum_num_recv_tokens_per_expert`。
  - 然后 `launch_dispatch_copy_epilogue`。
- `defer_epilogue=False`：立即在通信 stream 上跑 epilogue，再按 `async_with_compute_stream` 决定返回事件还是让计算 stream 等待。
- `defer_epilogue=True`（要求 `async_with_compute_stream=True`）：只返回事件和 lambda，Python 侧 `EventOverlap.register_hook_after_wait` 注册；用户调用 `.wait()`/`current_stream_wait()` 时才**在当前 stream 上**跑 CPU 等待和 epilogue。所以 CPU 自旋也被推迟了，主线程可以先去排后面的计算。

这就是 V2.5 替代 V1 LL recv hook 的方式：通信 kernel 照样占 SM，但发出后立即返回，等到真正需要结果时再收尾。

### 11. SM 与 QP 的解析估算

`get_theoretical_num_sms(num_experts, num_topk)`（`deep_ep/buffers/ep.py`）按均衡路由估算：

1. 期望命中组数 `E[g] = G × (1 − C(E − E/G, k) / C(E, k))`：k 个专家落在 G 个组里，平均命中几个组。分别算 scale-out 级（G = 节点数）和 rank 级（G = R）。
2. 以"每 rank 收到的 token 数据量"为单位 1，累加 SM 读写量 `sm_read / sm_write`、RDMA 流量、NVLink 流量（hybrid 与 direct 两套系数，注释写明了每一项对应哪类 warp 的工作）。
3. 选瓶颈链路：hybrid 时比较 `rdma_traffic / rdma_gbs` 与 `nvlink_traffic / nvlink_gbs`。
4. 需要的 SM 数 = 瓶颈带宽下，让每 SM 的 HBM 读/写带宽（`get_sm_read_gbs/get_sm_write_gbs`）跟得上的数量，dispatch、combine 各算两项取最大；再 ×1.3、至少 4、按 4 对齐；`prefer_overlap_with_compute=False` 时至少 64；不超过整卡（取偶数）。

带宽靠 `nvidia-smi`、`ibstat` 自动探测。DeepInfer 的用法是先拿这个估算值，再保证给计算留至少 `min(32, 总 SM / 4)` 个 SM（`third_party/DeepInfer/deepinfer/model_executor/parallel_utils/userbuffer.py` 的 `allocate_ep`）。

### 12. EPHandle 与 cached dispatch

`EPHandle`（`deep_ep/buffers/ep.py`）保存一次 dispatch 的全部布局信息：`psum_num_recv_tokens_per_scaleup_rank`、`psum_num_recv_tokens_per_expert`、`recv_src_metadata`（每个收到的 token：源全局下标、源 rank/槽位 × k + master lane、展开后的 k 个行号）、`dst_buffer_slot_idx`（发送端分配的槽位），hybrid 另有 `token_metadata_at_forward` 和 `channel_linked_list`。它有两个用途：

- **combine 的输入**：combine 不再看路由，只按 handle 原路返回。
- **cached dispatch**：传 `handle=` 时 launcher 把 notify warp 数设为 0、`reuse_slot_indices = true`，kernel 直接读 `dst_buffer_slot_idx` 作为槽位，不再计数、不再原子分配，也不能做 CPU 同步。训练反向里 `combine_backward` 就是用前向 handle 做一次 cached dispatch。`topk_idx` 被 handle 按引用持有，用 `_version` 检查在使用期间没被原地修改。

## 源码导读

| 路径 | 函数 / 符号 | 作用 |
|---|---|---|
| `third_party/DeepEP/deep_ep/buffers/ep.py` | `EPBuffer.__init__` | 通信器、缓冲大小、QP 默认值、构造 runtime |
| 同上 | `get_theoretical_num_sms`、`get_theoretical_num_qps` | SM/QP 解析估算 |
| 同上 | `EPBuffer.dispatch`、`finalize_dispatch` | 参数默认值、cached handle 展开、deferred hook |
| `third_party/DeepEP/csrc/buffers/ep.hpp` | `get_dispatch_buffer_size`、`get_combine_buffer_size`、`calculate_buffer_size` | 缓冲布局与大小 |
| 同上 | `EPBuffer::dispatch`、`EPBuffer::combine` | 主流程、epilogue lambda、CPU 等待 |
| 同上 | `stream_control_prologue`、`stream_control_epilogue` | 通信 stream 与事件 |
| `third_party/DeepEP/csrc/kernels/ep/dispatch.hpp` | `launch_dispatch`、`launch_dispatch_copy_epilogue` | warp 数、模板实参、launch 选项 |
| `third_party/DeepEP/csrc/kernels/comm/context.cpp` | `Context::Context`、`get_logical_domain_size` | dev comm、对称内存、窗口、逻辑域 |
| `third_party/DeepEP/csrc/runtime/jit.hpp` | `init_jit` | DeepJIT 配置：NVCC 选项，只开 GDAKI |
| `third_party/DeepEP/deep_ep/include/deep_ep/layout/ep/token.cuh` | `TokenLayout`、`BufferLayout` | token 与缓冲布局 |
| `third_party/DeepEP/deep_ep/include/deep_ep/layout/ep/workspace.cuh` | `EPSignals`、`EPWorkspaceLayout` | 计数与信号 |
| `third_party/DeepEP/deep_ep/include/deep_ep/comm/handle.cuh` | `NCCLGin` | GIN / LSA 封装 |
| `third_party/DeepEP/deep_ep/include/deep_ep/comm/barrier.cuh` | `get_qp_mode`、`gpu_barrier`、`nvlink_barrier_wo_local_sync`、`gin_barrier_wo_local_sync` | QP 分配与 barrier |
| `third_party/DeepEP/deep_ep/include/deep_ep/common/math.cuh` | `encode_decode_positive` | 计数编码 |
| `third_party/DeepEP/deep_ep/include/deep_ep/common/ptx.cuh` | `tma_load_1d`、`tma_store_1d`、`deduplicate`、`exchange` | PTX 封装 |

## 性能要点与常见坑

- **`num_max_tokens_per_rank` 是编译期常量，也决定显存**：所有 rank 必须一致；缓冲按它的最坏情况划分，decode 设小一点能省显存，但换值会触发重新编译。
- **SM 数要前后一致**：hybrid combine 要求 `num_channels % num_sms == 0`，断言提示"you may use a different SM count than dispatch"；EPHandle 里存了 `num_sms`，combine 默认沿用。
- **first call 很慢**：首次遇到新配置要 NVCC 编译。生产上应该预热，或把 `EP_JIT_CACHE_DIR` 放在持久目录。
- **`do_cpu_sync=False` 的输出不是全部有效**：输出按容量分配，GEMM 必须用 `handle.psum_num_recv_tokens_per_expert` 找有效区间；`do_zero_padding` 只清专家之间的对齐空隙，不会让多余容量变成有效数据（README 原话）。
- **PCIe 机器不支持**：`check_nvlink_connections` 在构造时检查；README 也写明 EP 不支持零 SM 的 RDMA 路径。
- **多 plane 网络必须用 hybrid**：`Context` 的断言提示，`allow_hybrid_mode=0` 在 multi-plane 网络上会因 GIN 类型不可用而失败（direct 需要 FULL 连接）。

## 面试自测

1. **问**：DeepEP V2.5 为什么把 rank 数、hidden、top-k 都做成模板参数？代价是什么？
   **要点**：kernel 里共享内存切分、循环上界、整数除法、warp 角色划分全部编译期确定，寄存器分配和展开更好；配合 DeepJIT 只为实际用到的配置编译。代价是首次编译延迟、每种配置一份二进制，`num_max_tokens_per_rank` 变化会重编。
2. **问**：V2.5 怎么保证接收端看到"完成"时数据已经到了？和 V1 LL 有什么不同？
   **要点**：kernel 末尾 `gpu_barrier`：先 `ncclGin::flush` 所有 QP（RC 本地完成 = 对端已 ACK），grid sync 后再用 QP 0 发 `SignalInc` 并等所有 rank。V1 LL 靠同一 QP 上"数据在前、原子计数在后"的 RC 顺序，不等待但要求数据和信号同 QP。
3. **问**：`-(count+1)` 编码解决了什么问题？
   **要点**：缓冲平时是 0，需要区分"还没到"和"到了但是 0"。编码后 0 → 解码 −1 = 未就绪，计数 0 → 写 −1 → 解码 0 = 就绪。读完清零即可复用，不需要额外标志位。
4. **问**：`defer_epilogue=True` 推迟了什么？为什么能提升重叠？
   **要点**：推迟的是 CPU 等计数、分配输出张量、按专家展开的 copy epilogue。主 kernel 发出后立即返回，CPU 可以继续排独立计算；`.wait()` 时才在当前 stream 上收尾。它替代了 V1 LL 的 recv hook，但通信 kernel 仍占 SM。
5. **问**：为什么 hybrid 的 QP 估算是每个 channel 一个，direct 却尽量少？
   **要点**：direct 要和所有 rank 建连接（FULL），QP 多了门铃开销大，注释写的是减少 DB ringing；hybrid 只连 rail 上的同号 GPU（RAIL 连接），每个 channel 一个独立 QP 可以避免 channel 之间抢同一个 SQ。

## 待深入

- 待确认：DeepJIT 的编译缓存键与多进程并发编译行为；确认方式：把 `deepseek-ai/DeepJIT` 加入 `repos.lock` 后读 `deep_jit/backend/cuda/backend.hpp`、`deep_jit/python_api.hpp`。
- 待确认：DeepEP 要求的 NCCL 2.32.3 中 `ginStrongLegacySignals` 的默认值（决定 `ncclGin_VASignalAdd` 是强信号还是弱信号）；确认方式：读对应版本 NCCL 的 `ncclDevCommCreate` 实现或 release notes。
