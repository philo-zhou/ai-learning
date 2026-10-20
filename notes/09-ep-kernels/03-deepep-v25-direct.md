# DeepEP V2.5 direct 模式精读：dispatch、按专家展开、combine

> 层次：中层（贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP@b5a33eef（`deep_ep/include/deep_ep/impls/ep/dispatch.cuh`、`dispatch_copy_epilogue.cuh`、`combine.cuh`、`combine_utils.cuh`、`combine_reduce_epilogue.cuh`，`csrc/kernels/ep/dispatch.hpp`、`combine.hpp`，`csrc/buffers/ep.hpp`）

前置：[02 总览](02-deepep-v25-overview.md)（token 布局、workspace、`NCCLGin`、`gpu_barrier`、计数编码）。

## 一句话

direct 模式下每个 token 对每个**目标 rank**只发一份：发送端用本地 `atomicAdd` 给它在对端 `[我的 rank][槽位]` 里分配一个位置，NVLink 可达就 TMA 直写，否则先 TMA 写进本地发送缓冲再 `gin.put`；同一个 kernel 里另有 4 个 notify warp 并行地统计并交换"每个 rank、每个专家各发了多少"。全组 barrier 之后，PDL 衔接的 copy epilogue 按前缀和把 token 展开成 grouped GEMM 需要的按专家排布。combine 原路返回：专家侧先把同一 token 的多个本地专家输出归约成一份，写回源 rank 的固定行，源 rank 的 reduce epilogue 再把来自不同 rank 的部分和加起来。

## 在调用链中的位置

```text
EPBuffer.dispatch ──► EPBuffer::dispatch ──► launch_dispatch            （num_scaleout_ranks == 1）
                                               └─ dispatch_impl<kIsScaleupNVLink, kDoCPUSync, kReuseSlotIndices, ...>
                      epilogue lambda ──────► launch_dispatch_copy_epilogue
                                               └─ dispatch_copy_epilogue_impl<kDoExpand, kCachedMode, kDoZeroPadding, ...>
EPBuffer.combine  ──► EPBuffer::combine  ──► launch_combine
                                               └─ combine_impl<kIsScaleupNVLink, kUseExpandedLayout, kAllowMultipleReduction, ...>
                      epilogue lambda ──────► launch_combine_reduce_epilogue
                                               └─ combine_reduce_epilogue_impl<...>
```

两种情况会走 direct：单节点（scale-up 就是 NVLink 域，`kIsScaleupNVLink = true`，`team_t = ncclTeamTagLsa`），或 `allow_hybrid_mode=False` 的多节点（`kIsScaleupNVLink = false`，`team_t = ncclTeamTagWorld`，NVLink 可达的 peer 仍走 TMA 直写）。

## 核心概念

### 1. dispatch 的整体时序

```text
           dispatch_impl（cooperative，num_sms 个 block，每 block = 4 notify warp + D 个 dispatch warp）
            │
 t0         ├─ gpu_barrier(kDispatchTag0, 不 flush)        确认所有 rank 都进入本轮：对端已用完共享缓冲、计数器已清零
            │
            ├─ notify warp（所有 SM）                       dispatch warp（所有 SM）
            │   ① 块内 atomicAdd_block 统计                   ① 取 token：TMA 读 hidden → smem，cp.async 读 SF
            │      每 rank（去重）/每专家 的 token 数          ② topk_idx / weights / 源下标写进 smem 里的 token
            │   ② red_add 到 notify_reduction                ③ 按目标 rank 去重，atomicAdd 发送端计数器 → 槽位
            │      高 32 位数 SM，低 32 位累加                ④ NVLink：TMA 直写对端 recv[my_rank][slot]
            │   ③ SM 0：等齐 → 编码 → 发给每个 rank          ⑤ RDMA：TMA 写本地 send[token]，再 gin.put
            │   ④ SM 0：等收齐 R + E 个计数                       到对端 recv[my_rank][slot]
            │   ⑤ SM 0：每专家求和、对齐、写 host workspace
            │   ⑥ SM 0：warp 0 每 rank 前缀和，warp 1 每专家前缀和
            │
 t1         ├─ gpu_barrier(kDispatchTag1, flush)           flush 所有 QP → grid sync → 发信号/等信号
            ├─ cudaTriggerProgrammaticLaunchCompletion()    放行 copy epilogue
            └─ SM 0 清零 scaleup_atomic_sender_count
```

**最重要的一点：计数交换和数据发送是并行的**。因为落点由发送端自己决定（槽位 = 本地原子计数器的旧值），发数据不需要等任何人的计数；计数只是给接收端后续"知道收了多少、怎么展开"用的。V1 normal 必须先 notify 再发数据（落点由接收端分配），这里把这一轮串行依赖消掉了。

### 2. notify warp：统计、网格归约、交换

对应 `dispatch.cuh` 中 `if (warp_idx < kNumNotifyWarps)` 分支。

**块内统计**：每个 warp 处理一个 token，lane `i < k` 读 `topk_idx[token][i]`：

- 专家计数：`atomicAdd_block(expert_count + dst_expert_idx, 1)`，每个 (token, 专家) 都计一次。
- rank 计数：`ptx::deduplicate(dst_rank_idx, lane_idx)` 只让同一 rank 的多个 lane 中的一个计数（`match.any` 后取最高 lane，见 `common/ptx.cuh`），所以 rank 计数是**去重后**的 token 数，正好等于要发给这个 rank 的 token 份数。

**网格归约**：每个 SM 把共享内存里的 `R + E` 个计数以 `(1 << 32) | count` 的形式 `red_add` 到 workspace 的 `notify_reduction[i]`。高 32 位因此数的是"已经加进来的 SM 个数"，SM 0 自旋到它等于 `kNumSMs`，低 32 位就是全网格总数——一次原子操作同时完成了"求和"和"到齐判定"，不需要额外的 grid sync。

**交换**（只有 SM 0 做）：

| 发什么 | 纯 NVLink（`kIsScaleupNVLink`） | 跨节点 |
|---|---|---|
| 发给 rank i 的 token 数 | `gin.put_value` 写到 i 的 `scaleup_rank_count[my_rank]`（`st.relaxed.sys`） | 同一调用，不可达时变成 `ncclGin::putValue` |
| 发给 rank i 上每个专家的 token 数 | 逐个 `put_value`（注释：共享内存和全局内存类型不同，不用 TMA） | 先把编码后的计数写进本地发送区，再对每个 rank 一次 `gin.put` 批量发 `kNumExpertsPerRank × 8` 字节 |

计数都先经 `encode_decode_positive` 编码。之后 SM 0 等自己的接收区 `R + E` 个位置全部就绪（解码 ≥ 0），读出后清零，再：

- 对每个本地专家把 R 个源 rank 的数量相加 → `num_unaligned_recv_tokens_per_expert`，再对齐到 `kExpertAlignment`（grouped GEMM 的 M 方向对齐，DeepGEMM 用 `get_mk_alignment_for_contiguous_layout()`）；`cumulative_local_expert_recv_stats` 不为空时累加进去（给 EPLB 统计负载）。
- `kDoCPUSync` 时把 R 个 rank 计数和本地专家计数编码写进映射的 host workspace，CPU 那边的 epilogue 正在自旋等它。
- warp 0 算每个源 rank 的**包含式**前缀和 → `psum_num_recv_tokens_per_scaleup_rank`；warp 1 算每个本地专家的**排他式**前缀和 → `psum_num_recv_tokens_per_expert`（展开时当作写入游标）。前缀和用 `warp_inclusive_sum` 逐 32 个一段做，注释说 `cub::BlockScan` 太重。

### 3. dispatch warp：取 token、分槽位、发出去

对应 `else` 分支。每个 warp 在共享内存里有一个带 mbarrier 的 token 缓冲（`tma_buffer`），token 按 `dispatch_warp_idx × kNumSMs + sm_idx` 起步、步长 `kNumDispatchWarps × kNumSMs` 轮转分配。每个 token：

1. `tma_store_wait()`：等上一个 token 从共享内存发出去，缓冲可复用。
2. 一个 lane `tma_load_1d` 把 hidden 读进共享内存（挂在 mbarrier 上）；FP8 时 32 个 lane 用 `cp.async` 读 SF，`cp_async_mbarrier_arrive` 挂到同一个 mbarrier。
3. lane `i < k` 读 `topk_idx`、`topk_weights` 写进共享内存 token 的 metadata 段，并算出目标 rank；一个 lane 写 `src_token_global_idx = rank × M + token`；`tma_store_fence()` 让这些普通写对后续 TMA 可见。
4. **分槽位**：`deduplicate` 选出每个目标 rank 的代表 lane，代表 lane 执行 `atomicAdd(scaleup_atomic_sender_count + dst_rank, 1)` 得到槽位；每个 lane 把 `rank × M + slot`（或 −1）写进 `dst_buffer_slot_idx[token][i]`，留给 combine 和 cached dispatch。cached 模式（`kReuseSlotIndices`）直接读回这个值，不再原子分配。
5. 等 mbarrier（`mbarrier_arrive_and_set_tx` + `mbarrier_wait_and_flip_phase`）：hidden 和 SF 都到了。
6. 跨节点时，一个 lane 把整个 token（hidden + SF + metadata）`tma_store_1d` 到本地发送缓冲 `send[token_idx]`——RDMA 的源地址必须在已注册窗口里，共享内存不行。
7. 每个代表 lane 用 `gin.get_sym_ptr<team_t>(recv[my_rank][slot], dst_rank)` 取对端地址：非空（NVLink 可达）就**每个 lane 各自** `tma_store_1d` 直写过去——一个 token 去几个 rank，就有几个 lane 同时发 TMA。
8. 跨节点时等第 6 步的写完成（`tma_store_wait<1>`），对 `dst_ptr == nullptr` 的目标 `gin.put` 从 `send[token]` 到对端 `recv[my_rank][slot]`。

接收端缓冲是 `[R][M]` 的二维数组，第 r 行是"来自 rank r 的 token"，槽位从 0 连续增长。所以第 r 行有效 token 恰好是 `[0, count_r)`，copy epilogue 用每 rank 的前缀和就能把全局下标 i 映射回 (r, slot)。

```text
rank 2 的接收缓冲 recv（direct，R = 4）
          slot:  0      1      2      3    ...   M-1
 from r0 │ t0,7 │ t0,9 │ t0,12│      │ ...       │   count_0 = 3
 from r1 │ t1,0 │      │      │      │           │   count_1 = 1
 from r2 │ t2,3 │ t2,4 │      │      │           │   count_2 = 2   ← 本 rank 自己的 token 也走这条路（本地 TMA）
 from r3 │      │      │      │      │           │   count_3 = 0
 tX,Y = rank X 的第 Y 个 token；每格是一个完整 TokenLayout（hidden + SF + topk + 源下标）
 psum_num_recv_tokens_per_scaleup_rank = [3, 4, 6, 6]
```

### 4. copy epilogue：按专家展开

`dispatch_copy_epilogue_impl`（`dispatch_copy_epilogue.cuh`）用整卡 SM，开头 `cudaGridDependencySynchronize()` 等主 kernel 的数据全部可见（注释提醒 PDL 下不要用 `__ldg`）。

- `do_cpu_sync=False` 时 host 传的 `num_recv_tokens` 是最坏值 `M × R`，kernel 里检测到就改读 `psum_num_recv_tokens_per_scaleup_rank[R-1]`。
- 每个 warp 处理一个全局下标 i：用 lane 并行读前缀和、`exchange` 广播，推进到 i 所在的源 rank，得到 `recv[r][i - start_r]`。
- 一个 lane `tma_load_1d` 整个 token 到共享内存；同时 lane `j < k` 直接从全局读 `topk_idx[j]`（"to tolerate TMA load latency"），筛出落在本 rank 专家区间 `[rank × E/R, (rank+1) × E/R)` 的那些，换算成本地专家号；`master_src_topk_idx` = 落在本 rank 的最高 lane（combine 用它选回写行）。
- 写到哪：
  - 非展开：一行，`recv_x[i]`，`recv_topk_idx[i]` 存本地专家号（不是本地的为 −1）。
  - 展开：每个命中的本地专家一行，行号 `atomicAdd(psum_num_recv_tokens_per_expert + e, 1)`——排他前缀和被当作游标原地推进，结束后变成"对齐起点 + 实际数量"（`EPHandle` 文档里那段 `psum[i] − align(psum[i−1])` 的来历）。cached 展开模式直接读 handle 里存的行号。
  - TMA 把 hidden 写到这些行；SF 由 32 个 lane 按 `recv_sf` 的步长逐列写（`use_tma_aligned_col_major_sf` 时是列主序，给下一个 GEMM 的 TMA 用）；展开模式 `recv_topk_weights` 是一维的，每行一个权重。
- `recv_src_metadata[i] = [src_token_global_idx, r × k + master_lane, 展开后的 k 个行号]`，combine 全靠它原路返回。
- `do_zero_padding`：用 `num_unaligned_recv_tokens_per_expert` 算出每个专家末尾到对齐边界之间的空行，所有 warp 分摊，用 TMA 写全零的共享内存缓冲把它们清零（权重、SF 一并清零）。

展开的行号来自原子操作，专家内部 token 的顺序每次可能不同；`EPBuffer(deterministic=True)` 时 Python 侧的 `EPHandle.deterministic_sort` 会按源下标重排。

### 5. combine：专家侧先归约，再写回源 rank 的固定行

`combine_impl`（`combine.cuh`）。输入 x 是专家输出（BF16，已乘过门控权重——README："the expert computation applies these router weights before combine; combine sums the supplied expert outputs"），按 dispatch 时的接收行组织（展开模式就是展开后的行）。

**回写地址**：源 rank 上的 combine 接收缓冲是 `[L][M]`，L = `kNumTokensInLayout`。每个 token 写到第 `src_token_idx` 行——就是它在源 rank 上的原始下标，接收端不用任何映射。L 那一维选哪一格：

- `kUseRankLayout`（允许多次归约且 R ≤ k）：按专家所在的 rank 号，L = R。
- 否则按 `master_src_topk_idx`（这个 token 落在本 rank 的那几个 top-k lane 里最高的那个），L = k。

一个 token 最多去 `min(R, k)` 个不同 rank，所以 L 取 `min(R, k)` 就够了（`get_num_tokens_in_layout`，`combine_utils.cuh`）。这也是 combine 缓冲比 dispatch 的 `[R][M]` 小得多的原因。

**每个收到的 token 三种情况**（kernel 里注释写着 "3 cases"）：

| 情况 | 条件 | 做法 |
|---|---|---|
| 不归约 | 非展开；或展开但只有一个本地专家命中 | TMA 读一行 → TMA 写到目标 |
| 本地归约 | 展开 + 允许多次归约 + 多个本地专家命中 | `combine_reduce` 把这几行在寄存器里相加（`int4` 向量、展开因子 ≤ 4），结果放共享内存，再 TMA 写出 |
| 全部发送 | 展开 + 不允许多次归约（`kDoExpandedSend`） | 每个命中的专家各写一行到目标的第 k 格，不在专家侧相加 |

目标 NVLink 可达时直接 TMA 写到 `get_sym_ptr` 给的对端地址；否则写到本地发送缓冲 `send[src_rank][src_token_idx]`，等写完再 `gin.put` 到源 rank 的接收缓冲。`topk_weights` 不为空（训练反向）时一并写进 token 的 weights 段。结尾 `gpu_barrier(kCombineTag1, flush)`。

**reduce epilogue**（`combine_reduce_epilogue_impl`）在源 rank 上对每个原始 token：读自己的 `topk_idx`，按目标 rank 去重得到有效格子，`combine_reduce` 把这些格子相加，可选再加 `bias_0`、`bias_1`（共享专家输出或残差），TMA 写到 `combined_x`。

```text
token t（源 rank 0）top-4 = {e5@r1, e6@r1, e9@r2, e14@r3}，R = 4，k = 4 → rank layout
 r1：x[e5] + x[e6] ──本地归约──► rank0.recv[1][t]
 r2：x[e9]          ──直接──────► rank0.recv[2][t]
 r3：x[e14]         ──直接──────► rank0.recv[3][t]
 rank0 epilogue：combined_x[t] = recv[1][t] + recv[2][t] + recv[3][t] (+ bias)
 不允许多次归约时：r1 把 x[e5]、x[e6] 分别写进第 0、1 格，epilogue 一次把 4 份加起来
```

`allow_multiple_reduction=False` 的意义是**精度和确定性**：BF16 只在最后归约一次，结果不随"哪些专家碰巧在同一 rank"而变。DeepInfer 在开启 batch invariant 且启用专家重排时就关掉它（`third_party/DeepInfer/deepinfer/model_executor/parallel_utils/userbuffer.py`），代价是发送量变成每个 (token, 专家) 一份。

## 源码导读

| 路径 | 位置 | 看什么 |
|---|---|---|
| `third_party/DeepEP/csrc/kernels/ep/dispatch.hpp` | `launch_dispatch` | notify warp 固定 4 个（cached 模式 0 个）；dispatch warp 数由共享内存决定 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/dispatch.cuh` | `dispatch_impl` 开头 | `get_qp_mode`、`gpu_barrier(kDispatchTag0)` |
| 同上 | `if (warp_idx < kNumNotifyWarps)` | 块内统计、`(1<<32)|count` 网格归约、计数交换、前缀和 |
| 同上 | `else` 分支 | token 循环：TMA/cp.async、去重、`atomicAdd` 槽位、NVLink TMA 与 `gin.put` |
| 同上 | 结尾 | `gpu_barrier(kDispatchTag1)`、`cudaTriggerProgrammaticLaunchCompletion`、清计数器 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh` | `dispatch_copy_epilogue_impl` | 前缀和定位、展开游标、`recv_src_metadata`、零填充 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/combine.cuh` | `combine_impl` | master 行选择、三种情况、NVLink/RDMA 两条路 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/combine_utils.cuh` | `use_rank_layout`、`get_num_tokens_in_layout`、`combine_reduce` | 接收缓冲行数、向量化归约 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh` | `combine_reduce_epilogue_impl` | 源 rank 上的最终归约与 bias |
| `third_party/DeepEP/tests/ep/test_ep.py` | `test_dispatch_combine` | 用参考实现校验 dispatch/combine 的写法，可当用法示例 |

## 和 V1 LL 对照

| | V1 LL（`hybrid-ep@10d4dd73/csrc/kernels/internode_ll.cu`，历史） | V2.5 direct |
|---|---|---|
| 发送粒度 | 每 (token, 专家) | 每 (token, 目标 rank)，rank 内多专家由接收端展开 |
| 接收缓冲 | `[本地专家][源 rank][M]` | `[源 rank][M]`，显存约为 V1 的 1/本地专家数 |
| 计数 | 每 (专家, 源 rank) 一个原子计数，兼作完成信号 | notify warp 单独交换 rank 级与专家级计数 |
| 完成 | 同 QP、计数在数据后 | `gpu_barrier`：flush 全部 QP + QP 0 信号 |
| 按专家排布 | kernel 内写入时即完成 | PDL 衔接的 copy epilogue |
| 等待期间 SM | recv hook 可释放 | 不支持；用 `defer_epilogue` 推迟收尾 |
| combine 回写 | 每个专家输出各写一份 | 专家侧先本地归约（可关），写回源 rank 原始行 |

更完整的谱系与选型见 [05](05-lineage-and-selection.md)。

## 性能要点与常见坑

- **去重的收益取决于路由**：每个 token 的 k 个专家落在同一 rank 的越多，direct 比 V1 LL 省得越多；均衡路由下期望目标 rank 数就是 [02](02-deepep-v25-overview.md) 第 11 节的 `E[g]`。
- **跨节点 direct 多一次本地拷贝**：RDMA 前要先 TMA 写进发送缓冲。多节点默认走 hybrid，direct 只在 `allow_hybrid_mode=False` 时才跨节点，而且需要 FULL 连接，多 plane 网络上不可用。
- **专家内顺序不确定**：展开行号来自 `atomicAdd`，需要逐位复现时开 `deterministic=True`。
- **combine 输入必须是 BF16**：`EPBuffer::combine` 断言 `x.scalar_type() == torch::kBFloat16`；权重要在专家计算里先乘好。
- **`num_tokens ≤ num_max_tokens_per_rank`**：超出会断言；combine 侧同样检查 `num_combined_tokens`。

## 面试自测

1. **问**：V2.5 direct dispatch 里，为什么计数交换和数据发送可以在同一个 kernel 里并行？
   **要点**：槽位由发送端对本地计数器 `atomicAdd` 决定，写到对端 `[我的 rank][槽位]`，不需要知道对端的任何状态；计数只给接收端做 CPU 同步和按专家展开用，所以可以和数据同时发。V1 normal 由接收端分配落点，必须先 notify。
2. **问**：notify 里 `(1 << 32) | count` 的 `red_add` 有什么用？
   **要点**：每个 SM 加一次，高 32 位累计到达的 SM 数、低 32 位累计计数；SM 0 看到高位等于 `kNumSMs` 就知道全网格加完了，一个原子操作同时完成求和和到齐判定，省一次 grid sync。
3. **问**：combine 的接收缓冲为什么只需要 `min(R, k)` 行而不是 R 行？
   **要点**：一个 token 最多去 min(R,k) 个不同 rank。R ≤ k 时按 rank 号分格；否则按"这个 token 落在该 rank 的最高 top-k lane"分格，lane 号在 [0,k)。每格的行号就是 token 在源 rank 的原始下标。
4. **问**：`allow_multiple_reduction` 关掉会怎样？什么时候要关？
   **要点**：专家侧不再把同一 token 的多个本地专家相加，每个 (token, 专家) 各发一份，源 rank 一次归约。发送量变大，但 BF16 只舍入一次、结果与专家放置无关，适合 batch invariant / 确定性推理或训练。
5. **问**：copy epilogue 如何知道全局第 i 个 token 在接收缓冲的哪里？
   **要点**：接收缓冲第 r 行的有效槽位是连续的 `[0, count_r)`，用每源 rank 的包含式前缀和找到 i 所在的 r，槽位 = i − psum[r−1]。

## 待深入

- 待确认：`combine_impl` 中 `warp_idx = (get_warp_idx() + rank_idx) % kNumWarps` 的旋转目的（推测是让不同 rank 的同号 warp 错开处理区间，避免同时打同一个目标）；确认方式：在多卡上对比去掉旋转前后的 combine 耗时，或查上游提交说明。
- 待确认：direct 跨节点时 notify 用 `gin.put` 批量发专家计数，对 NVLink 可达的 peer 也走网卡（`NCCLGin::put` 注释），实际开销多大；确认方式：在 `allow_hybrid_mode=False` 的双机环境用 `EP_BUFFER_DEBUG=1` 和 nsys 观察。
