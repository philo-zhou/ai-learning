# DeepEP V2.5 hybrid 模式精读：rail 上 RDMA、节点内 NVLink 转发、三级归约

> 层次：中层（贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP@b5a33eef（`deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh`、`hybrid_combine.cuh`、`dispatch_copy_epilogue.cuh`、`combine_reduce_epilogue.cuh`，`csrc/buffers/ep.hpp`、`csrc/kernels/ep/dispatch.hpp`、`combine.hpp`）、third_party/DeepInfer@774ce244

前置：[02 总览](02-deepep-v25-overview.md)、[03 direct](03-deepep-v25-direct.md)（notify 网格归约、copy epilogue、combine 的三种情况在 hybrid 里复用）。

## 一句话

多节点默认走 hybrid：一个 token 跨节点时，只经 RDMA 发给**目标节点上与自己 local rank 相同的 GPU**（rail 上的 peer）一次，到达后由那张卡的 forward warp 经 NVLink 用 TMA 扇出给节点内真正的目标 GPU。kernel 按 warp 划分 channel，每个 (channel, 目标节点) 有一段足够放下该 channel 全部 token 的接收区，所以跨节点那一级只需要单调递增的 tail 信号、不需要环形队列和背压；节点内那一级用 forwarder 本地 `atomicAdd` 分槽位。dispatch 时记下的转发元数据和 linked list 让 combine 原路返回，并在三处归约：专家所在 GPU 内、forwarder 节点内、源 GPU 跨节点。

## 在调用链中的位置

```text
EPBuffer::dispatch ─► launch_dispatch（num_scaleout_ranks > 1）─► hybrid_dispatch_impl
                   ─► epilogue ─► dispatch_copy_epilogue_impl（kDoCreateLinkedList = true）
EPBuffer::combine  ─► launch_combine ─► hybrid_combine_impl
                   ─► epilogue ─► combine_reduce_epilogue_impl（kNumRanks = 节点数）

team：ncclTeamTagRail（各节点上同号 GPU，RDMA）    ncclTeamTagLsa（本 NVLink 域，TMA 直写）
```

DeepInfer 固定 `allow_hybrid_mode=True`，prefill 与 decode 都走这条路径，只用 `do_cpu_sync` 区分（`third_party/DeepInfer/deepinfer/model_executor/models/v4_modules/mlp.py`）。

## 核心概念

### 1. channel、warp 角色与缓冲

**channel = 一个 warp**。`EPBuffer::dispatch`（`csrc/buffers/ep.hpp`）按共享内存算每 SM 的 channel 数：

```text
num_channels_per_sm = min( (smem − notify 区) / dispatch token 字节数,  32 − 4,
                           smem / combine token 字节数 ) / 2          ← 两类 warp 各占一半
                    ≤ kNumMaxChannelsPerSM = 8；prefer_overlap_with_compute=False 时再 ≤ 4
num_channels = num_sms × num_channels_per_sm
```

dispatch kernel 每个 block = 4 个 notify warp + C 个 scale-out warp + C 个 forward warp（C = 每 SM channel 数）。第 c 个 scale-out warp 和第 c 个 forward warp 属于同一个 channel。token 按 `token_idx = channel_idx + n × num_channels` 分给 channel，所以每个 channel 最多 `kNumMaxTokensPerChannel = ⌈M / num_channels⌉` 个 token。

缓冲（`hybrid_dispatch.cuh` 开头，S = 节点数、U = 每节点 GPU 数、M = `num_max_tokens_per_rank`）：

```text
scaleup_buffer        [U][S × M]                          节点内接收：第 j 行 = 由本节点 GPU j 转发来的 token
scaleout_send_buffer  [1][M]                              跨节点发送前的暂存（RDMA 源必须在窗口里）
scaleout_recv_buffer  [S][num_channels × kNumMaxTokensPerChannel]
                       └─ get_rank_buffer(src_node).get_channel_buffer(c)：每个 (源节点, channel) 一段，
                          长度 = 该 channel 最多的 token 数 → 不会写满，不需要回绕
```

### 2. 一个 token 的完整路径

```text
节点 A                                          节点 B
GPU j（源）                                     GPU j（forwarder，rail peer）          GPU k（目标专家所在）
 scale-out warp c                                forward warp c
 ① TMA 读 token → smem                            
 ② 按目标节点去重，槽位 = tail[c][B]（寄存器）
 ③ TMA 写 scaleout_send[token]
 ④ gin.put<Rail> ───── RDMA ─────────────────► scaleout_recv[A][c][slot]
 ⑤ 每 6 个 token：red_add_rel(tail) ─────────► signaled_tail[c][A]
                                                 ⑥ 轮询各源节点的 tail，取一批（≤ 6）
                                                 ⑦ TMA 读 token → smem，看 topk 落在本节点哪些 GPU
                                                 ⑧ 按目标 GPU 去重，atomicAdd → 节点内槽位
                                                 ⑨ TMA 写（NVLink）──────────────────► scaleup_buffer[j][slot']
                                                 ⑩ 记 token_metadata_at_forward、linked list 下标
目标就在节点 A 本身时：②之后直接 TMA 写本地 scaleout_recv[A][c][slot]，不走网卡（"Local rank can be bypassed"）
```

同一个 token 去节点 B 的 3 张卡，跨网只走一次；去 3 个节点，跨网走 3 次、各落在本 rail 上。注释和 `EPBuffer` docstring 都说 hybrid "more friendly to multi-plane/multi-rail networks"：流量只在同号 GPU 之间，天然对齐 rail-optimized 组网。

### 3. notify：两级计数交换

`if (warp_idx < kNumNotifyWarps)` 分支，前半段和 direct 一样（块内统计 → `(1<<32)|count` 网格归约），之后 SM 0 分两级：

1. **跨节点**：每张卡把"发往节点 n 上每张 GPU 的 token 数（按 rank 去重）"和"发往节点 n 上每个专家的数量"编码后，用 `gin.put<ncclTeamTagRail>` 发给节点 n 上的同号 GPU——也就是会替它转发的那张卡。
2. **节点内**：每张卡对收到的 S 份计数求和（`recv_and_reduce`，读到就清零），得到"我要转发给本节点 GPU k 多少个 token、给每个专家多少个"：
   - rank 级：`gin.put_value<ncclTeamTagLsa>` 写 `(U << 32) | count` 到 GPU k 的 `scaleup_rank_count[我]`；
   - 专家级：`gin.red_add_rel<ncclTeamTagLsa>` 加 `(1 << 32) | count` 到 GPU k 的 `scaleup_expert_count[e]`。
3. **收齐判定统一用高 32 位**：rank 级每格只被一个 forwarder 写一次，高位直接写成 U；专家级每格被 U 个 forwarder 各加一次 `1 << 32`，加满也是 U。所以接收方只要检查 `(status >> 32) == kNumScaleupRanks`。

之后和 direct 相同：写 host workspace、统计、按 forwarder 的包含式前缀和与按专家的排他式前缀和。注意这里的"每 scale-up rank"指的是**转发者**，`psum_num_recv_tokens_per_scaleup_rank` 描述的是 `scaleup_buffer` 的 U 行各有多少 token。

### 4. scale-out warp：channel 私有 tail，不用原子

`else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps)` 分支。每个 lane 在寄存器里保存一个节点的 tail（`stored_scaleout_tail`，要求 S ≤ 32）：

- **槽位**：lane `i < k` 算出目标节点，`ptx::exchange(stored_scaleout_tail, 目标节点)` 取该节点当前 tail 作为槽位，`deduplicate` 保证每个节点只用一次；然后对目标节点掩码里的每一位 tail 加 1。channel 私有，不和别的 warp 竞争，所以**不需要原子操作**。
- **发送缓冲可省**：如果目标只有本节点（`scaleout_rank_mask ^ (1 << 本节点) == 0`），跳过写 `scaleout_send_buffer`。
- **预取**：`preload_next_token` 在发 RDMA 之前就开始 TMA 读下一个 token（注释："overlapping with the IBGDA issues"）。
- **发送**：对每个远端目标节点 `gin.put<ncclTeamTagRail>(..., ncclGinOptFlagsAggregateRequests)`——只写 WQE 不敲门铃（见 [02](02-deepep-v25-overview.md) 第 8 节）。
- **tail 信号**：`update_scaleout_tail` 每推进 `kScaleoutUpdateInterval = 6` 个 token，对每个目标节点 `gin.red_add_rel<ncclTeamTagRail>` 一次，加的是"新 tail − 旧 tail"；值用 `pack2(finish_flag, tail)` 打包成 int64，结束时 `update_scaleout_tail(true)` 把 finish 位也加上。这一步不带聚合标志，会把之前积压的门铃一起敲掉。
- **顺序依赖**：tail 走 `ncclGin::signal`，与数据 put 在同一个 GIN context（同一 QP）。接收方看到 tail 就去读数据，依赖的是强信号语义"可见即之前的 put 已完成"，见 [02](02-deepep-v25-overview.md) 的信号强弱一段与待深入项。

和 V1 normal 的区别：V1 normal 的跨节点缓冲是环形队列，发送方要读接收方回写的 head 判断有没有空位；这里每段接收区按 channel 最大 token 数分配，tail 只是"进度通知"，没有反向流控。

### 5. forward warp：轮询、转发、记账

`else` 分支，在 forwarder（rail peer）上运行：

1. **选源节点**：`wip_mask` 收集"有新数据或还没结束"的源节点，从上次处理的节点之后 round-robin 选下一个；等它的 tail 前进或 finish（`ld_acquire_sys` 读 `scaleout_channel_signaled_tail[c][src]`，`unpack2` 拆出 finish 位和 tail）。
2. **取一批**：最多 `kNumSlotsPerForwardChunk = 6` 个槽位。每个 token：TMA 读进共享内存；从 metadata 里的全局 `topk_idx` 减去本节点专家起点，算出落在本节点哪张 GPU。
3. **linked list 下标**：每个 lane 持有"本 channel 已转发给 GPU p 的 token 数"（`stored_scaleup_send_counters`，U > 32 时每 lane 管多个）；用 `exchange` 取出目标 GPU 的当前计数作为这个 token 在 (channel, GPU p) 序列中的序号，经 `transform_linked_list_idx` 变成目标卡上 `channel_linked_list[c][序号][我]` 的扁平下标，写进 token 的 `linked_list_idx` 字段，跟着 token 一起发过去。
4. **节点内槽位**：按目标 GPU 去重，代表 lane `atomicAdd(scaleup_atomic_sender_count + p, 1)`——同一张 forwarder 上所有 channel 共享这个计数器，所以要原子；cached 模式改读 `dst_buffer_slot_idx`。
5. **TMA 写过去**：`gin.get_sym_ptr<ncclTeamTagLsa>(scaleup_buffer[我][slot'], p)` + `tma_store_1d`。
6. **记账**：`token_metadata_at_forward[c][n] = [src_token_global_idx, 是否本批最后一个, k 个目标 GPU 号, k 个节点内槽位]`，同时写 `dst_buffer_slot_idx`（cached dispatch 复用）。
7. **收尾**：metadata 末尾写 −1 作结束标记；对每个 NVLink peer 用 `st_relaxed_sys` 把它的 `channel_scaleup_tail[c][我]` 写成 linked list 的末尾下标；清零自己的 scale-out tail。

结尾只做 scale-up barrier（`gpu_barrier(..., do_scaleout = false, do_scaleup = true)`），注释："As scale-out tokens have already been consumed by forwarders, no need to do scale-out barrier again"——forwarder 已经按 tail 读完了所有跨节点数据，跨节点一级的完成已经隐含在这里。

### 6. copy epilogue 里建 linked list

同一个 `dispatch_copy_epilogue_impl`，hybrid 非 cached 时 `kDoCreateLinkedList = true`：

- 处理接收行 i 时，取这个 token 里 master lane 的 `linked_list_idx`，执行 `channel_linked_list[linked_list_idx] = i`。
- 全部处理完后，按 `channel_scaleup_tail[c][j]` 记的末尾位置写 −1，并清零 tail。

结果是目标卡上的一张表：

```text
channel_linked_list[c][n][j] = 第 j 张 forwarder 在 channel c 上转发给我的第 n 个 token，
                                在我的 combine 输入（dispatch 输出）里是第几行；−1 = 结束
```

`recv_src_metadata[i][1]` 在 hybrid 下存的是 `(槽位) × k + master_lane`（direct 存 `源 rank × k + master_lane`），combine 用它找回 forwarder 上的槽位。

### 7. hybrid combine：沿 linked list 回送，forwarder 重放 dispatch

`hybrid_combine_impl`（`hybrid_combine.cuh`），每 block = C 个 scale-up warp + C 个 forward warp。4 或 8 channel/SM 且非展开时，用 `warpgroup_reg_realloc` 把寄存器从 scale-up warp（40 个）挪给做归约的 forward warp（216 个）。

**scale-up warp（在专家所在 GPU 上）**：

- 每个 lane 管一个 forwarder j，从 `channel_linked_list[c][n][j]` 读下一个 token 的行号；所有 lane 都读到 −1 就结束。
- 在有 token 的 forwarder 之间 round-robin；对选中的 token 用和 direct 一样的三种情况（直接 / 本地专家归约 / 全部发送）得到一份部分和，TMA 写到 forwarder j 的 `scaleup_buffer`：rank layout 时写 `[我][dispatch 时的槽位]`，否则写 `[master_topk][源节点 × M + 源 token]`。
- 每处理 `kNumScaleupUpdateInterval = 3` 批，`st_release_sys` 把"已发给 j 的数量"写到 j 的 `channel_scaleup_tail[c][我]`（注释："`st.release.sys` is pretty slow, so do it by an interval"）。

**forward warp（在 forwarder 上）**：

- 按 `token_metadata_at_forward[c][*]` **以 dispatch 时的顺序重放**，直到遇到 −1。每个 token 的 metadata 告诉它当时转发给了哪些 GPU。
- 对这些 GPU，等它们的 tail 超过"已从它收到的数量"——因为 linked list 的顺序就是 forwarder 当时的转发顺序，两边对第 n 个 token 的认识一致。
- **节点级归约**：允许多次归约时，把这些 GPU 写来的部分和相加（`combine_reduce`），权重取 master lane 的那份；写进 `scaleout_send_buffer[c][n]`，源节点就是本节点时直接写本地接收区。
- **流水**：RDMA 推迟一拍发——归约下一个 token 的同时，`flush_last_tma_and_issue_rdma` 等上一个 token 的 TMA 写完并 `gin.put<ncclTeamTagRail>` 回源节点的同号 GPU（也就是原始发送者），目标是它的 combine 接收区 `[本节点 或 master_topk][源 token 下标]`。只有"本批最后一个"不带聚合标志，按批敲门铃。
- 不允许多次归约时，跳过归约，把每个 (token, 专家) 的那份逐个转发回去。
- **收尾**：清零 scale-up tail；`gin.flush`，再对每个 rail peer `red_add_rel` 一个 `pack2(1, 0)`，并等每个 rail peer 发来的同一个值——相当于一次跨节点 barrier。kernel 末尾注释 "No barrier at epilogue"。

**reduce epilogue（在源 GPU 上）**：`combine_reduce_epilogue_impl` 中 `kNumRanks = kNumScaleoutRanks`，按目标**节点**去重，把来自各节点的部分和相加，再加 bias。

```text
token t（节点 A 的 GPU j）top-4 = {B 上 GPU k1 的 e1、e2；B 上 GPU k2 的 e3；C 上 GPU k3 的 e4}
 ① 专家所在 GPU 内：k1 上 x[e1] + x[e2]                        （scale-up warp 本地归约）
 ② forwarder 节点内：B/j 上 (k1 的部分和) + (k2 的 x[e3])        （forward warp 归约）
    C/j 上只有 k3 的 x[e4]
 ③ 源 GPU 跨节点：A/j 上 (来自 B 的和) + (来自 C 的和) (+ bias)   （reduce epilogue）
 跨网字节：dispatch、combine 各 2 份（B、C 各一份），不是 4 份
```

## 源码导读

| 路径 | 位置 | 看什么 |
|---|---|---|
| `third_party/DeepEP/csrc/buffers/ep.hpp` | `EPBuffer::dispatch` 中 "Decide number of channels" | channel 数怎么由共享内存决定 |
| 同上 | "Hybrid mode handles" | `dst_buffer_slot_idx`、`token_metadata_at_forward`、`channel_linked_list` 的形状与含义注释 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh` | 开头 | 三块缓冲、`get_qp_mode`（每 channel 一个 QP） |
| 同上 | `if (warp_idx < kNumNotifyWarps)` | `recv_and_reduce`、rail 与 LSA 两级计数、高 32 位判齐 |
| 同上 | scale-out 分支 | `update_scaleout_tail`、`preload_next_token`、寄存器 tail 分槽位 |
| 同上 | forward 分支 | round-robin 选源节点、`transform_linked_list_idx`、`token_metadata_at_forward` |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh` | `kDoCreateLinkedList` 两处 | linked list 的写入与结束标记 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_combine.cuh` | scale-up 分支 | 沿 linked list 回送、`update_tails` |
| 同上 | forward 分支 | 重放 metadata、等 tail、节点级归约、推迟一拍的 RDMA、结尾跨节点同步 |
| `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh` | `kNumRanks` 模板默认值 | hybrid 下按节点去重 |

## 和前身对照

| | V1 normal（历史） | Hybrid-EP（`hybrid-ep` 分支，历史） | V2.5 hybrid |
|---|---|---|---|
| 跨节点落点 | notify 后由接收端分配，kernel 内环形队列 | AllGather 路由 + `scan` 预先算出 | channel 私有 tail，发送端寄存器里递增 |
| 节点内落点 | 队列 | 预先算出 | forwarder `atomicAdd` |
| 跨节点流控 | head/tail 背压 | 不需要（落点已定） | 不需要（每段按 channel 最大 token 数分配） |
| CPU 同步 | 必须 | 默认有，可关 | `do_cpu_sync` 可选 |
| combine 原路返回 | 记录的 head/tail 与 prefix | 预计算的映射 | `token_metadata_at_forward` + `channel_linked_list` |
| 搬运 / 网卡 | warp LD/ST / NVSHMEM | TMA / DOCA GPUNetIO | TMA / NCCL GIN（GDAKI，底层也是 DOCA GPUNetIO verbs） |

谱系与选型见 [05](05-lineage-and-selection.md)。

## 性能要点与常见坑

- **dispatch 和 combine 的 SM 数要一致**：combine 的 channel 数来自 handle 里 `token_metadata_at_forward` 的第一维，要求能被 `num_sms` 整除，且两类 warp 的共享内存放得下（`launch_combine` 的两个断言）。
- **至少 2 个 SM**：hybrid 的 `gpu_barrier` 让 SM 0 做 scale-up、其余 SM 做 scale-out，`EPBuffer::dispatch` 断言 "Hybrid dispatch requires at least 2 SMs"。
- **S ≤ 32、U ≤ 64**：scale-out 的 tail 每 lane 一个节点（`kNumScaleoutRanks <= 32`），节点内掩码最多 64 位（`kNumScaleupRanks <= 64`）。
- **QP 用量大**：默认每 channel 一个 QP，`num_allocated_qps` 默认 65 或 129；但 RAIL 连接只和 S−1 个 rail peer 建连，总 QP 数随节点数而不是 rank 数增长。
- **tail 信号的强弱语义**：跨节点 tail 用 `ncclGin_VASignalAdd`，如果 NCCL 把旧名映射成弱信号，"看到 tail 就读数据"就不安全；升级 NCCL 时要留意这一点。

## 面试自测

1. **问**：hybrid 跨节点为什么不用环形队列？代价是什么？
   **要点**：token 按 `c + n × num_channels` 分给 channel，每个 (源节点, channel) 的接收区按该 channel 最多的 token 数分配，永远写不满，tail 只是进度通知，不需要 head 回写和背压。代价是 scale-out 接收缓冲按 `S × (M + 1280)` 个 token 的最坏情况占显存。
2. **问**：为什么 scale-out 分槽位不用原子，节点内分槽位要用？
   **要点**：scale-out 的接收区按 channel 划分，一个 channel 只有一个 warp 写，tail 放在寄存器里递增即可；节点内的 `scaleup_buffer[forwarder]` 由这张 forwarder 上所有 channel 共享，只能 `atomicAdd`。
3. **问**：combine 时 forwarder 怎么知道该从哪些 NVLink peer 收、按什么顺序收？
   **要点**：dispatch 时 forward warp 把每个转发 token 的目标 GPU 和槽位记进 `token_metadata_at_forward`，combine 按同样顺序重放；目标 GPU 那边的 `channel_linked_list[c][n][j]` 记录了"forwarder j 在 channel c 上发来的第 n 个 token 是我的第几行"，scale-up warp 沿它回送，两边对第 n 个 token 的认识一致。
4. **问**：hybrid combine 在哪几处归约？为什么能省跨网流量？
   **要点**：专家所在 GPU 内（同 token 的多个本地专家）、forwarder 节点内（同 token 在本节点各 GPU 的部分和）、源 GPU 跨节点（各节点的部分和）。前两级让每个 (token, 目标节点) 只回传一份，跨网字节与 dispatch 相同。
5. **问**：dispatch 结尾为什么只做 scale-up barrier？
   **要点**：forward warp 已经按 tail（含 finish 位）读完了每个源节点的全部跨节点数据并转发出去，跨节点一级的完成在 kernel 内部已经确认；剩下要保证的只是节点内 TMA 写全部可见。

## 待深入

- 待确认：`scaleout_recv_buffer` 大小里加的 `kNumMaxChannels`（1280 个 token）是否就是为 `⌈M / num_channels⌉ × num_channels` 的向上取整留的余量；确认方式：对照 `get_dispatch_buffer_size` 注释 `kNumChannels * kNumMaxTokensPerChannel` 与 `BufferLayout::get_channel_buffer` 的断言推导。
- 待确认：forward warp round-robin 选源节点与每批 6 个 token 的取值依据（延迟与公平性的权衡）；确认方式：改 `kNumSlotsPerForwardChunk`、`kScaleoutUpdateInterval` 在多机上测 dispatch 耗时。
- not-run：hybrid 模式需要至少 2 个节点 + RDMA，本机无法运行 `third_party/DeepEP/tests/ep/test_ep.py`。
