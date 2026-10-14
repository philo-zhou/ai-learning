# α-β(-γ) 代价模型与经典集合通信算法

> 层次：中层 · 来源：third_party/nccl-tests@afd59ab（`doc/PERFORMANCE.md`）、third_party/nccl@3e46c08（`src/tuning/`、`src/graph/trees.cc`）、ai-infra-book@3bdcb4fc 第 6 章 6.4.2 节（式 6-9、6-10）、第 7 章 7.2.1 节（分层归约例题与逐轮链路负载模型）

## 一句话

把一次集合通信看成若干轮点对点传输，每轮代价 α + β·(最忙端口的字节数) + γ·(归约字节数)；不同算法就是在"轮数（α 项）"和"每轮字节数（β 项）"之间做取舍：小消息选轮数少的，大消息选总字节最少且能铺满链路的。

## 在调用链中的位置

- 上层：框架和 NCCL 自己都需要"给定 n、M 估算耗时"来选算法——NCCL 的 `src/tuning/*.cc` 就是一套带经验修正的 α-β 模型（见 [03-nccl-algorithms.md](03-nccl-algorithms.md)）。
- 下层：α 来自 kernel 启动/同步、网卡 doorbell、协议握手；β 来自 NVLink、PCIe、网卡带宽（`notes/01-rdma/`、`notes/02-scale-up/`）。

## 核心概念

### 模型与假设

记号（与 `labs/ccl_sim/cost_model.py` 一致）：

| 符号 | 含义 |
|---|---|
| n | rank 数 |
| M | 消息字节数：AllReduce/ReduceScatter 为每 rank 输入大小，AllGather 为每 rank 输出大小，All2All 为每 rank 发送总量 |
| α | 每轮固定开销（秒） |
| β | 每字节传输时间 = 1/B（B 为单向链路带宽） |
| γ | 每字节归约计算时间（GPU 上通常 γ ≪ β，常被忽略） |

假设：全双工、每个 rank 单端口（同一轮内收、发各自受 B 限制）、同一轮内各消息并行、轮与轮之间严格同步、无拥塞。ai-infra-book 7.2.1 节把它写成一般形式：

T = Σ_r ( α + max_e V_{r,e} / B_e )

即每轮被最忙的那条资源 e 拖住。`labs/ccl_sim/cost_model.py` 的 `cost_from_trace` 就按这个式子给模拟记录计时（e 取每个 rank 的收/发端口）。

**下界**（任何 AllReduce 算法都逃不掉）：

- 延迟：每个 rank 的结果依赖所有 rank 的输入，信息每轮最多翻倍扩散，至少 ⌈log₂ n⌉ 轮 → ⌈log₂ n⌉·α。
- 带宽：每个 rank 至少要送出自己的 (n-1)/n 份贡献、再收回 (n-1)/n 份结果 → 2(n-1)/n·M·β。
- 计算：(n-1)/n·M·γ（归约工作可以均摊到各 rank）。

一个算法若同时贴近两条下界，就在所有消息大小上都好；现实中通常只能贴近一条。

### 各算法推导

#### Ring

把 buffer 切成 n 块。ReduceScatter：第 s 轮 rank r 把某一块发给 r+1，接收方累加；一块从起点出发，经过其余 n-1 个 rank 后包含全部贡献，所以 n-1 轮。AllGather：把归约好的块再沿环转一圈，又是 n-1 轮。每轮每个 rank 只收发 M/n。

T_ring = 2(n-1)·α + 2(n-1)/n·M·β + (n-1)/n·M·γ

- β 项贴近下界、与 n 基本无关；α 项随 n 线性增长。
- 只和两个邻居通信，容易映射到物理拓扑（NVLink 环、每节点只跨一次网卡），且可以切 chunk 流水、开多个 channel 并行跑多个环。
- ai-infra-book 6.4.2 节式 (6-9) 与此相同；书里的发送顺序让 ReduceScatter 结束时卡 r 持有块 r+1，`labs/ccl_sim` 里改成持有块 r（满足 ReduceScatter 的对外语义），二者只差一个旋转。

#### Recursive Doubling（蝶形）

n 为 2 的幂：第 k 轮 rank r 与 r XOR 2^k 交换 *整个* buffer 并累加，log₂ n 轮后人人都有全和。

T_RD = log₂ n·(α + M·β + M·γ)

- α 项贴近下界，β 项是 log₂ n·M，大消息很差。
- n 不是 2 的幂：设 p 为不超过 n 的最大 2 的幂，先把多出来的 n-p 个 rank"折叠"进伙伴（1 轮），p 个 rank 做标准 RD，最后发回（1 轮），共 log₂ p + 2 轮。`labs/ccl_sim` 的 `recursive_doubling_allreduce` 实现了这个折叠。

#### Halving-Doubling（Rabenseifner）

= Recursive Halving 的 ReduceScatter + Recursive Doubling 的 AllGather。ReduceScatter 第 k 轮与距离 n/2^k 的伙伴配对，把当前持有的区间对半分，各保留一半、把另一半发给对方累加，发送量 M/2、M/4、…、M/n；AllGather 反过来，区间每轮翻倍。

T_HD = 2·log₂ n·α + 2(n-1)/n·M·β + (n-1)/n·M·γ

- β 项与 Ring 相同，α 项只有 2 log₂ n——在纯 α-β 模型里它对 2 的幂 *处处不差于 Ring*（`python3 -m labs.ccl_sim` 的表格中 1 MiB 以上 `best` 一列全是 `halving_doubling`）。
- 那为什么 NCCL 主力是 Ring？模型没算进去的东西：①伙伴距离是 2^k，大部分轮次不是物理邻居，在环形/分层拓扑上会和别人抢同一条链路；②要和 log₂ n 个不同的伙伴建连接、各占一份缓冲；③每轮消息大小不同，不利于固定 chunk 流水和多 channel 铺满；④非 2 的幂要折叠。这些在 NVSwitch 全交换的节点内影响小，在跨节点时影响大。
- `labs/ccl_sim` 中 N 不是 2 的幂时 `halving_doubling_allreduce` 刻意抛 `ValueError`。

#### Binary Tree 与 Double Binary Tree

AllReduce = Reduce 到根 + Broadcast。不切块时每轮都传完整 M：

- 二项树（ai-infra-book 式 6-10）：T = 2·log₂ n·(α + M·β)。
- 堆式二叉树（`labs/ccl_sim` 的 `binary_tree_allreduce`，父节点 (i-1)/2）：深度 ⌊log₂ n⌋，有两个孩子的父节点一轮要收 2M，T ≈ 2·⌊log₂ n⌋·(α + 2M·β)；`cost_model.binary_tree_allreduce` 按层精确计算 fan-in。

问题是 β 项随 log n 增长。两个修正：

1. **切块流水**：把 M 切成 k 块，第一块到根后，后面的块在不同层上同时流动。单棵树单阶段 ≈ (D + k - 1)·(α + M/k·β)，D 为深度。k 越大 β 项越接近 M·β，但 α 次数越多；最优 k* ≈ √((D-1)·M·β/α)。
2. **Double Binary Tree**（NCCL Tree）：一棵二叉树里约一半节点是叶子，叶子的"收"端口闲着、内部节点的端口却要承担两个孩子。NCCL 用两棵互补的树，各负责一半数据：`third_party/nccl/src/graph/trees.cc` 的 `ncclGetDtree` 注释写明，第二棵树在 nranks 为偶数时取第一棵的镜像（rank → nranks-1-rank），为奇数时整体平移一位。这样每个 rank 在一棵树里是内部节点、在另一棵里是叶子，负载均衡。

简化模型（`cost_model.double_binary_tree_allreduce`，Reduce 与 Broadcast 不重叠）：

T_DBT(k) ≈ 2·(D + k - 1)·(α + M/k·β)，D = ⌈log₂ n⌉

- k 足够大时 β 项 → 2M·β，与 Ring 同阶；α 项是 2D，对数级。
- 所以 DBT 在 *大规模*（n 大、Ring 的 2(n-1)α 很可观）的中等消息上优于 Ring。NCCL 的树在节点内其实是一条链、只在节点间成树：`src/tuning/tree.cc` 的延迟项是 `2 * ((nRanks/nNodes - 1) * intraLat + log2i(nNodes) * interLat)`。

#### Bruck

适用于 AllGather 与 All2All，且对任意 n 都是 ⌈log₂ n⌉ 轮。

- AllGather：第 k 轮把已有的 2^k 个块发给 r - 2^k，最后本地旋转。T ≈ ⌈log₂ n⌉·α + (n-1)/n·M·β，延迟和带宽都贴近下界。
- All2All：第 k 轮把"目的 rank 编号第 k 位为 1"的所有块一起发给 r + 2^k，每轮约发 M/2。T ≈ ⌈log₂ n⌉·(α + M/2·β)：用 log₂ n/2 倍的字节换来 log 级的轮数，只在小消息、rank 多时划算（见 [04-all2all.md](04-all2all.md)）。

#### 二维 / 分层（hierarchical）

节点内 g 个 rank、节点间 h 个节点（n = g·h），节点内链路快（α_i、β_i），节点间慢（α_e、β_e）：

1. 节点内 Ring ReduceScatter：每个 rank 得到本节点归约结果的 1/g。
2. 节点间：持有同一分片的 h 个 rank（每节点一个，各走自己那条 rail）做 AllReduce，消息只有 M/g。
3. 节点内 Ring AllGather。

T_hier = 2(g-1)·α_i + 2(g-1)/g·M·β_i + 2(h-1)·α_e + 2(h-1)/h·(M/g)·β_e

- 跨节点字节数被除以 g，且 g 张网卡同时工作。ai-infra-book 7.2.1 节的两台 8 卡服务器例题算过：192 MiB 梯度，连续环跨服务器 720 MiB（只用 2 张网卡），交错环 5760 MiB，分层 384 MiB（8 条 rail 并行）。
- 同构二维网格（a×b，如 2D torus）：行内 RS → 列内 AllReduce(M/a) → 行内 AG，总字节仍是 2(n-1)/n·M，但轮数从 2(n-1) 降到 2(a-1)+2(b-1) ≈ 4√n。
- 第 2 步也可以换成 Tree、RD 或交换机内归约（CollNet/SHARP），这就是 NCCL 的 CollNet、NVLS Tree 等算法的思路。

### 汇总

| 算法 | α 项 | β 项 | 适用 |
|---|---|---|---|
| Ring AllReduce | 2(n-1) | 2(n-1)/n·M | 大消息、n 不太大；拓扑友好 |
| Recursive Doubling | log₂ n | log₂ n·M | 小消息 |
| Halving-Doubling | 2 log₂ n | 2(n-1)/n·M | 中大消息、n 为 2 的幂、全交换拓扑 |
| 二项树（不切块） | 2 log₂ n | 2 log₂ n·M | 小消息 |
| Double Binary Tree（流水） | ≈ 2 log₂ n + 2k | → 2M | n 大时的中等消息 |
| Bruck AllGather | ⌈log₂ n⌉ | (n-1)/n·M | 小消息 AllGather、任意 n |
| Pairwise All2All | n-1 | (n-1)/n·M | 大消息 All2All |
| Bruck All2All | ⌈log₂ n⌉ | ⌈log₂ n⌉·M/2 | 小消息 All2All、n 大 |
| 分层 AllReduce | 2(g-1) + 2(h-1) | 节点内 2(g-1)/g·M，节点间 2(h-1)/h·M/g | 节点内外带宽差距大 |

交叉点：令两式相等即可。例如 Ring 与 RD（n = 8）：11α = 1.25·M·β → M* ≈ 8.8·α·B。α = 5 μs、B = 25 GB/s 时 α·B ≈ 122 KiB，M* ≈ 1.07 MiB；`python3 -m labs.ccl_sim` 的表里 1 MiB 一行 Ring 143.4 μs、RD 140.8 μs，正好在交叉点附近。ai-infra-book 6.4.2 节用 α = 0.822 μs、B = 450 GB/s 算出 8 卡 Ring 与二项树的交叉点约 680 KiB。

### algbw 与 busbw（nccl-tests 口径）

摘要自 `third_party/nccl-tests/doc/PERFORMANCE.md`：

- **algbw = S / t**，S 为操作的数据量（AllReduce 为每 rank 的 buffer 大小），适合估算"这么大的操作要多久"。但对集合通信，algbw 的理论峰值随 n 变化，不能直接和硬件带宽比。
- **busbw = algbw × 系数**，用来反映硬件瓶颈链路被用得多充分，可以和 NVLink/PCIe/网卡峰值直接比较。系数：AllReduce 2(n-1)/n；ReduceScatter、AllGather、AlltoAll (n-1)/n；Broadcast、Reduce 1。
- AllReduce 的推导：Ring 下 t = (S/B)·2(n-1)/n，反解 B = (S/t)·2(n-1)/n = algbw·2(n-1)/n。nccl-tests 源码里也是这么算的，例如 `third_party/nccl-tests/src/alltoall.cu` 中 `factor = (nranks-1)/nranks`、`busBw = baseBw * factor`。

我的理解：

- busbw 的系数是按"带宽最优的环形算法"推出来的，不管 NCCL 实际选了 Tree 还是 Ring 都用同一个系数。所以 busbw 是"等效的链路利用率"，不是测出来的链路流量。
- 用 NVLS（NVSwitch 内归约）或 CollNet/SHARP（交换机内归约）时，每个 rank 实际只发送约 M、接收约 M，突破了 2(n-1)/n·M 的"下界"（下界假设归约只能在端点做），于是 busbw 可能超过单链路带宽。
- `labs/ccl_sim/cost_model.py` 提供 `algbw`、`busbw`、`busbw_factor`；测试 `test_ring_bandwidth_term_approaches_2M_over_B` 验证了在 α = 0 时 Ring 的 busbw 恰好等于链路带宽。

## 源码导读

| 路径 | 与本篇的关系 |
|---|---|
| `third_party/nccl-tests/doc/PERFORMANCE.md` | algbw/busbw 定义与各原语系数 |
| `third_party/nccl-tests/src/alltoall.cu` | busbw 系数在测试代码中的实际计算 |
| `third_party/nccl-tests/src/hypercube.cu` | 用 `ncclSend`/`ncclRecv` 写的 Hypercube AllGather：`for (mask=1; mask<nRanks; mask<<=1)` 与 `rank^mask` 交换，数据量每轮翻倍——即 Recursive Doubling 式 AllGather |
| `third_party/nccl/src/graph/trees.cc` | `ncclGetBtree`（交替叶子与内部节点的二叉树）、`ncclGetDtree`（镜像/平移得到第二棵树） |
| `third_party/nccl/src/tuning/tuning_general.cc` | `ncclTuningGetNsteps`：AllReduce 2(nRanks-1)、RS/AG nRanks-1、其他 nRanks——就是 Ring 的轮数；`ncclTuningGetTime`：`lat * latCount + nBytes / (1000 * bw)`，即 α + M/B 的形式 |
| `third_party/nccl/src/tuning/ring.cc`、`tree.cc` | NCCL 的 Ring/Tree 代价模型：Ring 带宽取 `busBw * nRanks / nSteps`，Tree 取 `busBw * 0.5`，两者都对应约 2M/busBw |
| `labs/ccl_sim/algorithms.py`、`labs/ccl_sim/cost_model.py` | 本篇所有公式的可运行版本；测试 `test_analytic_formula_matches_simulated_trace` 验证"解析公式 = 按模拟器逐轮计时" |

## 性能要点与常见坑

- 先算 α·B：消息远小于它时一切看轮数（延迟），远大于它时看总字节和链路利用率。
- Ring 的 α 项随 n 线性增长：千卡规模的小消息 AllReduce 用 Ring 会非常慢，这是 NCCL 引入 Tree 和 PAT 的动机。
- 模型里 HD 处处优于 Ring，但真实拓扑上的链路争用、连接数、非 2 的幂都没算进去——"模型最优"不等于"实现最优"。
- 树不切块时大消息很差；切块数 k 的选择本身是 α 与 β 的权衡。
- busbw 是归一化指标，不同原语之间可以比较；algbw 不能跨 n 比较。
- 分层算法的收益来自"跨节点字节 ÷ g"与"多条 rail 并行"，前提是 rank 到 GPU/网卡的映射正确（同一编号的网卡在同一条 rail 上）。

## 面试自测

1. **推导 Ring AllReduce 的耗时，并说明为什么说它带宽最优。**
   要点：RS、AG 各 n-1 轮，每轮 M/n，T = 2(n-1)α + 2(n-1)/n·Mβ；β 项等于下界 2(n-1)/n·M，与 n 基本无关。
2. **Recursive Doubling 和 Rabenseifner 有什么区别？各适合什么消息？**
   要点：RD 每轮发整个 M、log n 轮，适合小消息；Rabenseifner 先折半 RS 再倍增 AG，2 log n 轮、总字节 2(n-1)/n·M，适合中大消息，要求 2 的幂。
3. **既然 Rabenseifner 在 α-β 模型下不比 Ring 差，为什么 NCCL 主要用 Ring？**
   要点：伙伴非邻居导致链路争用；需要更多连接和缓冲；每轮消息大小变化不利于流水和多 channel；非 2 的幂。Ring 能完美映射到 NVLink/PCIe/每节点一次跨网卡的拓扑上。
4. **Double Binary Tree 解决了二叉树的什么问题？**
   要点：单棵二叉树一半节点是叶子，带宽只用了一半；两棵互补树（镜像或平移）让每个节点在一棵树里是内部节点、在另一棵里是叶子，各带一半数据，配合切块流水后带宽接近 Ring，延迟 O(log n)。
5. **busbw 为什么要乘 2(n-1)/n？busbw 可能超过链路带宽吗？**
   要点：按 Ring 推导，链路上实际流过的字节是 S 的 2(n-1)/n 倍；乘上它后 busbw 与 n 无关，可直接和硬件峰值比较。用 NVLS/SHARP 等网内归约时，可能超过单链路带宽。
6. **分层 AllReduce 为什么快？**
   要点：节点内 RS 后，每个 rank 只需跨节点交换 M/g，且 g 条 rail 并行；跨节点轮数从 2(n-1) 降到 2(h-1)。

## 待深入

- 已读到：`third_party/nccl/src/device/all_reduce.h` 中 Tree/Simple 只在 `CUDART_VERSION` 11.2～11.4 且 SM80+ 时用 `runTreeUpDown`（先整体 Reduce 再整体 Broadcast），否则与 LL/LL128 一样用 `runTreeSplit`：把线程分成两组，一组向上 Reduce、一组向下 Broadcast，同时推进（LL/LL128 按注释是 70%/30% 分配）。所以真实 NCCL Tree 的两个阶段是重叠的，本篇"不重叠"的模型偏保守。待深入：重叠后的精确代价公式。
- 待确认：本模型没有考虑多 channel 并行和 chunk 大小对 α 的影响，NCCL 的 chunk 计算在 `third_party/nccl/src/enqueue/enqueue.cc` 的 `calcCollChunking`；确认方式：读该函数并与 `src/tuning/` 中的 `latCount` 对照。
- 用 nccl-tests 实测若干消息大小的 busbw，拟合出本机的 α、β，再代回 `python3 -m labs.ccl_sim --alpha-us ... --bw-gbps ...` 比较。
