# All2All：算法与 MoE 场景的特殊性

> 层次：中层（连接上层 EP） · 来源：third_party/nccl@3e46c08（`src/enqueue/enqueue.cc`、`src/enqueue/task_prep/task_classify.cc`、`src/init.cc`、`src/ce_coll.cc`、`src/tuning/sym_model/`）、third_party/nccl-tests@afd59ab（`src/alltoall.cu`、`src/alltoallv.cu`、`doc/PERFORMANCE.md`）、third_party/DeepEP@93eb6eb（`deep_ep/buffers/ep.py`）、ai-infra-book@3bdcb4fc 第 6 章 6.2.6、6.4.4 节，第 7 章 7.2.3 节，第 9 章 9.4.1 节

## 一句话

All2All 是"每个 rank 给每个 rank 发一份不同的数据"，总流量不可约（没有归约可以合并），所以算法能优化的只有轮数和路径；MoE 让它变成不等长（All2Allv）、不均衡、小消息、延迟敏感的版本，瓶颈往往是最忙的那张卡或那张网卡，而不是平均带宽。

## 在调用链中的位置

- 上层：MoE 层的 dispatch（token → 专家所在 rank）与 combine（专家输出 → token 原所在 rank，再按路由权重加权求和）；也用于 Ulysses 式序列并行的 head/序列维转置。
- 本层：NCCL 的 `ncclAlltoAll`（等长）；不等长时由框架用 `ncclGroupStart` + 多对 `ncclSend`/`ncclRecv` 自己拼；专用 EP 库（DeepEP）把路由、打包和通信融合到自定义 kernel 里。
- 下层：NVLink/NVSwitch（节点内）、RDMA 网卡（节点间），见 `notes/01-rdma/`、`notes/02-scale-up/`。

## 核心概念

### 语义与下界

rank r 的发送缓冲分成 n 段，第 j 段发给 rank j；结束时 rank r 的第 j 段 = rank j 原来的第 r 段，相当于把 n×n 的块矩阵转置。设每 rank 发送总量 M：

- 每个 rank 必须送出 (n-1)/n·M、收到 (n-1)/n·M，β 下界为 (n-1)/n·M·β，这也是 nccl-tests 中 AlltoAll 的 busbw 系数 (n-1)/n 的来源（`doc/PERFORMANCE.md`；`src/alltoall.cu` 中 `factor = (nranks-1)/nranks`）。
- 跨节点部分：若每节点 g 个 rank、共 h 个节点，均匀路由时每 rank 有 (h-1)/h 的数据要出节点，*分层不会减少跨节点字节数*（与 AllReduce 的分层不同，All2All 没有可以先在节点内归约的部分），只能把它们合并成更少、更大的消息或分摊到更多网卡。

### 算法

| 算法 | 轮数 | 每 rank 字节 | 说明 |
|---|---|---|---|
| 直接 P2P（一次全部发出） | 1 轮（n-1 条并发消息） | (n-1)/n·M | 所有 Send/Recv 一起提交，由硬件/调度器并发推进；n 大时同时在途的连接多、易拥塞（incast） |
| Pairwise（逐对） | n-1 | (n-1)/n·M | 第 s 轮 rank r 发给 (r+s) mod n、收自 (r-s) mod n（2 的幂时也常用 r XOR s）；每轮每 rank 恰好一收一发，端口无冲突。`labs/ccl_sim` 的 `pairwise_all2all(v)` |
| Bruck | ⌈log₂ n⌉ | ≈ ⌈log₂ n⌉·M/2 | 先本地旋转，第 k 轮把"相对偏移 (j−r) mod n 第 k 位为 1"的所有块打包发给 (r+2^k) mod n，最后逆旋转（标准 Bruck，算法通识）；用约 log₂ n/2 倍的字节换 log 级轮数，只适合小消息、n 大 |
| 分层（两阶段） | 节点内 + 节点间 | 跨节点字节不变 | 先在节点内按"目的节点"重排、聚合，使每个 rank 只负责发往若干节点的数据，再跨节点交换，最后节点内分发；把 n·(h-1) 条小消息合并成更少的大消息，并让同一 rail 上的网卡各自工作 |

Pairwise 的调度可以画成 n × n 的块矩阵：行是发送方 i，列是接收方 j，格子里写"块 i→j 在第几轮发出"，按上表规则它等于 (j - i) mod n。n = 4：

```text
                 dst 0   dst 1   dst 2   dst 3
    src 0          .       1       2       3
    src 1          3       .       1       2
    src 2          2       3       .       1
    src 3          1       2       3       .        "." = 本地块，直接拷贝

    round s=1: 0→1  1→2  2→3  3→0
    round s=2: 0→2  1→3  2→0  3→1
    round s=3: 0→3  1→0  2→1  3→2
```

每个轮号在每一行、每一列都恰好出现一次：每轮每个 rank 正好发一条、收一条，所以端口无冲突。直接 P2P 相当于把 1、2、3 三轮的格子同时发出，每个接收端同一时刻要接 n-1 条流。

表中"分层（两阶段）"一行的数据流示意如下（只画 node A 发出的数据）：

```text
node A (g ranks)
  G0   G1  ...  G(g-1)          each rank holds data for every destination rank
   │    │         │
   ▼    ▼         ▼   phase 1   intra-node: regroup / aggregate by destination node
  each rank of node A now owns node A's data for a few destination nodes
   │
   │                  phase 2   inter-node: few large messages; each NIC / rail
   │                            sends its own share
   ╚══════════════════════════════════════►  node B, node C, ...
                                                  │
                                                  ▼  phase 3   intra-node: scatter
                                                               to the final ranks
```

跨节点字节数与直接 P2P 相同；变的是消息条数（表中所说的把 n·(h-1) 条小消息合并成更少的大消息）和同时在途的连接数。

α-β 估算（`labs/ccl_sim/cost_model.py`）：

- Pairwise：T = (n-1)·α + (n-1)/n·M·β
- Bruck：T ≈ ⌈log₂ n⌉·(α + M/2·β)

`python3 -m labs.ccl_sim` 在 n = 8、α = 5 μs、25 GB/s 时，交叉点在 256 KiB～1 MiB 之间（每 rank 发送总量）：256 KiB 时 Bruck 30.7 μs 对 Pairwise 44.2 μs，1 MiB 时 77.9 μs 对 71.7 μs。

### NCCL 怎么做 All2All

- `ncclAlltoAll` 不走集合算法选择：`src/enqueue/enqueue.cc` 在 `ncclFuncAlltoAll` 分支中，对每个 peer r 用 `p2pTaskAppend` 追加一个 `ncclFuncSend` 和一个 `ncclFuncRecv`（偏移 `r * count * typeSize`）；新路径 `src/enqueue/task_prep/task_classify.cc` 中同样拆分。`src/enqueue/task_prep/task_prep.cc` 调 `ncclTuningCompute` 时显式跳过 AlltoAll。
- P2P 的执行顺序由 `src/init.cc` 的 `ncclP2pSchedule` 在初始化时生成：按节点内 rank 分组（组大小 `NCCL_P2P_SCHEDULE_GROUP_SIZE`），组间的 delta 用二次公式 (x·x + x)/2 mod N 枚举（注释给出 16 rank 时的序列 0, 1, 3, 6, 10, 15, 5, 12, …），组内按 `(local ± delta) % groupSize` 配对。每一 round 给出一对 (sendRank, recvRank)，本质上是一种打散了的 pairwise 调度，目的（我的理解）是让同一时刻各节点发往不同的目的节点、避免 incast。
- nccl-tests 的 `src/alltoall.cu` 优先调用 `ncclAlltoAll`，库中没有该符号时退回到 `ncclGroupStart` + 循环 `ncclSend`/`ncclRecv`；`src/alltoallv.cu` 用同样的方式测不等长版本。
- 本地版本还有基于 copy engine 的实现 `ncclCeAlltoAll` 与分层版本 `ncclHierCeAlltoAll`（`src/ce_coll.cc`），以及对称内存 kernel 的 A2A 代价模型（`src/tuning/sym_model/lsa_a2a.h`、`lsa_a2a.cc`，按 SM100/SM103 给参数表）。

### MoE 场景的特殊性

1. **不等长（All2Allv）**：每对 rank 之间的 token 数由路由决定、每步都变。等长 `ncclAlltoAll` 表达不了，需要先交换计数（一次小的 All2All 或 AllGather）、算偏移、再发数据，多一轮延迟（ai-infra-book 6.4.4 节）。
2. **不均衡（skew）**：热点专家所在 rank 收的多、发回的也多。ai-infra-book 7.2.3 节指出，dispatch 的热点接收方就是 combine 的热点发送方，并给出分析方法：把 rank i → j 的载荷记成矩阵 V_ij，行和是发送量、列和是接收量、割集之和是跨边界量，各自除以带宽取最大值，就是这次交换的时间下界。`labs/ccl_sim` 的 `pairwise_all2allv` 测试 `test_pairwise_all2allv_uneven_blocks` 演示了"一条 50 元素的热点消息决定该轮的 `max_port_elems`"。9.4.1 节进一步说明：EP 越大，每卡平均行数下降得越快，但最忙卡下降得慢，整层按最忙的卡计时。

   用 V_ij 矩阵把这个分析画出来（n = 4，假设 rank 2 上有热点专家；数字是示意用的相对单位，不是实测值）：

   ```text
   dispatch: V[i][j] = bytes from rank i to rank j

                 to 0   to 1   to 2   to 3  │ send (row sum)
     from 0       .      1      4      1    │   6
     from 1       1      .      4      1    │   6
     from 2       1      1      .      1    │   3
     from 3       1      1      4      .    │   6
     ───────────────────────────────────────┼─────
     recv         3      3     12      3
     (col sum)                  ▲
                                └─ lower bound = max(all row / col sums) / B  →  rank 2 recv (12)

   combine:  V' = transpose(V), so rank 2 becomes the busiest sender
   ```
3. **小消息、延迟敏感**：推理 decode 每个 rank 只有几到几百个 token，每条消息只有 KB 级，处在 α 主导区；pairwise 的 (n-1)·α 在 EP 大时很可观，且 dispatch/combine 每层各一次、都在关键路径上。
4. **与计算的依赖**：combine 的数据要等专家算完才就绪；dispatch 可以和 attention 之后的计算重叠，combine 可以和下一个 micro-batch 重叠（两批交错，DeepEP 等库提供这类接口）。
5. **数据格式不对称**：dispatch 常用 FP8（附 scale）、combine 用 BF16，回程字节约为去程的两倍（ai-infra-book 7.2.3 节）。
6. **专用库**：DeepEP 的 `deep_ep/buffers/ep.py` 中 `EPBuffer` 提供 `dispatch` 与 `combine`，README 称其提供"高吞吐和低延迟"两类 EP all-to-all kernel，并以 NCCL Gin 后端复用 NCCL communicator；具体算法与 NCCL `ncclAlltoAll` 的区别留到 `notes/05-framework-integration/`。

## 源码导读

| 路径 | 函数 / 符号 | 作用 |
|---|---|---|
| `third_party/nccl/src/collectives.cc` | `ncclAlltoAll`、`ncclAlltoAllConfigImpl` | API 入口，构造 `ncclInfo{ncclFuncAlltoAll, ...}` |
| `third_party/nccl/src/enqueue/enqueue.cc` | `taskAppend` 中 `info->coll == ncclFuncAlltoAll` 分支、`p2pTaskAppend` | 拆成 n 对 Send/Recv；检查 send/recv buffer 注册以决定能否用 user buffer |
| `third_party/nccl/src/enqueue/task_prep/task_classify.cc` | `coll->func == ncclFuncAlltoAll` 分支、`classifyEnqueueP2pTask` | 新任务路径中的同样拆分 |
| `third_party/nccl/src/init.cc` | `ncclP2pSchedule`、`NCCL_P2P_SCHEDULE_GROUP_SIZE` | 生成 P2P round 调度 |
| `third_party/nccl/src/enqueue/enqueue.cc` | `scheduleP2pTasksToPlan`、`addP2pToPlan` | 按调度把 P2P 任务装进 kernel plan |
| `third_party/nccl/src/ce_coll.cc` | `ncclCeAlltoAll`、`ncclHierCeAlltoAll` | copy engine 版 All2All 及分层版本 |
| `third_party/nccl-tests/src/alltoall.cu`、`alltoallv.cu` | — | 测试实现与 busbw 系数 |
| `third_party/DeepEP/deep_ep/buffers/ep.py` | `dispatch`、`combine` | MoE 专用 EP 通信接口 |
| `labs/ccl_sim/algorithms.py` | `pairwise_all2all`、`pairwise_all2allv` | 等长/不等长 pairwise 的逐步模拟 |

## 性能要点与常见坑

- 不能用固定大小的 `ncclAlltoAll` 测试结果代表 MoE 通信：真实负载不等长、不均衡、还带计数交换（ai-infra-book 6.4.4 节也强调了这一点）。
- 看最忙的卡/网卡/PCIe 插槽，而不是平均带宽：ai-infra-book 7.2.3 节的例子里，32 MiB 全进一张 50 GB/s 网卡要 0.67 ms，分到八张网卡只要 0.08 ms，但两个端口共用一个 32 GB/s PCIe 插槽时又被卡在 1.05 ms。
- 分层 All2All 不减少跨节点字节，它减少的是消息条数（α）和同时在途的连接数。
- Send/Recv 必须放在同一个 group 里；漏掉一对会死锁。
- n 很大时直接 P2P 一次全发容易 incast；pairwise/打散调度更稳。
- 专家放置（把热门专家放近、复制热门专家）改变的是 V_ij 矩阵本身，往往比换通信算法收益更大。

## 面试自测

1. **All2All 和 AllGather 的区别？为什么 All2All 更难优化？**
   要点：AllGather 每个 rank 发同一份数据给所有人，可以转发复用（Ring/Bruck 都利用这点）；All2All 每份数据只有一个目的地，无法复用、无法归约，总流量不可约，只能优化轮数、路径与消息合并。
2. **写出 pairwise All2All 的调度与耗时。**
   要点：第 s 轮（s = 1..n-1）rank r 发给 (r+s) mod n、收自 (r-s) mod n；T = (n-1)α + (n-1)/n·Mβ。
3. **Bruck All2All 什么时候比 pairwise 好？**
   要点：⌈log₂ n⌉ 轮、每轮约 M/2；当 (n-1-⌈log₂ n⌉)α > (⌈log₂ n⌉/2 - (n-1)/n)·Mβ，即小消息、n 大时。
4. **MoE 的 All2All 有什么特殊性？**
   要点：不等长需先交换计数；路由不均衡导致热点卡/网卡成为瓶颈；decode 小消息延迟敏感；与专家计算有依赖、可做 micro-batch 交错重叠；dispatch/combine 数据格式不同。
5. **NCCL 的 `ncclAlltoAll` 是怎么实现的？**
   要点：enqueue 时拆成对每个 peer 的 Send+Recv，按初始化时 `ncclP2pSchedule` 生成的打散调度执行，不经过集合算法的代价模型。

## 待深入

- 待确认：`ncclP2pSchedule` 生成的 round 如何映射到 kernel 中的并发度（每个 kernel 同时推进多少对）；确认方式：读 `third_party/nccl/src/enqueue/enqueue.cc` 的 `scheduleP2pTasksToPlan` 与 `third_party/nccl/src/enqueue/task_sched/p2p_sched.cc`。
- 待确认：`ncclHierCeAlltoAll` 的分层方式与启用条件；确认方式：读 `third_party/nccl/src/ce_coll.cc` 第 2116 行起及第 2405 行附近的调用分支。
- 待确认：DeepEP 高吞吐与低延迟两类 kernel 的通信算法（是否分层、是否用 NVLink 转发跨节点数据）；确认方式：读 `third_party/DeepEP/csrc/kernels/ep/dispatch.hpp`、`combine.hpp`。
