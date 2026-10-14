# 集合通信原语：语义、组合关系与在训练/推理中的位置

> 层次：中层 · 来源：third_party/nccl@3e46c08（`src/nccl.h.in`、`src/include/nccl_common.h`、`src/enqueue/enqueue.cc`）、third_party/nccl-tests@afd59ab（`doc/PERFORMANCE.md`）、ai-infra-book@3bdcb4fc 第 6 章 6.2、6.4.1 节（`manuscripts/06-超节点.md`）

## 一句话

集合通信原语按"开始时每个 rank 持有什么、结束时每个 rank 持有什么"定义；AllReduce 可以拆成 ReduceScatter + AllGather，并行训练/推理的各种切分方式，本质上就是在不同的算子边界插入不同的原语。

## 在调用链中的位置

- 上层：PyTorch `torch.distributed`（DDP/FSDP）、Megatron 的 TP/SP/PP、vLLM/SGLang 的 TP、MoE 框架和 DeepEP 的 EP，按切分方式调用对应原语。
- 本层：NCCL 对外的 C API。本地 `third_party/nccl/src/nccl.h.in` 中能看到 `ncclBroadcast`、`ncclReduce`、`ncclAllReduce`、`ncclReduceScatter`、`ncclAllGather`、`ncclAlltoAll`、`ncclGather`、`ncclScatter`、`ncclSend`、`ncclRecv`、`ncclGroupStart`/`ncclGroupEnd`，以及各自的 `...Config` 变体。
- 下层：每个原语被 NCCL 映射成某种算法（Ring/Tree/NVLS/...）和协议（LL/LL128/Simple），再落到 P2P/SHM/NET transport（见 [03-nccl-algorithms.md](03-nccl-algorithms.md)）。

## 核心概念

### 语义表

记 n 个 rank，rank r 的输入为 `in_r`，"⊕" 为归约运算（sum/max/...，要求满足结合律）。下表的"大小"一栏是 nccl-tests 的口径 S（见 `third_party/nccl-tests/doc/PERFORMANCE.md`），busbw 系数也来自该文档的 Summary 段。

| 原语 | 开始时 | 结束时 | busbw 系数 |
|---|---|---|---|
| Broadcast(root) | root 有 x | 所有 rank 都有 x | 1 |
| Reduce(root) | 各 rank 有 `in_r`（同长度） | root 有 ⊕_r `in_r` | 1 |
| AllReduce | 各 rank 有 `in_r` | 所有 rank 都有 ⊕_r `in_r` | 2(n-1)/n |
| ReduceScatter | 各 rank 有长度 n·k 的 `in_r` | rank r 有归约结果的第 r 段（长度 k） | (n-1)/n |
| AllGather | rank r 有长度 k 的分片 | 所有 rank 都有按 rank 顺序拼接的 n·k | (n-1)/n |
| AllToAll | rank r 有 n 段，第 j 段发给 j | rank r 的第 j 段 = rank j 原来的第 r 段（一次"转置"） | (n-1)/n |
| Gather / Scatter(root) | 分片在各 rank / 整体在 root | 拼接到 root / 分片到各 rank | — |
| Send / Recv | 点对点 | 必须成对出现；多对需要放进同一个 `ncclGroupStart/End` 一起推进 | — |

易混点：

- AllGather 只拼接、不求和；AllToAll 只重新分配、不归约。MoE 的 combine 是"AllToAll 回传 + 恢复 token 顺序 + 按路由权重加权求和"，不是 AllReduce（ai-infra-book 6.4.1 节表格后的说明）。
- ReduceScatter 的结果"谁拿哪一段"是语义的一部分：rank r 拿第 r 段。实现可以在内部换顺序，但对外必须满足这一点。
- NCCL 内部 `ncclFunc_t`（`src/include/nccl_common.h`）除了上面这些，还有 `ncclFuncSendRecv`、`ncclFuncAllGatherV`、`ncclFuncPutSignal` 等内部/新接口类型。

### 组合关系

1. **AllReduce = ReduceScatter + AllGather**：先归约并让每个 rank 留一段（RS），再把这些段收集到所有 rank（AG）。两步各发 (n-1)/n·M，合计 2(n-1)/n·M——这正是 AllReduce busbw 系数 2(n-1)/n 的来源，也是 Ring 与 Halving-Doubling 的骨架。`labs/ccl_sim` 中 `ring_allreduce` 就是 `_ring_rs_phase` 接 `_ring_ag_phase`，测试 `test_allreduce_equals_reduce_scatter_plus_allgather` 验证了 `ring_allgather(ring_reduce_scatter(x)) == ring_allreduce(x)`。
2. **AllReduce = Reduce + Broadcast**：先归约到 root，再广播。这是树形算法（Binary Tree、Double Binary Tree）的骨架。
3. **AllGather 是 ReduceScatter 的"对偶"**：把 RS 的通信过程时间反转、把"累加"换成"拷贝"，就是 AG。所以两者的代价公式只差 γ 项。
4. **Broadcast = Scatter + AllGather**（大消息广播的经典做法，带宽项从 log n·M 降到约 2M）。
5. **AllToAll = n 组成对的 Send/Recv**：NCCL 本地实现正是如此——`src/enqueue/enqueue.cc` 中处理 `ncclFuncAlltoAll` 时对每个 peer r 调用一次 `p2pTaskAppend(... ncclFuncSend ...)` 和一次 `p2pTaskAppend(... ncclFuncRecv ...)`；`src/enqueue/task_prep/task_classify.cc` 里也有同样的拆分。Gather/Scatter 同样拆成 P2P。

### 在训练/推理中各用在哪

| 并行方式 | 切分的对象 | 用到的原语 | 时机与特点 |
|---|---|---|---|
| DP（DDP） | 样本 | 梯度 AllReduce | 反向传播中按 bucket 发起，与反向计算重叠；消息大（MB 级），带宽敏感 |
| DP + ZeRO/FSDP | 参数、梯度、优化器状态 | 前向/反向前 AllGather 参数；反向后 ReduceScatter 梯度 | 把 AllReduce 拆成 RS + AG 并放到不同时间点：每卡只存 1/n 的状态，通信总量与 DDP 同阶 |
| TP（Megatron 列并行 + 行并行） | 一层内的矩阵 | 行并行线性层后 AllReduce 部分和（Attention 与 MLP 各一次/层，反向对称） | 在关键路径上、无法与本层计算重叠；推理 decode 时消息很小（一个 token 的隐藏向量，KB 级），延迟敏感（ai-infra-book 6.2.2、6.4.2 节用 10 KiB 举例） |
| SP（配合 TP 的序列并行） | LayerNorm/Dropout 等逐 token 算子沿序列分片 | 把 TP 的 AllReduce 拆成 ReduceScatter（行并行之后）+ AllGather（列并行之前） | 通信量不变，激活显存降为 1/t（ai-infra-book 6.2.3 节） |
| CP / Ring Attention | 同一序列的注意力 | K/V 块的 Send/Recv 环形传递，或 AllGather K/V | 可与注意力计算流水重叠 |
| EP（MoE） | 专家集合 | dispatch：AllToAll(v)；combine：AllToAll(v) 回传 + 加权求和 | 每对 rank 之间的数据量随路由变化（不等长、不均衡）；推理 decode 小消息、延迟敏感，详见 [04-all2all.md](04-all2all.md) |
| PP | 按层分阶段 | 相邻 stage 之间 Send/Recv 激活（前向）和激活梯度（反向） | 点对点、消息中等；关注气泡而非带宽 |
| 其他 | — | Broadcast：初始权重/RNG 种子同步；AllGather：收集 logits、KV 元数据；Reduce：少见 | — |

## 源码导读

| 路径 | 看什么 |
|---|---|
| `third_party/nccl/src/nccl.h.in` | 对外 API 声明（`ncclAllReduce`、`ncclAlltoAll`、`ncclSend` 等）及 `...Config` 变体 |
| `third_party/nccl/src/include/nccl_common.h` | `ncclFunc_t` 枚举：Broadcast=0、Reduce=1、AllGather=2、ReduceScatter=3、AllReduce=4、SendRecv=5 …… AlltoAll=8、Scatter=9、Gather=10、AllGatherV=11 |
| `third_party/nccl/src/collectives.cc` | 各 API 的入口，把参数填进 `struct ncclInfo` 后交给 enqueue（例如 `ncclAlltoAllConfigImpl`） |
| `third_party/nccl/src/enqueue/enqueue.cc` | `taskAppend` 一带：AlltoAll/Gather/Scatter 被拆成 P2P 任务（`p2pTaskAppend`），其余进入集合任务队列（`collTaskAppend`） |
| `third_party/nccl-tests/doc/PERFORMANCE.md` | 每种原语的 algbw、busbw 定义与推导 |

## 性能要点与常见坑

- 原语的选择决定通信量：DDP 的 AllReduce 与 FSDP 的 RS+AG 总字节数同阶，FSDP 省的是显存而不是带宽。
- TP 的 AllReduce 在关键路径上，TP 组通常放在 NVLink 域内；DP/EP 才跨节点。
- AllToAll 的 busbw 系数和 AllGather 相同，但实际难达到：消息被切成 n 份，每份更小，且 MoE 下各份不等长。
- 原语调用必须在通信组内所有 rank 上以相同顺序发起，否则死锁；多对 Send/Recv 不放进同一个 group 会串行等待对方。
- ReduceScatter/AllGather 要求每个 rank 的分片等长；不等长要用 AllGatherV 或 P2P 自己拼。

## 面试自测

1. **为什么说 AllReduce = ReduceScatter + AllGather？这个分解有什么用？**
   要点：RS 让每个 rank 得到 1/n 的完整归约结果，AG 再把它们发给所有人。好处：①带宽最优算法（Ring、Rabenseifner）都以此为骨架，总发送 2(n-1)/n·M；②ZeRO/FSDP、SP 把两半拆到不同算子边界，换取显存节省。
2. **FSDP 相比 DDP 通信量是否增加？**
   要点：DDP 每步 AllReduce 梯度 ≈ 2M；FSDP 前向 AG 参数 + 反向 AG 参数 + RS 梯度 ≈ 3M（若前向后释放参数需反向重新 AG），即约 1.5 倍；换来每卡状态降到 1/n。
3. **MoE 的 combine 能用 AllReduce 吗？**
   要点：不能直接用。combine 需要把专家输出送回 token 原所在 rank 并按路由权重加权求和，数据按目的地重新分配，是 AllToAll(v) 语义。
4. **TP 的 AllReduce 和 DP 的 AllReduce 在性能上有什么不同？**
   要点：TP 每层都有、在关键路径上，decode 时消息 KB 级，延迟主导（α 项）；DP 梯度桶 MB 级、可与反向重叠，带宽主导（β 项）。所以 NCCL 对两者往往选不同的算法/协议（LL vs Simple）。
5. **NCCL 的 AlltoAll 是一个独立 kernel 吗？**
   要点：本地源码中 `ncclFuncAlltoAll` 在 enqueue 阶段被拆成对每个 peer 的 Send+Recv P2P 任务，走 P2P 调度。

## 待深入

- 待确认：`ncclFuncAllGatherV` 对应的公开 API 名称与使用方式；确认方式：在 `third_party/nccl/src/nccl.h.in` 与 `third_party/nccl/src/scheduler/allgatherv_sched.cc` 中查找。
- 待确认：PyTorch FSDP、Megatron SP 在本地框架源码中的具体调用点；确认方式：后续在 `notes/05-framework-integration/` 中对照框架源码。
