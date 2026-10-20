# 03 集合通信算法

> 层次：中层 · 来源：third_party/nccl@3e46c08、third_party/nccl-tests@afd59ab、third_party/ai-infra-book@3bdcb4fc（`manuscripts/06-超节点.md` 6.2、6.4 节；`manuscripts/07-数据中心网络.md` 7.2、7.6 节）

## 一句话

本章对应岗位要求里的"熟悉常见集合通信算法（Ring、Tree、Halving-Doubling），了解 NCCL 实现原理"。先讲原语的语义，再用 α-β 模型推导各算法的代价，最后对照 NCCL 源码看它实际实现了哪些算法和协议、怎么选。

## 阅读顺序

| 笔记 | 内容 | 读完能回答 |
|---|---|---|
| [01-primitives.md](01-primitives.md) | Broadcast / Reduce / AllReduce / ReduceScatter / AllGather / AllToAll / Send-Recv 的语义、组合关系，以及在 DP、TP、SP、EP、PP 中各用在哪 | 为什么 AllReduce = ReduceScatter + AllGather；ZeRO/FSDP、SP、MoE 分别用哪些原语 |
| [02-alpha-beta-model.md](02-alpha-beta-model.md) | α-β(-γ) 模型；Ring、Double Binary Tree、Recursive Doubling、Halving-Doubling、Bruck、分层算法的推导；algbw 与 busbw | 给定 n、M、α、β，哪种算法更快；busbw 为什么乘 2(n-1)/n |
| [03-nccl-algorithms.md](03-nccl-algorithms.md) | NCCL 的 Ring / Tree / CollNet / NVLS / PAT 算法和 LL / LL128 / Simple 协议；`src/tuning/`、`src/enqueue/`、`src/device/` 源码导读 | NCCL 怎么在算法×协议之间选；`NCCL_ALGO` / `NCCL_PROTO` 怎么生效 |
| [04-all2all.md](04-all2all.md) | All2All 的 pairwise、直接 P2P、Bruck、分层算法；MoE dispatch/combine 的特殊性 | 为什么 MoE All2All 难优化；NCCL 的 `ncclAlltoAll` 是怎么实现的 |
| [05-handwrite-ring-fullmesh.md](05-handwrite-ring-fullmesh.md) | **手写题**：Ring AllReduce / RS / AG、Full-mesh RS / AG / All2All 的 MPI 伪代码；Full-mesh one-shot / two-shot AllReduce 的 GPU kernel 伪代码（IPC + flag 屏障，对照 vLLM custom allreduce） | 面试白板上怎么写、下标怎么推、怎么避免死锁、何时选 Ring 或 Full-mesh |

五篇之间的依赖关系（箭头表示"读后面一篇时会用到前面一篇的结论"）：

```text
01 primitives ──────► 02 alpha-beta model ──────► 03 NCCL algo x proto selection
 (what each rank       (rounds vs bytes:            (src/tuning/ applies the same
  holds before/after)   Ring, Tree, HD, Bruck)       model with tuned constants)
        │                      │
        │                      └──────────────► 04 All2All (pairwise / Bruck / MoE skew)
        │                      │
        └──────────────────────┴──────────────► 05 handwrite Ring / Full-mesh
                                                      ▲
                                    labs/ccl_sim ─────┘  (runnable reference + tests)
```

## 配套实验

`labs/ccl_sim/`（说明见 [labs/README.md](../../labs/README.md)）：用 Python list 模拟 N 个 rank 的 buffer，逐 step 执行 Ring、Recursive Doubling、Halving-Doubling、Binary Tree、Full-mesh（one-shot / two-shot）AllReduce，以及 Pairwise / Full-mesh All2All，并用 α-β 模型对比耗时。

```bash
python3 -m unittest discover -s labs -t .   # 正确性、step 数、发送量、"公式 = 模拟记录"一致性
python3 -m labs.ccl_sim                     # 不同消息大小下各算法的估算耗时对比表
```

## 与其他章节的关系

- 下层：`notes/01-rdma/`（α、β 从哪来：网卡、RoCE/IB）与 `notes/02-scale-up/`（NVLink/NVSwitch、NVLS 的硬件基础）。
- 同层：`notes/04-nccl-internals/`（channel、transport、proxy 的细节，本章的 03 只讲算法和协议的选择）。
- 上层：`notes/05-framework-integration/`（PyTorch/vLLM/SGLang/DeepEP 怎么调用这些原语）。
- 实践：`notes/07-perf-troubleshooting/`（用 nccl-tests 测 busbw，对照本章的模型）。

## 待深入

- 各外部仓库的固定 commit 以 `docs/references/third-party.md`（及其指向的 `third_party/repos.lock`）为准；本章笔记头部的短 commit 与其一致。
- 本章 `04-all2all.md` 额外引用了 third_party/DeepEP@b5a33ee 的接口名，DeepEP 的深入导读在 `notes/05-framework-integration/`。
- 待确认：Double Binary Tree 在 Reduce/Broadcast 重叠（`runTreeSplit`）下的精确代价公式；确认方式：读 `third_party/nccl/src/device/all_reduce.h` 的 `runTreeSplit` 并与 `third_party/nccl/src/tuning/tree.cc` 的带宽系数对照。
