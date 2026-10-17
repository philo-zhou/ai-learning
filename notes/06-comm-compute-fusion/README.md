# 06 通算融合（Communication-Computation Fusion）

> 层次：跨层 · 来源：third_party/Triton-distributed@7908e4ea、third_party/DeepEP@93eb6eb2、third_party/nccl@3e46c089、third_party/ai-infra-book@3bdcb4fc（第 5.5.4、6.4.3、6.4.4、7.3.2、9.4.3、10.3.3 节）

## 一句话

通算融合 = 不让通信独占关键路径：要么把通信藏到计算后面（重叠），要么把通信和计算放进同一个 kernel 里按数据块（tile）流水，要么把通信交给不占 SM 的硬件（copy engine、NIC）。对应 JD 中"设计通信算子与计算算子的融合方案、用 Triton 开发通算融合算子、优化端到端通信延迟"。

## 章节索引

| 文件 | 内容 | 状态 |
|---|---|---|
| [01-overlap-taxonomy.md](01-overlap-taxonomy.md) | 重叠/融合分类：stream 级、算子拆分流水、kernel 内融合、copy engine、SM 权衡 | 已写 |
| [02-triton-distributed.md](02-triton-distributed.md) | Triton-distributed 仓库结构、原语、AG+GEMM / GEMM+RS / EP all2all 导读 | 已写 |
| [03-nvshmem-and-device-initiated.md](03-nvshmem-and-device-initiated.md) | NVSHMEM 概念、DeepEP 与 Triton-distributed 的用法、NCCL device API / GIN 导读 | 已写 |
| [04-fusion-design-checklist.md](04-fusion-design-checklist.md) | 设计通算融合算子的检查清单与测试/基准方法 | 已写 |

下图是本章四篇笔记的阅读顺序，以及每篇回答的问题。

```text
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
│ 01 taxonomy     │──►│ 02 triton-dist   │──►│ 03 nvshmem           │──►│ 04 checklist     │
└─────────────────┘   └──────────────────┘   └──────────────────────┘   └──────────────────┘
 重叠/融合五层分类       tile 级融合怎么写       device-initiated 底座       怎么设计、测试、基准
```

相关章节：[笔记总入口](../00-roadmap/README.md)、[01-rdma](../01-rdma/README.md)（device-initiated 通信最终落到的 NIC/QP 机制）、[04-nccl-internals](../04-nccl-internals/README.md)、[05-framework-integration](../05-framework-integration/README.md)（DeepEP 与推理框架如何调用）、[07-perf-troubleshooting](../07-perf-troubleshooting/README.md)、[08-interview](../08-interview/README.md)、[JD 拆解](../00-roadmap/02-jd-breakdown.md)、[学习路线](../00-roadmap/03-learning-path.md)。

## 为什么需要通算融合

### 1. 通信暴露在关键路径上

- TP 的每层 AllReduce、MoE 的 dispatch/combine、ZeRO-3 的首个 AllGather 和末个 ReduceScatter，都夹在计算依赖链中间，数据没到后一个算子就不能开始。
- 书中 `third_party/ai-infra-book/manuscripts/10-训练系统.md` 第 10.3.3 节的算例：3 ms 的归约若与 5 ms 无关计算同时跑可完全藏住；但链路若先被 4 ms 的专家交换占住，归约被推迟，有 2 ms 暴露在计算之后，直接延长训练步。结论：数据依赖决定最早开始时刻，资源争用把它往后推，两者共同决定"通信等待时间"。
- 书中 `third_party/ai-infra-book/manuscripts/09-分布式推理.md` 第 9.4.3 节：dispatch / 专家计算 / combine 串行 0.827 ms，拆成两个 micro-batch 流水理想值 0.581 ms；这里最长的一段是通信，流水节奏由网卡决定。

### 2. kernel launch 与主机交接开销

- 小消息场景下，通信的固定成本（launch、主机提交、CPU 代理线程轮询）比传输本身还大。书中 `third_party/ai-infra-book/manuscripts/06-超节点.md` 第 6.4.4 节：Qwen3-32B 八卡算例里 128 次归约的启动开销约 1.47 ms。
- 同书第 5.5.4 节（`third_party/ai-infra-book/manuscripts/05-算子与运行时.md`）用 persistent kernel 说明：少一次 kernel launch 加上按块重叠，是融合收益的两个来源；两个阶段耗时越接近，按块重叠省得越多。
- 第 6.4.4 节还对比了 CPU 代理、GPU SM、NIC 上处理器三种发起位置，控制路径分别穿越 PCIe 三次、两次、零次。

### 3. SM 争用（通信与计算抢同一批资源）

- 传统 NCCL kernel 本身占 SM、占 HBM 带宽；重叠后矩阵乘变慢。书中第 6.4.3 节：实验 6-5 里 64 MiB 通信与矩阵乘同时运行只省约 2.7 ms，而不是矩阵乘的全部 11 ms，原因就是资源争用。
- 第 9.4.3 节同样指出：网卡 DMA 与 GEMM 争用 HBM，使 dispatch/combine 各慢约 49% 时，两 micro-batch 流水收益完全消失。
- 所以通算融合不只是"并发"，还要回答"通信占几个 SM、占多少 HBM 带宽、能否换成 copy engine"。见 [01-overlap-taxonomy.md](01-overlap-taxonomy.md)。

下图把上面三类问题和面试自测第 2 题的三类收益来源对应起来。

```text
 problem                          gain source
┌───────────────────────┐       ┌───────────────────────┐
│ exposed comm          │──────►│ hide time behind comp │   ← 通信暴露在关键路径 → 隐藏时间
├───────────────────────┤       ├───────────────────────┤
│ launch / host handoff │──────►│ cut fixed cost        │   ← launch 与主机交接 → 降低固定开销
├───────────────────────┤       ├───────────────────────┤
│ SM / HBM contention   │──────►│ free SMs / use CE     │   ← SM 争用 → 让出 SM 或改走专用引擎
└───────────────────────┘       └───────────────────────┘
```

## 面试自测

1. 为什么"通信与计算并发执行"不等于"通信被完全隐藏"？
   - 要点：共享 SM / HBM / 互联带宽，并发后各自变慢；评价应比较同时运行的总时间而不是单独时间相减；还受依赖（最早开始时刻）和链路占用约束。
2. 通算融合的三类收益来源是什么？
   - 要点：把通信藏进计算（隐藏时间）、减少 launch 与主机交接（降低固定开销）、让出 SM 或改走专用引擎（减少争用）。
3. 小消息与大消息场景，通算融合的主要收益有何不同？
   - 要点：小消息看延迟，收益主要来自省 launch / 主机往返 / 同步；大消息看带宽，收益主要来自与计算重叠，同时要防 HBM/SM 争用。

## 待深入

- 待确认：本机无 GPU，上述各文档中的性能数字均来自书籍或上游文档，未在本地复现；确认方式：在 GPU 机器上运行 02 中列出的测试脚本。
