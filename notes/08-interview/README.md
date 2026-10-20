# 08 面试准备

> 层次：求职 · 来源：third_party/ai-infra-book@3bdcb4fc、third_party/nccl@3e46c089、third_party/nccl-tests@afd59aba、third_party/rdma-core@bd3282a1、third_party/Triton-distributed@7908e4ea、third_party/DeepEP@b5a33eef、third_party/vllm@02653fa2

## 一句话

按目标岗位（字节跳动基础设施网络团队-集合通信库研发）JD 整理的题库与项目表述，题目答案只给要点，细节回链到各章节笔记和本地源码路径。

## 章节内容

| 文件 | 内容 |
|---|---|
| [01-question-bank.md](01-question-bank.md) | 题库：RDMA、算法与代价模型、NCCL 原理、scale-up、通算融合、框架集成、性能排查、项目表述 |
| [手写 Ring / Full-mesh](../03-ccl-algorithms/05-handwrite-ring-fullmesh.md) | 白板编码题：Ring 与 Full-mesh 各原语的伪代码、GPU one-shot / two-shot kernel、追问与答案 |

## 与 JD 的对应

| JD 条目（见 [02-jd-breakdown.md](../00-roadmap/02-jd-breakdown.md)） | 题库分区 |
|---|---|
| R2 / Q2 RDMA 与网络通信原理 | 一、RDMA 与网络 |
| Q3 Ring/Tree/Halving-Doubling、R4 AllReduce/All2All 优化 | 二、集合通信算法与代价模型 |
| Q4 NCCL 实现原理、R1 CCL 设计开发 | 三、NCCL 原理 |
| R2 scale-up 互联 | 四、scale-up |
| R3 通算融合（Triton） | 五、通信与计算融合 |
| R5 分布式训练/推理通信问题 | 六、框架与场景 |
| Q5 性能调优与故障排查 | 七、性能与排查 |
| 综合 | 八、项目表述与开放题 |

## 使用方法

1. 先按 [00-roadmap/03-learning-path.md](../00-roadmap/03-learning-path.md) 完成 S1~S7，再系统刷题。
2. 每题先闭卷口述，再对照要点答案；答不上的回到对应章节笔记补。
3. 题目标注难度：★ 基础必答、★★ 常见追问、★★★ 深挖/开放。
4. 带"待确认"的要点答案不能在面试中当作确定事实陈述，需先按给出的确认方式核实。

第 1、2 步构成一个刷题循环：

```text
S1~S7 学完 ──► 闭卷口述 ──► 对照要点答案 ──► 答上了 ──► 下一题
                 ▲               │
                 │               └─ 答不上 ──► 回到对应章节笔记补 ──┐
                 └────── 补完再口述 ────────────────────────────────┘
```

## 全部章节

- 返回总入口：[00-roadmap](../00-roadmap/README.md)
- [01-rdma](../01-rdma/README.md) · [02-scale-up](../02-scale-up/README.md) · [03-ccl-algorithms](../03-ccl-algorithms/README.md) · [04-nccl-internals](../04-nccl-internals/README.md) · [05-framework-integration](../05-framework-integration/README.md) · [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) · [07-perf-troubleshooting](../07-perf-troubleshooting/README.md)
