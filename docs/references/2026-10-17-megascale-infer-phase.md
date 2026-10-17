# MegaScale-Infer 的 AF 分离作用于哪个阶段

- 访问日期：2026-10-17
- 问题：attention/FFN 分离（AF 分离）用在 prefill 还是 decode？
- 来源：[S1] MegaScale-Infer，arXiv 2504.02263，HTML 全文 https://arxiv.org/html/2504.02263 （一手，字节论文）

## 结论

| 编号 | 结论 | 置信度 | 来源 |
|---|---|---|---|
| M1 | 论文的 AF 分离（disaggregated expert parallelism）只针对 **decode** 阶段。引言原文："we focus on the decoding phase, aiming to address its inefficiency." | high（原文直述） | [S1] |
| M2 | 动机：decode 阶段 attention 要读全部历史 KV cache，GPU 利用率低；MoE 的稀疏激活又让每个 expert 分到的 token 很少，FFN 从计算密集变成访存密集。拆开后，多个 attention 节点的 token 汇聚到 expert 节点，增大每个 expert 的 batch | high（原文直述） | [S1] |
| M3 | prefill 与 decode 分到不同集群（PD 分离），prefill 不走 AF 分离机制；论文评估中提到异构部署"does not enhance resource utilization during the prefill phase" | medium（经摘要工具转述，未逐段核对原文位置） | [S1] |

## 未决问题

- M2N 库的建连方式、ping-pong 的 micro-batch 数量如何确定：需逐节阅读论文 §4、§5。
