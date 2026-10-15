# 从这里开始：从大模型推理出发，看懂并行、通信与 EP

> 层次：总览（贯穿上中下三层） · 来源：[调研报告](../../docs/references/2026-10-14-inference-parallelism-ep-research.md)（2026-10-14，外部来源编号记为 S1~S17）、third_party/ai-infra-book@3bdcb4fc（§6、§7.2.3、§9.4）、third_party/nccl@3e46c089、third_party/vllm@02653fa2、third_party/sglang@d2a43644、third_party/DeepEP@93eb6eb2

## 一句话

一次推理请求会经历 prefill 和 decode 两个阶段。为了装下模型、提高吞吐，人们用 TP/PP/DP/CP/EP/PD 分离来切分它，每种切分都会引入一种特定的通信。在 MoE 模型里，EP 的 dispatch/combine（一种动态、不均匀、对延迟敏感的 all-to-all）成了最主要的通信开销。围绕它，通信库正在从"CPU 发起的集合通信"转向"GPU 发起的点对点 RDMA / NVLink 访存"。这份入口把这条主线串起来，并把仓库的每一章挂到主线上。

## 怎么使用这份入口

1. **先通读本页**（约 1 小时），建立全景：每一节末尾的"深入"链接指向仓库内的详细笔记。
2. **按第 7 节的"主线学习顺序"推进**，它是 [10 周学习路线](03-learning-path.md) 的问题驱动版：路线按周排时间，本页按"要回答什么问题"排顺序。
3. **每学完一节**，回到 [JD 拆解](02-jd-breakdown.md) 更新自评，在 `docs/PROGRESS.md` 记录进度。
4. **面试前**做第 9 节的自测，再刷 [面试题库](../08-interview/01-question-bank.md)。
5. 外部事实（部署规模、性能数字、版本要求）的出处和置信度统一放在 [调研报告](../../docs/references/2026-10-14-inference-parallelism-ep-research.md)，本页用 `[F/G 编号]` 引用。

## 在调用链中的位置

本页本身不属于某一层，它是把三层串起来的"问题线"：

```text
推理请求 → 引擎调度（vLLM / SGLang）→ 并行策略决定"谁和谁通信、传什么"   ← 第 1~3 节
         → 通信抽象（GroupCoordinator / All2AllManager）→ 通信库（NCCL / DeepEP / NCCL EP …）  ← 第 4 节
         → transport（NVLink LSA / Multimem · RDMA GIN / IBGDA / CPU proxy）→ 网卡、交换机    ← 第 5 节
```

## 1. 推理的两个阶段：为什么通信特征完全不同

| | Prefill（处理整段输入） | Decode（逐 token 生成） |
|---|---|---|
| 计算形态 | 一次处理成千上万个 token，GEMM 大，计算受限 | 每步每请求 1 个 token，GEMM 小，**显存带宽受限** [F1] |
| 单次通信消息 | 大（MB 级），看**带宽** | 小（KB 到几百 KB），看**延迟** [F1] |
| 关键路径 | 通信可被大计算部分掩盖 | 每层都有通信，且几乎无法被小计算掩盖，直接叠加到每 token 时延（TPOT）上 |
| 典型手段 | 双 micro-batch 重叠、分层（节点内聚合）all-to-all | CUDA Graph、固定容量缓冲、低延迟 kernel、5 级流水 [F5] |

> 估算口径：消息大小 ≈ token 数 × 隐藏维度 × 每元素字节。以书中表 2-A 的 Qwen3-8B（$d=4096$，BF16）为例，decode batch 为 64 时，每次 TP AllReduce 只有 $64 \times 4096 \times 2\,\text{B} = 512\,\text{KiB}$；同一模型 prefill 8k token 时是 64 MiB。差两个数量级，最优算法和协议也就不同（见 [α-β 模型](../03-ccl-algorithms/02-alpha-beta-model.md) 与 NCCL 的 LL/LL128/Simple）。

深入：[并行策略与通信模式](../05-framework-integration/01-parallelism-and-comm-patterns.md) 的"训练 vs 推理"一节；ai-infra-book 第 6、9 章（见 [阅读地图](01-ai-infra-book-map.md)）。

## 2. 并行策略 × 通信：一张总表

| 并行策略 | 切什么 | 推理中的通信 | 量级（单次） | 关键路径？ | 主要落在 |
|---|---|---|---|---|---|
| **TP** 张量并行 | 每层的权重矩阵按行/列切 | 每层 2 次 **AllReduce**（attention 输出、MLP 输出）；开 SP 时拆成 **ReduceScatter + AllGather** | decode 时 KB 级到几百 KB；prefill 时 MB 级 | 是，decode 下尤其严重 | 节点内 NVLink（custom allreduce / NCCL / NVLS） |
| **PP** 流水线并行 | 按层切成 stage | stage 之间 **Send/Recv** 激活 | 每个 micro-batch 一份激活，KB~MB | 只在 stage 边界，可被流水掩盖；代价是流水气泡 | 跨节点 RDMA 也可接受 |
| **DP** 数据并行（推理副本） | 整模型复制 | 推理时副本之间几乎**无通信**（训练才同步梯度） | — | 否 | — |
| **DP-attention** | MoE 模型中 attention 按请求分到各 rank，不再 TP | attention 本身无通信；进入 MoE 前后需要与 EP 的 dispatch/combine 衔接（或 AllGather / ReduceScatter） | 同 EP | 是 | 避免 TP 下 KV cache 被重复存储 [F7] |
| **CP / SP** 上下文/序列并行 | 长序列按 token 切到多卡 | ring attention 中逐跳 **Send/Recv** 交换 KV 块，或 **AllGather** KV | 与序列长度成正比，MB 级 | prefill 长上下文时是 | 主要用于长上下文 prefill；推理中的量级未经验证（见 [调研报告·未决问题](../../docs/references/2026-10-14-inference-parallelism-ep-research.md#未决问题)） |
| **EP** 专家并行 | MoE 层的专家分布到多卡 | 每个 MoE 层 2 次 **All2All**：dispatch + combine [F3] | 见第 3 节公式；动态、不均匀 | **是，MoE 推理最主要的通信开销** | 节点内 NVLink + 跨节点 RDMA |
| **PD 分离** | prefill 和 decode 放到不同 GPU 池 | prefill → decode 的 **KV cache 传输**（点对点、大消息）[F2] | 每请求 MB~GB 级 | 影响首 token 之后的衔接，可与计算流水重叠 | RDMA 传输引擎（Mooncake、NIXL）[G4] |

组合方式随模型而定。Dense 模型常用"节点内 TP + 节点间 PP/DP"。大 MoE 模型（DeepSeek-V3 类）的主流是 **DP-attention + EP + PD 分离**：

- DeepSeek：prefill EP32、decode EP144 [F4]
- SGLang：96 卡 H100 上 prefill EP32、decode EP72 [F7]
- vLLM：`EP_SIZE = TP × DP` [F8]

深入：[并行策略与通信模式](../05-framework-integration/01-parallelism-and-comm-patterns.md)、[集合原语语义](../03-ccl-algorithms/01-primitives.md)。

## 3. EP 通信深入：dispatch / combine 为什么难

### 3.1 语义

```text
     router(top-k)            dispatch (All2All-v)          expert GEMM          combine (All2All-v)
token ──────────> 选中 k 个专家 ──> 把 token 发到专家所在 GPU ──> 各 GPU 算本地专家 ──> 结果按原顺序送回并加权求和
```

- **dispatch**：每个 token 按路由结果发往持有其 top-k 专家的 GPU。同一 token 去往同一 GPU 上的多个专家时，只需传一次 [F3]。
- **combine**：专家输出按原 token 顺序送回源 GPU，再按门控权重相加；可以在接收端合并，也可以在发送端预先合并。
- 两步都是 **All2All-v**：每对 rank 之间的数据量由当前 batch 的路由决定，不是静态的。

### 3.2 数据量估算

ai-infra-book §7.2.3 给出了单层的 dispatch + combine 跨边界传输量：

$$V \approx 2\,n\,k\,f\,d$$

其中 $n$ 为 token 数，$k$ 为每个 token 选择的专家数，$f$ 为跨边界比例，$d$ 为每份激活的字节数。书中例子：$n=1024,k=8$，每份 8 KiB，一半跨服务器，dispatch 与 combine 各 32 MiB。降低 $f$ 靠专家放置，降低 $d$ 靠低精度（DeepSeek 的 dispatch 用 FP8、combine 用 BF16 [F6]）。

### 3.3 与普通 AllReduce 的五点不同（面试高频）

1. **动态、不均匀**：每步的路由都不同，热点专家会让某个接收端成为瓶颈（书 §9.4.1：整层耗时由最忙的那张卡决定）。缓解靠 EPLB 冗余专家（DeepSeek 用 32 个 [F4]，vLLM、SGLang 也有实现：`third_party/vllm/vllm/distributed/eplb/`、`third_party/sglang/python/sglang/srt/eplb/`）。
2. **要先交换计数**：接收端事先不知道会收到多少数据。高吞吐模式先交换 layout（计数），再传数据；低延迟模式按最大容量预分配固定缓冲，换取 CUDA Graph 兼容 [F8]。
3. **两种工况两种 kernel**：
   - HT/normal（prefill）：连续布局，分层传输（节点内 NVLink 聚合，再跨节点 RDMA），追求带宽。
   - LL（decode）：直接点对点，masked 布局，追求延迟 [F8][F11]。
4. **通信与数据重排融合**：dispatch 要把 token 按专家重排给 grouped GEMM，combine 要做加权求和。这些逻辑和通信融合在同一个 kernel 里，通用 AllReduce 没有这一层。
5. **SM 争用**：通信 kernel 会占用 SM。占得越少，留给计算的越多，这直接影响通算重叠的效果。

### 3.4 怎么把 EP 通信藏起来

- **Prefill**：双 micro-batch 重叠，一个 micro-batch 的通信藏在另一个的计算后面 [F5]。实现：SGLang `third_party/sglang/python/sglang/srt/batch_overlap/two_batch_overlap.py`，vLLM DBO `third_party/vllm/vllm/v1/worker/ubatching.py` [G6]。
- **Decode**：各阶段时长不均，把 attention 拆成两步，组成 5 级流水 [F5]。
- **更进一步**：attention 与 FFN 分离部署（MegaScale-Infer），用 ping-pong 流水在两组 GPU 之间来回传 micro-batch，中间走 M2N 通信 [F13]。
- **kernel 级**：LL 模式的"只发送 + 稍后完成"接口（NCCL EP 的 `send_only` + `ncclEpComplete` [F11]；DeepEP V1 的 recv hook），把等待推迟到计算之后。

深入：[DeepEP](../05-framework-integration/04-deepep.md)、[All2All 算法与 MoE 特殊性](../03-ccl-algorithms/04-all2all.md)、[通算重叠分类](../06-comm-compute-fusion/01-overlap-taxonomy.md)；动手：`labs/ccl_sim` 的 `pairwise_all2allv` 可以模拟热点专家。

## 4. 主流 EP 通信方案全景

用三个问题给方案分类：**谁发起通信**（host / GPU）、**谁驱动网卡**（GPU 直接写网卡门铃 / CPU proxy 代发 verbs）、**数据走哪**（NVLink / RDMA / 分层）。

| 方案 | 发起方 | 跨节点如何驱动网卡 | 节点内 | 定位与要点 | 证据 |
|---|---|---|---|---|---|
| NCCL `ncclAlltoAll` / AG+RS 回退 | host | NCCL proxy 线程 | NVLink P2P | 通用、易用；动态不均匀 + 小消息时开销大。vLLM 默认 `allgather_reducescatter` | [F8]；`third_party/vllm/vllm/config/parallel.py` |
| **DeepEP V1** | GPU | NVSHMEM + IBGDA（GPU 直接驱动网卡） | NVLink | 事实标准：normal 与 LL 两种 kernel，奠定 dispatch/combine 接口；只能跑在 NVIDIA GPU + 网卡组合上 | [S2]；V1 源码 `third_party/DeepEP-hybrid-ep/csrc/kernels/internode_ll.cu`，详见 [09/02](../09-ep-kernels/02-deepep-low-latency.md) |
| **Hybrid-EP**（NVIDIA，DeepEP `hybrid-ep` 分支） | GPU | DOCA GPUNetIO（或 NIXL） | NVLink（TMA） | 训练/prefill 用：路由 AllGather 预计算落点，跨节点只发同号 GPU，TMA warp 专业化流水，4~16 个 SM | `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/`；[09/03](../09-ep-kernels/03-hybrid-ep.md) |
| **DeepEP V2.x** | GPU | NCCL GIN（GDAKI，或 Proxy 后端） | NCCL LSA | 移除 NVSHMEM，改建在 NCCL device API 上 | [F10]；`third_party/DeepEP/README.md`、`third_party/DeepEP/csrc/kernels/comm/context.cpp` |
| **NCCL EP** | GPU | GIN put/signal | LSA | NCCL 官方的 MoE 扩展，有 LL 与 HT 两种模式（对标 DeepEP）；需 NCCL 2.29+、CUDA 13+ | [F11]；`third_party/nccl/contrib/nccl_ep/device/ll_ep.cuh`、`hybrid_ep.cuh` |
| **pplx-kernels → pplx-garden** | GPU | 旧版：NVSHMEM（IBGDA/IBRC）；新版：fabric-lib RDMA P2P | NVLink | Perplexity 方案，旧 kernel 已弃用；新版支持 AWS EFA | [G1][G2] |
| **UCCL-EP** | GPU 发命令 | **CPU proxy 线程调用 libibverbs** | NVLink | 厂商中立：可跑在 EFA、Broadcom、AMD Pensando 等网卡上；每通道一个 QP；在 GH200 上反超 DeepEP | [G3] |
| **MegaScale-Infer M2N**（字节） | — | 自研 M2N 库 | — | attention 与 FFN 分离，M 个 attention 节点对 N 个专家节点的非对称通信；省掉 GPU→CPU 拷贝与同步 | [F13] |
| TensorRT-LLM wide-EP | GPU | — | NVL72 NVLink | 72 卡 scale-up 域内的 all-to-all，用自定义 NCCL kernel；EP32 的每 GPU 吞吐为 EP8 的 1.8 倍（厂商数据） | [G5] |
| 框架中其他后端 | — | — | — | vLLM：`mori_*`（AMD）、`moonep`、`nixl_ep`、`flashinfer_nvlink_one/two_sided`。SGLang：`mooncake`、`nixl`、`mori`、`ascend_fuseep`、`megamoe`。**待调研** | 本地 `third_party/vllm/vllm/config/parallel.py`、`third_party/sglang/python/sglang/srt/layers/moe/utils.py` |
| Triton-distributed | GPU | NVSHMEM 等多后端 | — | 用 Triton 写通算融合算子，含 EP all-to-all 示例；与 JD 中"用 Triton 开发融合算子"直接对应 | [Triton-distributed 导读](../06-comm-compute-fusion/02-triton-distributed.md) |

**KV cache 传输**（PD 分离）是另一类通信：大块点对点、可流水。

- Mooncake Transfer Engine：支持 RDMA、EFA、NVLink、TCP 等，多网卡聚合，8×400 Gbps 上约 190 GB/s [G4]。
- NIXL：vLLM 的 KV connector 使用 [G6]。

两者都被 vLLM 和 SGLang 集成。

> 设计取舍：
>
> - **GPU 直接驱动网卡**（IBGDA/GDAKI）：延迟最低、不占 CPU，但绑定特定厂商的 GPU 与网卡。
> - **CPU proxy**（UCCL-EP、GIN Proxy、NCCL 传统路径）：可移植、可观测、可做拥塞控制，代价是多一跳 GPU→CPU 通知。
>
> 字节这类自研网卡与协议的团队，更关心后者带来的可控性。这是面试时可以展开的观点。

## 5. 发展趋势

1. **通信从 host 发起转向 GPU 发起**：NCCL 2.28 加入 Device API（LSA / Multimem / GIN）[F9]，DeepEP V2 改建在 GIN 上 [F10]，NCCL EP 进入官方 contrib [F11]。通用通信库正在"收编"EP 专用库。
2. **GPU 直驱与 CPU proxy 并存**：GIN 同时提供 GDAKI 与 Proxy 两个后端 [F10]；UCCL-EP 用 CPU proxy 换取跨厂商支持 [G3]。非 NVIDIA 网卡（EFA、RoCE 自研网卡）必须走后者或自研 GDA。
3. **scale-up 域变大**：NVL72 的 1.8 TB/s/GPU 让 72 卡内的 EP 完全走 NVLink [F12][G5]，算法从"先网络后 NVLink"转向基于内存语义的 load/store 与 Multimem；超出 scale-up 域（EP144）仍要 RDMA。
4. **通信与计算融合**：从 stream 级重叠（DBO/TBO）[F5][G6]，到 kernel 内融合（dispatch 与 GEMM 流水、copy engine 卸载），再到架构级拆分（attention/FFN 分离 [F13]）。
5. **低精度与压缩通信**：FP8 dispatch [F6]，压缩的是公式里的 $d$。
6. **负载均衡与通信布局协同**：EPLB 冗余专家、在线重排 [F4][G5]，降低的是热点与公式里的 $f$。
7. **非对称与异构通信**：M2N [F13]、PD 分离的 KV 传输 [G4]，CCL 不再只处理对称的集合通信。

> 注意：上面各项多数依据厂商或项目方的自述，目前缺少独立的横向测评（[调研报告·未决问题](../../docs/references/2026-10-14-inference-parallelism-ep-research.md#未决问题)）。面试时以"趋势判断"的口吻讲，不要把倍数说成定论。

## 6. 仓库框架：这条主线怎样落到各章

| 主线问题 | 落到哪一章 | 关键笔记 |
|---|---|---|
| 推理里哪些地方要通信、传多少？ | 上层 [05-framework-integration](../05-framework-integration/README.md) | [01-parallelism](../05-framework-integration/01-parallelism-and-comm-patterns.md)、[02-vllm](../05-framework-integration/02-vllm-communication.md)、[03-sglang](../05-framework-integration/03-sglang-communication.md) |
| 这些通信用什么算法最快？ | 中层 [03-ccl-algorithms](../03-ccl-algorithms/README.md) | [α-β 模型](../03-ccl-algorithms/02-alpha-beta-model.md)、[All2All](../03-ccl-algorithms/04-all2all.md)、`labs/ccl_sim` |
| NCCL 怎么把算法落到硬件？ | 中层 [04-nccl-internals](../04-nccl-internals/README.md) | channel / 协议 / proxy / net plugin |
| EP 专用库为什么存在、往哪走？ | 上层与跨层 [05](../05-framework-integration/04-deepep.md)、[06](../06-comm-compute-fusion/03-nvshmem-and-device-initiated.md) | DeepEP、NCCL EP、GIN |
| EP 的 dispatch/combine 在 kernel 里怎么做？ | 中层 [09-ep-kernels](../09-ep-kernels/README.md) | DeepEP LL、Hybrid-EP、谱系与选型 |
| 通信怎么和计算融合？ | 跨层 [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 重叠分类、Triton-distributed、设计清单 |
| 节点内靠什么？ | 下层 [02-scale-up](../02-scale-up/README.md) | NVLink/NVSwitch、P2P/SHM/NVLS |
| 节点间靠什么？ | 下层 [01-rdma](../01-rdma/README.md) | Verbs/QP、RoCEv2 拥塞控制、GPUDirect、NCCL IB |
| 慢了、卡死了怎么查？ | 实践 [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | 环境变量、方法论、排障清单 |
| 面试怎么讲？ | [08-interview](../08-interview/README.md) | 题库 |

## 7. 主线学习顺序（建议）

1. **本页 → [并行与通信模式](../05-framework-integration/01-parallelism-and-comm-patterns.md)**：能说清第 2 节总表的每一行。
2. **[集合原语](../03-ccl-algorithms/01-primitives.md) → [α-β 模型](../03-ccl-algorithms/02-alpha-beta-model.md)**：跑 `python3 -m labs.ccl_sim`，能解释 decode 小消息为何选择低延迟算法。
3. **[All2All](../03-ccl-algorithms/04-all2all.md) → [DeepEP](../05-framework-integration/04-deepep.md) → [EP kernel 专章](../09-ep-kernels/README.md)**：能写出第 3.2 节的公式，讲清第 3.3 节的五点不同；再讲清 LL 的"固定槽位 + 同 QP 计数"与 Hybrid-EP 的"路由预计算 + 分层"。
4. **[vLLM 通信层](../05-framework-integration/02-vllm-communication.md)**：对照 `third_party/vllm/vllm/distributed/device_communicators/all2all.py`，看后端怎样被选中。
5. **[device-initiated 通信](../06-comm-compute-fusion/03-nvshmem-and-device-initiated.md)**：读 `third_party/nccl/contrib/nccl_ep/README.md`，理解 GIN 的 GDAKI 与 Proxy 两个后端。
6. **下沉到 [RDMA](../01-rdma/README.md) 与 [scale-up](../02-scale-up/README.md)**：理解 GDAKI 为什么要求 GPU 能直接写网卡门铃，CPU proxy 又多付出了什么。
7. **[NCCL 内部](../04-nccl-internals/README.md)**：理解通用路径（channel、proxy）与 EP 专用路径的差异。
8. **[通算融合](../06-comm-compute-fusion/README.md) → [性能排查](../07-perf-troubleshooting/README.md) → [面试题库](../08-interview/01-question-bank.md)**。

## 8. 与目标岗位的对应

| JD 条目 | 本页对应 |
|---|---|
| 集合通信库设计、开发与优化，支持多卡/多机 | 第 2、4 节 |
| 基于 RDMA 和 scale-up 互联研发高速通信库 | 第 4 节的取舍、第 5 节趋势 2 与 3 |
| 通信与计算协同，通信算子与计算算子融合 | 第 3.4 节、第 5 节趋势 4 |
| 优化 AllReduce/All2All 的单机内/多机间性能 | 第 1、3 节 |
| 解决分布式推理通信问题 | 第 3.3、3.4 节 |
| 利用 Triton 开发通信计算融合算子 | 第 4 节 Triton-distributed |
| 洞察趋势，推动自研通信库架构演进 | 第 5 节 |

## 面试自测

1. **问**：decode 阶段为什么 TP AllReduce 对延迟敏感？该选什么算法和协议？
   **答**：decode 受显存带宽限制，每层的 AllReduce 消息只有 KB 到几百 KB，落在 α 主导的区间 [F1]，而且每层都要做、无法被计算掩盖。应选步数少的算法（tree / 单步 one-shot、custom allreduce）和低延迟协议（LL/LL128），并用 CUDA Graph 消除 launch 开销。
2. **问**：MoE 推理为什么常用 DP-attention + EP，而不是全 TP？
   **答**：TP 下每个 rank 都要存完整 KV cache 头或重复 latent，浪费显存；DP-attention 让每个 rank 只存自己请求的 KV。专家参数大但稀疏，适合按 EP 分布 [F7]。代价是 MoE 前后要做 dispatch/combine。
3. **问**：写出单层 EP 的通信量，说说怎么降低。
   **答**：$V \approx 2nkfd$。降 $f$：专家放置、复制热门专家、优先节点内；降 $d$：FP8 dispatch [F6]；同一 GPU 上的多个目标专家只传一次。
4. **问**：EP 的 All2All 和 NCCL 的 `ncclAlltoAll` 差在哪里？
   **答**：数据量动态且不均、需要先交换计数、要和 token 重排及加权求和融合、prefill 和 decode 需要两种 kernel、对 SM 占用敏感、需兼容 CUDA Graph（第 3.3 节）。
5. **问**：DeepEP V1 用 IBGDA，UCCL-EP 用 CPU proxy，各自的优缺点？如果是你设计自研通信库，选哪个？
   **答**：IBGDA 延迟低、不占 CPU，但绑定 NVIDIA GPU 与网卡；proxy 可移植、可观测、可做拥塞控制与多路径，代价是一跳 GPU→CPU 通知，需要多通道多 QP 来补吞吐 [G3]。自研网卡场景下可以两者都留：NCCL GIN 就同时有 GDAKI 与 Proxy 两个后端 [F10]。
6. **问**：NVL72 对 EP 通信意味着什么？
   **答**：72 卡内的 all-to-all 全走 NVLink，每 GPU 1.8 TB/s [F12]，可以用 load/store、Multimem 等内存语义；跨域的更大 EP 仍要 RDMA，形成"scale-up 内 + scale-out 间"的两级设计。
7. **问**：PD 分离引入了什么新的通信？怎么做？
   **答**：prefill→decode 的 KV cache 点对点大块传输 [F2]，用 Mooncake、NIXL 等传输引擎，靠多网卡聚合与按层流水重叠 [G4]。
8. **问**：字节的 MegaScale-Infer 和 DeepSeek 的方案在通信上有什么本质不同？
   **答**：DeepSeek 是对称 EP，每个 rank 既跑 attention 又跑专家，做对称 all-to-all；MegaScale-Infer 把 attention 与 FFN 分到不同节点，做 M→N 的非对称通信加 ping-pong 流水 [F13]。

## 待深入

- 待确认：CP/SP 在推理中的通信量级，以及 ring attention 在 vLLM/SGLang 的实现位置；确认方式：在 `third_party/vllm`、`third_party/sglang` 中 grep `context_parallel` / `ring_attn`，并补充外部来源。
- 待确认：vLLM 的 `mori_*`、`moonep`、`nixl_ep`、`flashinfer_nvlink_*` 后端，以及 SGLang 的 `mooncake`、`megamoe` 后端各自的实现与适用场景；确认方式：读 `third_party/vllm/vllm/distributed/device_communicators/all2all.py` 及 SGLang `layers/moe/` 下对应的 dispatcher。
- 待确认：MegaScale-Infer M2N 的实现细节；确认方式：阅读 arXiv 2504.02263 正文。
- 待确认：各 EP 库在同一硬件上的横向性能对比；确认方式：等待第三方测评，或在 GPU 环境中用各库自带的 benchmark（如 `third_party/nccl/contrib/nccl_ep/ep_bench.cu`）实测。
