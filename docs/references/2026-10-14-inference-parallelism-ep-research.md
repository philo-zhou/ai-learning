# 大模型推理并行策略与 EP 通信方案调研（2026-10-14）

- 来源：多源 Web 调研（deep-research 工作流：5 个检索角度、抓取 20 个来源、抽取 97 条可证伪结论，其中 25 条做了 3 票对抗验证），另对 7 个缺口来源做了补充抓取，并在本地 `third_party/` 中核对
- URL：见下方"来源清单"
- 固定版本或 commit：网页无版本号，以访问日期为准；本地核对版本见 `third_party/repos.lock`（nccl 2.32.3、vllm@02653fa2、sglang@d2a43644、DeepEP 2.5.0）
- 访问时间：2026-10-14
- License：各来源归其作者所有；本文只做摘要和短句引用

## 它是什么

本文汇总外部证据：推理中各并行策略对应什么通信，MoE EP 的 dispatch/combine 通信有哪些主流方案，以及这些方案的发展趋势。学习入口 [`notes/00-roadmap/00-start-here.md`](../../notes/00-roadmap/00-start-here.md) 依据本文写成。

## 为什么与本项目有关

目标岗位要求"解决分布式训练/推理场景中的通信问题""优化 All2All 等集合通信性能"，并"洞察集合通信技术趋势"。EP 通信是当前推理通信里变化最快、也最贴近通信库研发的方向。

## 关键证据

置信度沿用工作流的判定：high 表示 3:0 通过且有一手来源；medium 表示仅有单一来源、含推断，或投票 2:1。

| # | 结论 | 置信度 | 来源 |
|---|---|---|---|
| F1 | Dense decode 受显存带宽限制，FSDP 式权重聚合不可行，主要靠 TP 切分。推理激活很小，总字节数低于约"互联带宽 × 1 µs"时集合通信变为延迟受限。注意：1 µs 是 TPU ICI 的假设，GPU/NCCL 的固定开销为几到几十 µs，转折点更高 | medium（阈值一项投票 2:1） | [S1] |
| F2 | PD 分离：prefill 与 decode 常分到不同 GPU 池；prefill 生成的 KV cache 经网络传给 decode，形成独立的点对点大消息传输 | high | [S1][S2][S3] |
| F3 | EP 语义：dispatch 按 router 决策把 token 发往持有对应专家的 GPU；combine 把专家输出按原 token 顺序送回源 GPU。两步合起来在互联上产生大量 all-to-all 流量 | high | [S4][S7] |
| F4 | DeepSeek-V3/R1 线上推理采用 PD 分离：prefill 为 EP32 + DP32 attention（4 节点，每卡 9 个路由专家 + 1 个共享专家）；decode 为 EP144 + DP144（18 节点，每卡 2 个路由专家）。两者都带 32 个冗余专家，用于负载均衡 | high | [S2] |
| F5 | DeepSeek 隐藏 EP 通信的方法：prefill 用 dual micro-batch overlap，一个 micro-batch 的通信藏在另一个的计算后面；decode 各阶段时长不均，把 attention 拆成两步，组成 5 级流水 | high | [S2] |
| F6 | DeepSeek-V3/R1 推理中，dispatch 用 FP8 传输（与训练一致），combine 用 BF16。这是 DeepSeek 的做法，不是 EP 的通用规则 | high | [S2] |
| F7 | SGLang 大规模 EP 部署：12 节点 × 8 H100（96 卡）；attention 用 DP（避免 KV cache 重复），dense FFN 用 DP，MoE FFN 用 EP；prefill EP32、decode EP72。2000 token 输入时每节点 52.3k 输入、22.3k 输出 tok/s，输出吞吐最高为同硬件 vanilla TP 的 5 倍（这是自家 benchmark，对比的基线偏弱） | high | [S3] |
| F8 | vLLM 用 `--all2all-backend` 选择 MoE 通信后端，默认 `allgather_reducescatter`：`deepep_high_throughput` 用连续布局 + grouped GEMM，面向 prefill；`deepep_low_latency` 支持 CUDA Graph、masked 布局，面向 decode。`EP_SIZE = TP × DP` 自动计算；需开 `--enable-expert-parallel` | high | [S5]；本地 `third_party/vllm/vllm/config/parallel.py` |
| F9 | NCCL 2.28 之前所有操作都由 host 发起。2.28 引入 Device API，有三种模式：LSA（经 NVLink/PCIe load/store）、Multimem（NVLink SHARP 硬件多播）、GIN（kernel 直接发 RDMA put/signal，不经 CPU） | high | [S6][S8] |
| F10 | GIN 分三层：host 侧的 communicator 与窗口注册、device 侧的远程内存操作、语义双重的网络插件。后端有两个：GDAKI 用 DOCA GPUNetIO 让 GPU 直接驱动 NIC（类似 IBGDA）；Proxy 经无锁 GPU→CPU 队列在标准 RDMA 上实现同等功能，可覆盖不支持 GDA 的网卡。NVIDIA 已用它集成 DeepEP 作验证 | high（NVIDIA 自述） | [S8]；本地 `third_party/nccl/docs/contrib/GIN/` |
| F11 | NCCL EP（`contrib/nccl_ep`）基于 Device API：RDMA 走 GIN put/signal，NVLink 走 LSA。需要 NCCL 2.29+、CUDA 13+，在 Hopper/Blackwell 上测试过。LL 模式面向小 batch decode，直接点对点 all-to-all，可用 `send_only` + `ncclEpComplete` 做分段重叠；HT 模式面向训练和 prefill，分层传输：节点内 NVLink 聚合，跨节点走 RDMA，使用 TMA 和 warp specialization | high | [S7]；本地 `third_party/nccl/contrib/nccl_ep/` |
| F12 | 第五代 NVLink 每 GPU 双向 1.8 TB/s，单域最多 72 块 Blackwell（NVL72）。规模在 72 卡以内的 EP，all-to-all 可以全走 NVLink（推断）；DeepSeek EP144 这类更大规模仍需 RDMA scale-out | medium（含推断） | [S4] |
| F13 | 字节 MegaScale-Infer 把每层的 attention 与 FFN（专家）拆到不同 GPU，两者可独立扩缩容、各用各的并行策略、部署在异构硬件上。用 ping-pong 流水在两者之间来回传 micro-batch；两者之间用自研 M2N 通信库，省掉多余的 GPU→CPU 拷贝、group 初始化开销和 GPU 同步；每 GPU 吞吐最高为 SOTA 的 1.90 倍 | medium（单一来源） | [S9] |

### 补充抓取（单一来源，未经 3 票验证）

| # | 结论 | 来源 |
|---|---|---|
| G1 | pplx-kernels 提供 dispatch/combine，传输支持 NVLink、IBGDA、IBRC；已声明弃用基于 NVSHMEM 的旧 kernel，转向 pplx-garden | [S10] |
| G2 | pplx-garden 的 fabric-lib 是面向 LLM 系统的 RDMA 点对点 TransferEngine（MLSys'26），提供 P2P MoE all-to-all，支持 AWS EFA | [S11] |
| G3 | UCCL-EP：GPU 发起通信，但不直接控制 NIC。GPU 经共享内存把轻量命令转给 CPU proxy 线程，由后者调用厂商中立的 libibverbs；每 GPU 多通道、每通道一个 QP，可达每秒 5000 万次以上 RDMA 操作。在 AWS EFA + H200 上 dispatch 超过 50 GB/s、combine 约 40 GB/s；在 GH200 上反超 DeepEP。支持 EFA、Broadcom Thor、Google Falcon、AMD Pensando | [S12] |
| G4 | Mooncake 采用以 KV cache 为中心的 PD 分离架构。Transfer Engine 支持 TCP、RDMA、EFA、NVMe-oF、NVLink 等传输，可多网卡聚合并按拓扑选路：4×200 Gbps RoCE 上约 87 GB/s，8×400 Gbps 上约 190 GB/s。vLLM 与 SGLang 都用它传 PD 分离的 KV cache | [S13] |
| G5 | TensorRT-LLM 在 GB200 NVL72 上做 wide-EP：依赖 NVL72 130 TB/s 的聚合带宽，用自定义 NCCL kernel 处理不定长 all-to-all；EPLB 有静态和在线两种模式。在 100 tok/s/user 下，EP32 的每 GPU 输出吞吐最高为 EP8 的 1.8 倍 | [S14] |
| G6 | vLLM 大规模部署：Wide-EP（EP + DP attention）、PD 分离、DBO（双 batch 重叠）、分层 EPLB。all-to-all 后端有 DeepEP HT（DeepSeek 默认）、Perplexity kernel、NCCL AG-RS 回退。在 H200 + IB/CX-7 上每卡约 2.2k tok/s | [S15] |

### 本地源码核对（2026-10-14）

- vLLM 的 `All2AllBackend` 取值有 `naive`、`pplx`、`deepep_high_throughput`、`deepep_low_latency`、`deepep_v2`、`mori_high_throughput`、`mori_low_latency`、`moonep`、`nixl_ep`、`allgather_reducescatter`、`flashinfer_nvlink_two_sided`、`flashinfer_nvlink_one_sided`、`passthrough`。其中 `pplx`、`naive` 会被重映射到 `allgather_reducescatter`，见 `third_party/vllm/vllm/config/parallel.py`。
- SGLang 的 `MoeA2ABackend` 取值有 `deepep`、`deepep_v2`、`mooncake`、`nixl`、`mori`、`ascend_fuseep`、`flashinfer`、`megamoe`、`pplx` 等，见 `third_party/sglang/python/sglang/srt/layers/moe/utils.py`。
- 本地 NCCL 为 2.32.3（`third_party/nccl/makefiles/version.mk`），带 `contrib/nccl_ep/`；DeepEP 为 2.5.0，已移除 V1 与 NVSHMEM，见 `third_party/DeepEP/README.md`。

### 被推翻的结论

- "PD 分离的动机是 DeepEP 的两种模式（normal 不兼容 CUDA Graph，LL 兼容）"：投票 1:2 被否。模式兼容性是相关因素，不能说成 PD 分离的动机。

## 我们借鉴什么

- 以"推理阶段（prefill/decode）× 并行维度 × 通信原语 × 消息量级 × 是否在关键路径"作为组织学习入口的主轴。
- 把 EP 方案按"谁发起通信（host / GPU）× 谁驱动网卡（GPU 直驱 / CPU proxy）× 数据走哪（NVLink / RDMA / 分层）"三个维度分类。

## 我们明确不借鉴什么

- 不把厂商或项目自家 benchmark 的倍数当作结论：F7 的 5 倍、F13 的 1.90 倍、G5 的 1.8 倍，都只作量级参考。
- 不把 DeepSeek 的具体配置（EP 规模、FP8/BF16）当作通用规则。

## 影响的本地决策

- [`notes/00-roadmap/00-start-here.md`](../../notes/00-roadmap/00-start-here.md)：学习入口的结构与 EP 方案对比表。
- [`notes/05-framework-integration/04-deepep.md`](../../notes/05-framework-integration/04-deepep.md)、[`notes/03-ccl-algorithms/04-all2all.md`](../../notes/03-ccl-algorithms/04-all2all.md)：深入阅读的落点。

## 未决问题

- 同一硬件上，NCCL EP（GIN）、DeepEP V1/V2、pplx、UCCL-EP 的延迟和带宽对比如何？尤其在 RoCE、非 NVIDIA 网卡上，GIN Proxy 后端损失多少？目前没有独立第三方测评。
- MegaScale-Infer 的 M2N 与对称 all-to-all 在实现上有什么差异？消息规模和关键路径时延是多少？
- Mooncake / NIXL 的 KV 传输如何分层、流水，并与 decode 重叠？
- 在 NVL72 上 EP 全走 NVLink / Multimem 时，算法和内存语义有哪些新的设计空间？EPLB 冗余专家如何与通信布局配合？
- 本轮没有拿到经验证结论的方向：PP/SP/CP 在推理中的通信量级、DeepEP V1 内部实现、Triton-distributed 的 EP 性能、EPLB 算法细节。

## 来源清单

| ID | 来源 | 类型 |
|---|---|---|
| S1 | [How To Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/) | 一手（Google） |
| S2 | [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | 一手（2025-02） |
| S3 | [SGLang: Large-scale EP on 96 H100](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | 一手 |
| S4 | [NVIDIA: Demystifying AI Inference Deployments for Trillion-Parameter LLMs](https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/) | 一手（厂商） |
| S5 | [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) | 官方文档 |
| S6 | [NVIDIA: Device API and Copy Engine Collectives in NCCL 2.28](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/) | 一手（厂商） |
| S7 | [NCCL contrib/nccl_ep](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep) | 一手（代码） |
| S8 | [GPU-Initiated Networking for NCCL（arXiv 2511.15076）](https://arxiv.org/abs/2511.15076) | 一手（NVIDIA 论文） |
| S9 | [MegaScale-Infer（arXiv 2504.02263）](https://arxiv.org/abs/2504.02263) | 一手（字节论文） |
| S10 | [pplx-kernels](https://github.com/perplexityai/pplx-kernels) | 一手（代码） |
| S11 | [pplx-garden](https://github.com/perplexityai/pplx-garden) | 一手（代码） |
| S12 | [UCCL-EP 博客](https://uccl-project.github.io/posts/uccl-ep/)；[uccl 仓库](https://github.com/uccl-project/uccl) | 项目博客 |
| S13 | [Mooncake](https://github.com/kvcache-ai/Mooncake) | 一手（代码） |
| S14 | [NVIDIA: Wide EP on GB200 NVL72](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/) | 一手（厂商） |
| S15 | [vLLM Large-Scale Serving（2025-12-17）](https://vllm.ai/blog/2025-12-17-large-scale-serving) | 一手 |
| S16 | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | 二手（训练为主） |
| S17 | [LMSYS: GB200 NVL72 Part 1](https://lmsys.org/blog/2025-06-16-gb200-part-1/) | 一手 |

## 新鲜度与更新触发器

- vLLM/SGLang 的后端名称、NCCL 版本要求（2.28/2.29、CUDA 13）、GIN 后端变化很快，面试前按来源清单复核一次。
- DeepSeek 的配置是 2025-02 的公开值，之后可能已调整。
- 出现独立第三方的 EP 库横向测评时，更新"未决问题"一节。
