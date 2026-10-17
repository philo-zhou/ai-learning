# ai-infra-book 通信相关阅读地图

> 层次：总览 · 来源：third_party/ai-infra-book@3bdcb4fc（正文 `third_party/ai-infra-book/manuscripts/`，案例 `third_party/ai-infra-book/case-studies/`）

## 一句话

ai-infra-book 不是一本"NCCL 教程"，它用 α-β 时间模型 + 物理路径计数的方法贯穿全书；对 CCL 岗位最有价值的是第 6 章（集合通信与超节点）、第 7 章（数据中心网络 / RDMA / 拥塞控制），其次是第 4、5、9、10 章里与通信重叠、加速器发起通信、MoE dispatch/combine 相关的小节。

## 使用方式

- 章节路径都以 `third_party/ai-infra-book/manuscripts/` 为前缀，节号对应文件中的 `##`/`###` 标题（已用 `grep -n '^#'` 核对）。
- "JD 条目"编号见 [02-jd-breakdown.md](02-jd-breakdown.md)：
  - R1 CCL 设计开发优化、多卡/多机通信
  - R2 基于 RDMA 与 scale-up 互联研发通信库
  - R3 通信与计算融合算子（Triton）
  - R4 优化 AllReduce/All2All 单机内/多机间性能
  - R5 解决分布式训练/推理通信问题
  - Q1 分布式系统 / AI 集群架构；Q2 网络通信原理、RDMA；Q3 Ring/Tree/Halving-Doubling；Q4 NCCL 实现原理；Q5 性能调优与故障排查
- 书里给出的数字（如 H100 NVLink 每轮 0.822 μs、RoCE RC 启动间隔 18.6 ns）是书中教学算例的取值，引用时要说明"书中取值"，不是本仓库实测。

## 建议阅读顺序（按优先级排列）

三轮阅读与书中章节、notes 目录的对应关系（只列每轮的主要章节；书没覆盖的 NCCL 实现细节在最后一节从源码补）：

```text
             书中章节（manuscripts/）                       主要落到的 notes 目录
第一轮 ─┬─ 04 §4.5.3 点对点 T = α + V/R ──────────────────► 03-ccl-algorithms
 (必读) ├─ 06 §6.2 / §6.4 并行方式、环与树、库与资源竞争 ─► 03 / 04 / 05 / 06 / 07
        └─ 07 §7.1~§7.6 跨服务器流量、RDMA、拥塞 ─────────► 01-rdma / 03 / 07
第二轮 ─┬─ 06 §6.5 拓扑与超节点 ──────────────────────────► 02-scale-up / 03
(细节)  ├─ 07 §7.1.3 / §7.2.3 / §7.4 QP 状态、EP 负载 ────► 01-rdma / 05
        └─ 04 §4.4.4 / §4.5.2 fence 交接、CPU↔GPU ────────► 06-comm-compute-fusion
第三轮 ─┬─ 05 §5.1.3 / §5.5 stream、CUDA Graph ───────────► 04 / 06
(场景)  ├─ 09 §9.2.2 / §9.4 KV 传输、大 EP 流水 ──────────► 05 / 06
        └─ 10 / 11 训练系统、调度 ────────────────────────► 05 / 07
书未覆盖 ── NCCL 初始化 / channel / 协议 / proxy ─────────► 04-nccl-internals（读 third_party/nccl）
```

### 第一轮：必读主线（约 2 周）

| 顺序 | 章节路径 | 讲了什么 | 对应 JD | 对应 notes 目录 |
|---|---|---|---|---|
| 1 | `04-加速器架构.md` §4.5.3 加速器之间的数据交换 | 点对点时间 $T=\alpha+V/R$；8 KiB 与 2 MiB 两个算例说明"小消息看 α、大消息看 β" | Q3、R4 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| 2 | `06-超节点.md` §6.2 六种并行方式（§6.2.1~6.2.6） | DP/TP/SP/CP/PP/EP 各切哪个维度，各自产生什么通信 | Q1、R5 | [05-framework-integration](../05-framework-integration/README.md) |
| 3 | `06-超节点.md` §6.4.1 根据模型切分方式确定通信需求 | AllReduce = ReduceScatter + AllGather；四种集合操作的"开始/结束时各卡持有什么"表；MoE combine 不等于 AllReduce | Q3、R4 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| 4 | `06-超节点.md` §6.4.2 环、树与通信轮次 | 环形 AllReduce 逐块跟踪；式 (6-9) $T_{ring}=2(n-1)\alpha+2(n-1)M/(nB)$；式 (6-10) 二项树；八卡环/树交点约 680 KiB；"带宽翻倍不如 α 减半"的 TP decode 算例 | Q3、R4 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| 5 | `06-超节点.md` §6.4.3 通信库与计算的资源竞争 | communicator/rank/stream 语义；NCCL 成组 Send/Recv 表达不等长 All2All；**algbw vs busbw** 定义（16 rank 64 MiB 算例）；通信与 GEMM 争用 SM/HBM；AutoCCL 在线调优 | Q4、Q5、R1 | [04-nccl-internals](../04-nccl-internals/README.md)、[07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| 6 | `06-超节点.md` §6.4.4 通信融合、加速器发起与专用卸载 | AllReduce+残差+RMSNorm 融合的顺序约束；CPU 代理 / GPU SM 发起（GPUDirect Async、DeepEP 低延迟 kernel）/ 网卡处理器发起三种控制路径与 PCIe 穿越次数；Copy Engine 与 proxy 线程 | R3、R2、Q4 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md)、[04-nccl-internals](../04-nccl-internals/README.md) |
| 7 | `07-数据中心网络.md` §7.1.2 流量与资源模型、§7.2.1 数据并行的跨服务器流量 | 两台 HGX H100 共 16 rank 的贯穿算例；连续环 / 交错环 / **分层归约** / **在网归约（SHARP）** 四方案的跨服务器字节数对比（例题 7.2） | Q3、R4、Q1 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| 8 | `07-数据中心网络.md` §7.2.4 多网卡、§7.2.5 多轨拓扑 | 多 NIC 只有在有数据分配时才加速；rail-optimized 拓扑（第 i 张网卡接第 i 台叶交换机）；FuseLink | Q1、R4 | [01-rdma](../01-rdma/README.md)、[07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| 9 | `07-数据中心网络.md` §7.3.1~7.3.4 远程访问的路径与并发 | 读写/消息/RPC 语义；**GPUDirect RDMA** 与加速器发起；Load/Store vs Read/Write；在途槽位与启动间隔 δ 限制有效带宽（例题 7.3，RoCE RC δ≈18.6 ns 为书中取值） | Q2、R2 | [01-rdma](../01-rdma/README.md) |
| 10 | `07-数据中心网络.md` §7.5.1~7.5.4 共享网络中的拥塞与可靠性 | incast 流体模型；**PFC** 一跳反馈 vs **ECN/DCQCN** 端到端反馈的分工与 headroom 计算；多路径与重传；死锁 | Q2、R5 | [01-rdma](../01-rdma/README.md) |
| 11 | `07-数据中心网络.md` §7.6.1~7.6.3 | 关键路径从"数据就绪"开始；MegaScale 发现的发起时间偏差；分层网络耗时计算（例题 7.6）；训练（大梯度、带宽主导）vs 推理 decode（小消息、启动主导） | Q5、R5 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |

### 第二轮：scale-up 与 RDMA 细节（约 1 周）

| 顺序 | 章节路径 | 讲了什么 | 对应 JD | 对应 notes 目录 |
|---|---|---|---|---|
| 12 | `06-超节点.md` §6.5.1 直连、交换与分层拓扑 | 逻辑算法 vs 物理拓扑；TP16 跨两台 HGX 的算例；递归（距离翻倍）算法与 Swing 在物理环上的链路负载对比 | Q3、R2 | [02-scale-up](../02-scale-up/README.md)、[03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| 13 | `06-超节点.md` §6.5.2~6.5.5 | 互联介质与物理规模；NVIDIA 从 DGX-1 混合立方体到 NVSwitch 再到 GB200 NVL72；TPU 拓扑；Unified Bus 与 CloudMatrix | R2、Q1 | [02-scale-up](../02-scale-up/README.md) |
| 14 | `07-数据中心网络.md` §7.1.3 统一互联、§7.4.1~7.4.4 | 缓冲区复用与完成通知；事务层/传输层分离（Jetty vs RoCE RC 每连接 512 B QP 上下文，书中取值）；QP 状态随连接数平方增长导致网卡缓存溢出；操作顺序与背压 | Q2、R2 | [01-rdma](../01-rdma/README.md) |
| 15 | `07-数据中心网络.md` §7.2.3 专家并行的通信负载 | MoE dispatch 的小消息数量；网卡报文速率 $1/\delta$ 成为瓶颈的条件 | R4、R5 | [05-framework-integration](../05-framework-integration/README.md) |
| 16 | `04-加速器架构.md` §4.4.4 SM 之间的交接：fence 与同步层级、§4.5.2 CPU 与加速器之间的数据访问 | fence/flag 交接（通算融合 kernel 的基础）；CPU↔GPU 路径 | R3 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |

### 第三轮：通算重叠与上层场景（约 1 周）

| 顺序 | 章节路径 | 讲了什么 | 对应 JD | 对应 notes 目录 |
|---|---|---|---|---|
| 17 | `05-算子与运行时.md` §5.1.3 stream、event 与完成顺序 | 通信 kernel 与计算 kernel 的跨流依赖；集合通信调用返回 ≠ 完成 | Q4、R5 | [04-nccl-internals](../04-nccl-internals/README.md) |
| 18 | `05-算子与运行时.md` §5.5.2 CUDA Graph、§5.5.4 Persistent Kernel | 图捕获对通信的影响；按数据块调度、完成标志（通算融合的调度原型） | R3 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |
| 19 | `10-训练系统.md` §10.2.1 数据并行与 ZeRO／FSDP 状态分片 | ZeRO/FSDP 把 AllReduce 换成 ReduceScatter + AllGather | Q1、R5 | [05-framework-integration](../05-framework-integration/README.md) |
| 20 | `10-训练系统.md` §10.3.3 反向依赖、梯度分桶与通信重叠 | DDP bucket 大小权衡；"通信等待时间"定义；链路被专家交换占用时的重叠失败 | R5、Q5 | [05-framework-integration](../05-framework-integration/README.md)、[07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| 21 | `10-训练系统.md` §10.3.4 MoE 与长上下文的调度变化、§10.4.5 掉队者与慢节点 | MoE/CP 下的通信调度；straggler 期望步时间（N 个样本最大值） | R5、Q5 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| 22 | `09-分布式推理.md` §9.2.2 KV 传输与两端内存占用 | PD 分离的 KV 传输量与时间 | R5 | [05-framework-integration](../05-framework-integration/README.md) |
| 23 | `09-分布式推理.md` §9.4.1~9.4.3 | 大 EP 计算/通信 skew；专家副本与放置；dispatch / 计算 / combine 的 micro-batch 流水与 DMA-GEMM 争用 HBM | R4、R3 | [05-framework-integration](../05-framework-integration/README.md)、[06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |
| 24 | `10-训练系统.md` §10.5.2 各阶段的资源分配与权重同步 | RL 训推之间的权重同步（广播/点对点） | R5 | [05-framework-integration](../05-framework-integration/README.md) |
| 25 | `11-资源调度与运行环境.md` §11.3.1 异构资源、成组分配与放置约束 | 拓扑感知的成组调度 | Q1 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |

### 选读：书的配套案例（`third_party/ai-infra-book/case-studies/`）

| 文件 | 讲了什么 | 对应 JD | 对应 notes 目录 |
|---|---|---|---|
| `collective-paths-and-diagnosis.md` | 链路上界 vs 实际分配；框架如何选归约实现；"等通信可能是在等更晚到达的计算" | Q5、R5 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| `communication-tuning.md` | AutoCCL 的判断与适用条件；2025–2026 库接口变化 | R1、Q5 | [04-nccl-internals](../04-nccl-internals/README.md) |
| `network-planning-and-collectives.md` | Swing、CASSINI：算法字节 → 物理链路字节 → 作业错峰 | Q3、Q1 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| `remote-ordering-and-completion.md` | 远端访问的顺序与完成语义；对照 NVSHMEM 实现 | Q2、R3 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |
| `pcie-staging-and-numa.md` | PCIe 中转与 NUMA 放置；TCCL | R2、Q5 | [02-scale-up](../02-scale-up/README.md) |
| `expert-dispatch-and-resizing.md` | 专家分派、重排与扩缩容 | R4 | [05-framework-integration](../05-framework-integration/README.md) |
| `mesh-shape-and-slicing.md` | 通信组形状与通信分块粒度（MeshSlice） | R3、R4 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |
| `programmable-nic.md` | 可编程网卡与数据面卸载 | R2 | [01-rdma](../01-rdma/README.md) |

## 书没有覆盖、需要从源码补的部分

书只在概念层提到 NCCL，下列内容要到 `third_party/` 源码里读：

| 主题 | 去哪里读 | 对应 notes 目录 |
|---|---|---|
| NCCL 初始化、拓扑搜索、channel、协议（LL/LL128/Simple）、transport、proxy | `third_party/nccl/src/init.cc`、`src/graph/`、`src/device/prims_*.h`、`src/transport/`、`src/proxy.cc`、`src/tuning/` | [04-nccl-internals](../04-nccl-internals/README.md) |
| Verbs 编程（QP 建链、MR、post_send、poll_cq） | `third_party/rdma-core/libibverbs/examples/rc_pingpong.c` | [01-rdma](../01-rdma/README.md) |
| 网卡层带宽/时延基线 | `third_party/perftest/src/write_bw.c`、`read_lat.c` 等 | [01-rdma](../01-rdma/README.md) |
| 集合通信基准与 busbw 换算 | `third_party/nccl-tests/src/all_reduce.cu` 中 `AllReduceGetBw` | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| Triton 通算融合 | `third_party/Triton-distributed/tutorials/`、`python/triton_dist/kernels/nvidia/` | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |
| MoE EP 通信 | `third_party/DeepEP/deep_ep/buffers/ep.py`、`csrc/kernels/ep/` | [05-framework-integration](../05-framework-integration/README.md) |
| 推理框架通信层 | `third_party/vllm/vllm/distributed/device_communicators/`、`third_party/sglang/python/sglang/srt/distributed/device_communicators/` | [05-framework-integration](../05-framework-integration/README.md) |

- Halving-Doubling（Rabenseifner）：书中只在 §6.5.1 以"递归算法每轮把对端距离翻倍"的形式出现，没有完整推导。本地源码核实：算法编号只有 Tree/Ring/CollNetDirect/CollNetChain/NVLS/NVLSTree/PAT 七种（`third_party/nccl/src/include/plugin/nccl_tuner.h` 的 `NCCL_ALGO_*`，`src/init.cc` 的 `ncclAlgoStr`），`src/config/algorithm_registry.cc` 的 `algRegistry[]` 中也没有 halving/doubling 条目，`grep -riE 'halving|doubling' third_party/nccl/src` 无匹配，所以 NCCL 没有独立的 Halving-Doubling（Rabenseifner）AllReduce；最接近的是 PAT——仅 AllGather/ReduceScatter、仅 Simple，`src/include/collectives.h` 的 `PatRSAlgorithm`/`PatAGAlgorithm` 按 2 的幂维度（`nrPow2`、`firstBitSet`）选对端，是对数步算法。
- PyTorch 未克隆到 `third_party/`，`ProcessGroupNCCL` 相关内容需另行确认。

## 待深入

- 书中 §6.4.3 提到的"实验 7-3 公开记录"和 `calculations/` 下的复算脚本，可作为本仓库 `labs/ccl_sim/` 代价模型实验的对照。待确认：`third_party/ai-infra-book/calculations/calc.py` 的主题列表与运行方式；确认方式：`python3 third_party/ai-infra-book/calculations/calc.py --help`（只读运行，不修改仓库）。
