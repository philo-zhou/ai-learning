# JD 拆解：集合通信库研发

> 层次：总览 · 来源：third_party/ai-infra-book@3bdcb4fc、third_party/nccl@3e46c089、third_party/nccl-tests@afd59aba、third_party/perftest@05200c2f、third_party/rdma-core@bd3282a1、third_party/Triton-distributed@7908e4ea、third_party/DeepEP@93eb6eb2、third_party/vllm@02653fa2、third_party/sglang@d2a43644

## 一句话

把字节跳动基础设施网络团队"集合通信库研发"JD 的 5 条职责（R1~R5）和 5 条要求（Q1~Q5）拆成可自评的知识点，每个知识点落到一个 notes 目录和一个可动手的实践。

## 使用方式

- 自评栏四档：**未开始 / 了解 / 能讲清 / 能动手**。"能讲清"= 能不看资料在白板上推导或画图并回答追问；"能动手"= 有本仓库可复现的实验或代码产出。
- 自评栏初始全部为"未开始"，学习时直接改表格。整体进度另记在 `docs/PROGRESS.md`。
- 实践列中 `labs/ccl_sim/` 已建成（`python3 -m labs.ccl_sim`、`python3 scripts/check_repo.py labs`），其余 `labs/` 子目录为计划新建的实验，标"待建"。
- 实践需要的硬件：标 [CPU] 的可在笔记本完成；[1×GPU]/[多GPU]/[RDMA] 需要对应机器。

下图把后文各表的"notes 目录"列汇总成一张矩阵：X 表示该 JD 条目至少有一个知识点落在这个目录（01~02 下层、03~04 中层、05 上层、06 跨层、07 实践）。可以据此看出 04-nccl-internals 覆盖的 JD 条目最多（5 条）。

```text
      │   01       02       03       04       05       06       07    │
      │  rdma    scaleup   algo     nccl     frame   fusion    perf   │
──────┼───────────────────────────────────────────────────────────────┤
  R1  │    .        .        X        X        .        .        X    │
  R2  │    X        X        .        X        .        .        .    │
  R3  │    .        .        .        .        .        X        .    │
  R4  │    .        X        X        X        X        .        .    │
  R5  │    .        .        .        .        X        .        X    │
  Q1  │    X        X        .        .        X        .        .    │
  Q2  │    X        .        .        .        .        .        .    │
  Q3  │    .        .        X        X        .        .        .    │
  Q4  │    .        .        .        X        .        .        .    │
  Q5  │    .        .        .        .        .        .        X    │
──────┴───────────────────────────────────────────────────────────────┘
```

## 职责拆解

### R1 负责 CCL 设计开发优化，支持多卡/多机通信

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| communicator / rank / uniqueId 与 bootstrap 建组 | [04-nccl-internals](../04-nccl-internals/README.md) | 阅读 `third_party/nccl/src/init.cc` 的 `ncclCommInitRank` 与 `src/bootstrap.cc`，画出初始化时序 | 未开始 |
| 集合操作语义（AllReduce/ReduceScatter/AllGather/Broadcast/Reduce/All2All/Send-Recv） | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | [CPU] `labs/ccl_sim/algorithms.py` 用纯 Python 模拟多 rank 的各操作，`labs/ccl_sim/tests/` 校验结果 | 未开始 |
| 异步语义：调用返回 ≠ 完成；stream/event 依赖；group 调用 | [04-nccl-internals](../04-nccl-internals/README.md) | 阅读 `third_party/nccl/src/group.cc` 的 `ncclGroupStart`；书 `05-算子与运行时.md` §5.1.3 | 未开始 |
| 通信库架构分层（API → enqueue → 调度 → device kernel → transport → proxy） | [04-nccl-internals](../04-nccl-internals/README.md) | 按 `third_party/nccl/src/enqueue/`、`src/device/`、`src/transport/`、`src/proxy.cc` 各写一页导读 | 未开始 |
| 算法/协议的自动选择与在线调优（cost model、AutoCCL 思路） | [04-nccl-internals](../04-nccl-internals/README.md) | 阅读 `third_party/nccl/src/tuning/tuning.cc`、`cost_model.cc`；书 §6.4.3 | 未开始 |
| 容错与可观测（RAS、profiler 插件、超时） | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | 浏览 `third_party/nccl/src/ras/`、`src/plugin/`；nccl-tests 中 `profiler_v*.h` | 未开始 |

### R2 基于 RDMA 与 scale-up 互联研发高性能通信库

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| Verbs 对象：PD / MR / CQ / QP / WR / WC；RC/UC/UD 区别 | [01-rdma](../01-rdma/README.md) | [RDMA] 编译运行 `third_party/rdma-core/libibverbs/examples/rc_pingpong.c`，逐行注释 `ibv_create_qp`→`ibv_modify_qp`→`ibv_post_send`→poll CQ | 未开始 |
| QP 状态机 RESET→INIT→RTR→RTS 与建链参数（GID、MTU、PSN、timeout） | [01-rdma](../01-rdma/README.md) | 同上，记录每次 `ibv_modify_qp` 的 attr mask | 未开始 |
| 单边（Write/Read/Atomic）与双边（Send/Recv）语义差异 | [01-rdma](../01-rdma/README.md) | [RDMA] perftest：`ib_write_bw`/`ib_read_bw`/`ib_send_bw`/`ib_write_lat`（源码 `third_party/perftest/src/write_bw.c` 等） | 未开始 |
| IB vs RoCEv2；PFC / ECN / DCQCN；无损网络与 headroom | [01-rdma](../01-rdma/README.md) | 书 `07-数据中心网络.md` §7.5.2 习题手算；[RDMA] 待确认：交换机侧能否观测 PFC 计数 | 未开始 |
| GPUDirect RDMA、GDRCopy、NIC 与 GPU 的 PCIe 亲和 | [01-rdma](../01-rdma/README.md)、[02-scale-up](../02-scale-up/README.md) | [RDMA+GPU] perftest 加 `--use_cuda=<gpu id>`（参数见 `third_party/perftest/src/perftest_parameters.c` 的帮助输出）跑 GPU 内存，对比主机内存 | 未开始 |
| NVLink/NVSwitch、P2P（CUDA IPC）、SHM transport | [02-scale-up](../02-scale-up/README.md) | 阅读 `third_party/nccl/src/transport/p2p.cc`、`shm.cc`；[多GPU] `nvidia-smi topo -m` | 未开始 |
| NVLS（NVLink SHARP / multicast）与 MNNVL | [02-scale-up](../02-scale-up/README.md) | 阅读 `third_party/nccl/src/transport/nvls.cc`（`NCCL_NVLS_ENABLE` 默认值 2）、`src/mnnvl.cc` | 未开始 |
| NCCL IB transport 实现（建链、多 QP、GDR、GDAKI） | [04-nccl-internals](../04-nccl-internals/README.md) | 阅读 `third_party/nccl/src/transport/net_ib/connect.cc`（`IB_GID_INDEX`、`IB_TIMEOUT`、`IB_QPS_PER_CONNECTION` 参数）、`gdr.cc`、`gdaki/` | 未开始 |
| 网卡状态规模：QP 上下文、连接数扩展性 | [01-rdma](../01-rdma/README.md) | 书 §7.4.2 算例手算 | 未开始 |

### R3 研发通信与计算融合算子（Triton）

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| 重叠的三种粒度：stream 级重叠、kernel 内 tile 级融合、persistent kernel | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 书 §5.5.4、§6.4.4 | 未开始 |
| 对称内存 / NVSHMEM 编程模型（put/get/signal/wait、fence/quiet） | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 跑 `third_party/Triton-distributed/tutorials/01-distributed-notify-wait.py`、`02-intra-node-allgather.py` | 未开始 |
| AllGather+GEMM、GEMM+ReduceScatter、GEMM+AllReduce | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 阅读 `third_party/Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py`、`gemm_reduce_scatter.py`、`gemm_allreduce.py` | 未开始 |
| MoE All2All 与 GEMM 融合、低延迟 All2All | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 阅读 `.../kernels/nvidia/all_to_all_single_gemm.py`、`low_latency_all_to_all.py`、`ep_a2a.py`；tutorial `04-deepseek-infer-all2all.py` | 未开始 |
| 通信与计算争用 SM/HBM、SM 分配 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | [CPU] 复现书实验 6-5 思路：Gloo AllReduce 与矩阵乘并发；[1×GPU] 待建 `labs/overlap/` | 未开始 |
| 融合的正确性约束（AllReduce 后 RMSNorm 不能交换顺序） | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 书 §6.4.4 实验 6-5(c) 手算 | 未开始 |
| 性能模型指导 tile 切分 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) | 阅读 `.../kernels/nvidia/comm_perf_model.py`、`gemm_perf_model.py` | 未开始 |

### R4 优化 AllReduce / All2All 在单机内、多机间的性能

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| α-β 代价模型；Ring / Tree 公式与交点 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | [CPU] `labs/ccl_sim/cost_model.py` 已实现各算法 α-β 公式；待做：代入书 §6.4.2 的参数复现 680 KiB 交点（`python3 -m labs.ccl_sim --alpha-us ... --bw-gbps ...`） | 未开始 |
| 分层 AllReduce（机内 RS → 机间 AR → 机内 AG）与多轨 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | [CPU] 复现书例题 7.2 三方案跨服务器字节数 720/5760/384 MiB | 未开始 |
| NCCL 协议 LL / LL128 / Simple 的取舍 | [04-nccl-internals](../04-nccl-internals/README.md) | 阅读 `third_party/nccl/src/device/prims_ll.h`、`prims_ll128.h`、`prims_simple.h`；[多GPU] nccl-tests 扫 `NCCL_PROTO` | 未开始 |
| channel 数、buffer 大小与带宽/SM 占用 | [04-nccl-internals](../04-nccl-internals/README.md) | `NCCL_MIN_NCHANNELS`（`src/graph/connect.cc`）、`NCCL_BUFFSIZE`（`src/init.cc`）；[多GPU] nccl-tests 扫参 | 未开始 |
| All2All：等长 vs 不等长，Send/Recv 组调用，busbw 换算 $(n-1)/n$ | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | 阅读 `third_party/nccl-tests/src/alltoall.cu` 的 `AlltoAllGetBw`；[多GPU] `alltoall_perf` | 未开始 |
| 推理小消息 AllReduce：custom allreduce / one-shot / two-shot / symm mem | [05-framework-integration](../05-framework-integration/README.md) | 阅读 `third_party/vllm/vllm/distributed/device_communicators/custom_all_reduce.py`、`symm_mem.py`、`quick_all_reduce.py` | 未开始 |
| MoE EP dispatch/combine（NVLink 域 + RDMA 域） | [05-framework-integration](../05-framework-integration/README.md) | 阅读 `third_party/DeepEP/deep_ep/buffers/ep.py` 的 `dispatch`/`combine` 与 `csrc/kernels/ep/` | 未开始 |
| 在网归约（SHARP / CollNet / NVLS） | [02-scale-up](../02-scale-up/README.md) | 书 §7.2.1 第四种方案；阅读 `third_party/nccl/src/transport/coll_net.cc` | 未开始 |

### R5 解决大规模分布式训练、推理中的通信问题

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| 各并行方式的通信模式（DP/FSDP/TP/SP/CP/PP/EP） | [05-framework-integration](../05-framework-integration/README.md) | 书 §6.2、§10.2.1；画每种并行一层的通信表 | 未开始 |
| DDP 梯度分桶与重叠 | [05-framework-integration](../05-framework-integration/README.md) | 书 §10.3.3；待确认：PyTorch 源码未克隆，`reducer` 实现需另取 | 未开始 |
| 推理框架通信层（vLLM GroupCoordinator / PyNcclCommunicator） | [05-framework-integration](../05-framework-integration/README.md) | 阅读 `third_party/vllm/vllm/distributed/parallel_state.py` 的 `GroupCoordinator`、`device_communicators/pynccl.py` 的 `PyNcclCommunicator` | 未开始 |
| PD 分离 KV 传输、RL 权重同步 | [05-framework-integration](../05-framework-integration/README.md) | 书 §9.2.2、§10.5.2；阅读 `third_party/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | 未开始 |
| hang / 超时 / straggler / 慢链路定位 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | 书 §7.6.1（MegaScale）、§10.4.5；`NCCL_DEBUG=INFO` 日志解读 | 未开始 |

## 要求拆解

### Q1 分布式系统、AI 集群架构

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| 服务器内拓扑（GPU-NVSwitch-PCIe Switch-NIC-CPU NUMA） | [02-scale-up](../02-scale-up/README.md) | [多GPU] `nvidia-smi topo -m`、`NCCL_TOPO_DUMP_FILE` 导出 NCCL 识别的拓扑 | 未开始 |
| 集群网络：Clos / fat-tree / rail-optimized / 超售比 | [01-rdma](../01-rdma/README.md) | 书 §7.1.2、§7.2.5 | 未开始 |
| 超节点（NVL72、UB/CloudMatrix）对并行策略的影响 | [02-scale-up](../02-scale-up/README.md) | 书 §6.5.3~6.5.5、§7.6.4 | 未开始 |
| 一致性/同步/故障域等分布式基础 | [05-framework-integration](../05-framework-integration/README.md) | 书 §7.4.3、§10.4 | 未开始 |

### Q2 网络通信原理、RDMA

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| RDMA 为什么快：kernel bypass、zero-copy、硬件传输层 | [01-rdma](../01-rdma/README.md) | 对比 `ib_write_lat` 与 TCP ping-pong 时延 | 未开始 |
| RoCEv2 报文封装（UDP 4791）、GID 选择 | [01-rdma](../01-rdma/README.md) | `show_gids`/`ibv_devinfo -v`；待确认：机器上是否有 `show_gids` | 未开始 |
| 拥塞控制、ECMP 哈希冲突、多路径 | [01-rdma](../01-rdma/README.md) | 书 §7.5.3；`NCCL_IB_QPS_PER_CONNECTION` 增加 QP 分散 ECMP | 未开始 |
| 在途窗口与报文速率（小消息瓶颈） | [01-rdma](../01-rdma/README.md) | 书例题 7.3；perftest 扫消息大小看 msg rate | 未开始 |

### Q3 集合通信算法（Ring / Tree / Halving-Doubling 等）

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| Ring：$2(n-1)$ 步，每卡发送 $2(n-1)M/n$ | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | [CPU] `labs/ccl_sim/` 逐步模拟，测试断言步数与发送量 | 未开始 |
| Tree / double binary tree：$O(\log n)$ 轮，pipeline 分段 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | 阅读 `third_party/nccl/src/graph/trees.cc`、`src/device/all_reduce.h` 的 `runTreeUpDown`/`runTreeSplit` | 未开始 |
| Recursive Halving-Doubling（Rabenseifner）、Recursive Doubling | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | [CPU] `labs/ccl_sim/` 已实现并与 ring 对比；NCCL 中无独立 Halving-Doubling 算法（`NCCL_ALGO_*` 只有 Tree/Ring/CollNet×2/NVLS/NVLSTree/PAT，候选表见 `third_party/nccl/src/config/algorithm_registry.cc`；最接近的是只做 AG/RS 的 PAT，详见 03-ccl-algorithms） | 未开始 |
| PAT、NVLS、CollNet 等 NCCL 内置算法 | [04-nccl-internals](../04-nccl-internals/README.md) | 阅读 `third_party/nccl/src/tuning/pat.cc`、`nvls.cc`、`collnet.cc` | 未开始 |
| 拓扑感知算法（Swing、分层） | [03-ccl-algorithms](../03-ccl-algorithms/README.md) | 书 §6.5.1；`case-studies/network-planning-and-collectives.md` | 未开始 |

### Q4 NCCL 实现原理

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| 拓扑探测与路径计算 | [04-nccl-internals](../04-nccl-internals/README.md) | `third_party/nccl/src/graph/topo.cc` 的 `ncclTopoGetSystem`、`paths.cc` 的 `ncclTopoComputePaths` | 未开始 |
| ring/tree 图搜索与 channel 建立 | [04-nccl-internals](../04-nccl-internals/README.md) | `src/graph/search.cc` 的 `ncclTopoCompute`、`rings.cc`、`connect.cc` | 未开始 |
| 调优模型与算法选择 | [04-nccl-internals](../04-nccl-internals/README.md) | `src/include/graph.h` 中 `ncclTopoTuneModel` 声明，`src/tuning/` | 未开始 |
| device 端 primitives 与 kernel | [04-nccl-internals](../04-nccl-internals/README.md) | `src/device/primitives.h`、`all_reduce.h` 的 `runRing` | 未开始 |
| proxy 线程与网络插件 | [04-nccl-internals](../04-nccl-internals/README.md) | `src/proxy.cc` 的 `ncclProxyProgress`、`src/include/net.h` | 未开始 |
| P2P_LEVEL / CROSS_NIC 等拓扑相关开关 | [04-nccl-internals](../04-nccl-internals/README.md) | `src/graph/paths.cc`（`NCCL_P2P_DISABLE`/`NCCL_P2P_LEVEL` 读取处）、`src/graph/search.cc`（`CROSS_NIC` 默认 2） | 未开始 |

### Q5 性能调优与故障排查

| 知识点 | notes 目录 | 推荐实践 | 自评 |
|---|---|---|---|
| algbw vs busbw 与理论峰值对比 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | 阅读 `third_party/nccl-tests/src/all_reduce.cu` 的 `AllReduceGetBw`（factor = 2(n-1)/n） | 未开始 |
| nccl-tests 使用（`-b/-e/-f/-g`、多机 MPI 启动） | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | [多GPU] 编译 `third_party/nccl-tests`，跑 `all_reduce_perf`，记录曲线 | 未开始 |
| 关键环境变量（`NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS`、`NCCL_ALGO`、`NCCL_PROTO`、`NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、`NCCL_IB_GID_INDEX`、`NCCL_NET_GDR_LEVEL`、`NCCL_P2P_DISABLE`、`NCCL_SHM_DISABLE`） | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | 每个变量在 `third_party/nccl/src/` 中找到读取点，写成速查表 | 未开始 |
| 分层排查：perftest（网卡）→ nccl-tests（库）→ 框架 profile | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | [RDMA] 先 `ib_write_bw` 确认链路，再 nccl-tests | 未开始 |
| hang 定位、发起时间偏差、straggler | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) | 书 §7.6.1、`case-studies/collective-paths-and-diagnosis.md` | 未开始 |
