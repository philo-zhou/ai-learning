# 分阶段学习路线（10 周）

> 层次：总览 · 来源：third_party/ai-infra-book@3bdcb4fc、third_party/nccl@3e46c089、third_party/nccl-tests@afd59aba、third_party/perftest@05200c2f、third_party/rdma-core@bd3282a1、third_party/Triton-distributed@7908e4ea、third_party/DeepEP@b5a33eef、third_party/vllm@02653fa2、third_party/sglang@d2a43644

## 一句话

先用代价模型建立"算多少字节、走哪条路、等多久"的直觉，再自下而上（RDMA / scale-up → NCCL → 框架）读源码，最后用通算融合和排查实践收尾，每阶段都有可验收的产出。

开始第 1 周之前，先通读 [00-start-here.md](00-start-here.md) 建立推理通信的全景；它的"主线学习顺序"是本路线的问题驱动版，两者可以对照着用。

## 路线总览

```text
周次   1     2     3     4     5     6     7     8     9     10
      [算法+代价模型][RDMA/网络 ][scale-up][ NCCL 内部 ][框架][通算融合][排查][面试]
阶段    S1(1-2)      S2(3)      S3(4)    S4(5-6)     S5(7) S6(8)   S7(9) S8(10)
```

把同样的 8 个阶段按层次摆开，可以看到"先代价模型、再自下而上、最后收尾"的走向：

```text
跨层/实践       S8 interview ◄── S7 perf ◄── S6 fusion
                                                ▲
上层                                         S5 framework
                                                ▲
中层      S1 algo + cost model               S4 NCCL internals
             │                                  ▲
下层         └───► S2 RDMA ───► S3 scale-up ────┘
```

- 书章节路径均以 `third_party/ai-infra-book/manuscripts/` 为前缀；详细章节说明见 [01-ai-infra-book-map.md](01-ai-infra-book-map.md)。
- 每阶段末尾到 [02-jd-breakdown.md](02-jd-breakdown.md) 更新自评，并在 `docs/PROGRESS.md` 记录证据。
- 没有 GPU / RDMA 机器时，S2、S3、S7 的实践部分降级为"读源码 + 手算"，并在验收中注明 not-run。

## S1（第 1~2 周）集合通信算法与代价模型

- **目标**：能推导 Ring / Tree / Recursive Halving-Doubling 的轮数与字节数，能用 α-β 模型判断小消息、大消息谁占主导；能解释 algbw 与 busbw。
- **阅读材料**：
  - 书 `04-加速器架构.md` §4.5.3；`06-超节点.md` §6.2、§6.4.1~6.4.2、§6.5.1（递归 vs Swing）。
  - 书 `07-数据中心网络.md` §7.2.1（例题 7.2：连续环 / 交错环 / 分层 / 在网归约）。
  - `third_party/nccl-tests/src/all_reduce.cu` 的 `AllReduceGetBw`、`src/alltoall.cu` 的 `AlltoAllGetBw`。
- **产出物**：
  - [03-ccl-algorithms](../03-ccl-algorithms/README.md) 下 Ring、Tree、Halving-Doubling、分层 AllReduce 各一篇笔记。
  - `labs/ccl_sim/`（目录已存在）：Python 模拟器，可按 n、M、α、β 计算各算法时间，并逐步模拟数据块流动校验正确性。
- **验收方式**：模拟器复现书中八卡 α=0.822 μs、B=450 GB/s 下环/树交点约 680 KiB；复现例题 7.2 跨服务器字节 720 / 5760 / 384 MiB；能闭卷推导 busbw 系数 $2(n-1)/n$。

## S2（第 3 周）RDMA 与数据中心网络

- **目标**：能讲清 Verbs 对象和 QP 建链流程、单边/双边语义、IB 与 RoCEv2 差异、PFC + ECN/DCQCN 分工、GPUDirect RDMA。
- **阅读材料**：
  - `third_party/rdma-core/libibverbs/examples/rc_pingpong.c`（`ibv_reg_mr`、`ibv_create_qp`、`ibv_modify_qp`、`ibv_post_send`、CQ 轮询）。
  - `third_party/perftest/src/write_bw.c`、`write_lat.c`、`read_bw.c`、`send_bw.c`；`perftest_parameters.c`（参数含义）。
  - 书 `07-数据中心网络.md` §7.1.2、§7.2.4~7.2.5、§7.3.1~7.3.4、§7.4.2、§7.5.1~7.5.4。
- **产出物**：[01-rdma](../01-rdma/README.md) 下 Verbs 编程、QP 状态机、RoCEv2 与拥塞控制、perftest 使用四篇笔记；`rc_pingpong.c` 带注释的调用时序图。
- **验收方式**：有 RDMA 机器时跑通 `ib_write_bw` / `ib_write_lat` 并记录结果与链路理论值比值；无机器时完成书 §7.5.2 习题（PFC headroom、端到端反馈允许的最大 N）手算，结果写进笔记。

## S3（第 4 周）scale-up 互联

- **目标**：能画出 HGX 类服务器的 GPU / NVSwitch / PCIe Switch / NIC / NUMA 拓扑；讲清 NCCL 的 P2P、SHM、NVLS transport 各自在何时被选用。
- **阅读材料**：
  - 书 `06-超节点.md` §6.5.1~6.5.5、`04-加速器架构.md` §4.5.2；案例 `third_party/ai-infra-book/case-studies/pcie-staging-and-numa.md`。
  - `third_party/nccl/src/transport/p2p.cc`、`shm.cc`、`nvls.cc`、`src/mnnvl.cc`；`src/graph/paths.cc`（P2P 可达性判定、`NCCL_P2P_LEVEL`）。
- **产出物**：[02-scale-up](../02-scale-up/README.md) 下 NVLink/NVSwitch、P2P/SHM、NVLS 三篇笔记。
- **验收方式**：[多GPU] 用 `NCCL_TOPO_DUMP_FILE` 导出拓扑 XML 并逐项解释；用 `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` 对比 nccl-tests 带宽变化并解释原因。无机器时记 not-run，改为口述 transport 选择流程。

## S4（第 5~6 周）NCCL 内部实现

- **目标**：能从 `ncclCommInitRank` 一路讲到 kernel 中的 `runRing`，包括拓扑探测、图搜索、channel、协议、tuning、proxy。
- **阅读材料**（`third_party/nccl/src/` 下）：
  - 第 5 周：`init.cc`（`ncclCommInitRank`）、`bootstrap.cc`、`graph/topo.cc`（`ncclTopoGetSystem`）、`graph/paths.cc`（`ncclTopoComputePaths`）、`graph/search.cc`（`ncclTopoCompute`）、`graph/rings.cc`、`graph/trees.cc`、`graph/connect.cc`、`channel.cc`。
  - 第 6 周：`collectives.cc`（`ncclAllReduce`）、`group.cc`、`enqueue/`、`tuning/`（`tuning.cc`、`ring.cc`、`tree.cc`、`pat.cc`、`nvls.cc`）、`device/primitives.h`、`device/prims_ll.h`、`prims_ll128.h`、`prims_simple.h`、`device/all_reduce.h`、`proxy.cc`（`ncclProxyProgress`）、`transport/net.cc`、`transport/net_ib/`。
  - 书 `05-算子与运行时.md` §5.1.3；`06-超节点.md` §6.4.3~6.4.4；案例 `communication-tuning.md`。
- **产出物**：[04-nccl-internals](../04-nccl-internals/README.md) 下初始化、拓扑与图搜索、channel 与协议、tuning、transport 与 proxy 五篇源码导读，每篇含"路径 + 函数 + 作用"表。
- **验收方式**：不看资料画出 NCCL 一次 AllReduce 的完整调用链；[多GPU] 用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` 跑 nccl-tests，逐行解释日志中的 ring/tree/channel 信息；扫 `NCCL_PROTO=LL,LL128,Simple` 并解释曲线交叉点。

验收要求"画出完整调用链"，可以先按下面的骨架读上面列出的文件（左边是函数或文件，均以 `third_party/nccl/src/` 为前缀；右边是作用）：

```text
初始化：每个 communicator 一次（第 5 周）
ncclCommInitRank (init.cc)
  ├─► bootstrap.cc                               带外建立 rank 之间的联系
  ├─► ncclTopoGetSystem (graph/topo.cc)          探测机内拓扑
  ├─► ncclTopoComputePaths (graph/paths.cc)      算路径、判定 P2P 可达
  ├─► ncclTopoCompute (graph/search.cc)          搜索 ring / tree 图
  │     └─► graph/rings.cc, graph/trees.cc
  └─► graph/connect.cc, channel.cc               建 channel 与 peer 连接

每次集合调用（第 6 周）
ncclAllReduce (collectives.cc)
  └─► group.cc ─► enqueue/ ─► tuning/            选算法 × 协议
        └─► device kernel: device/all_reduce.h (runRing ...)
              primitives.h + prims_ll.h / prims_ll128.h / prims_simple.h
                └─► transport/net.cc, transport/net_ib/ ◄──► proxy.cc (ncclProxyProgress)
```

## S5（第 7 周）框架集成

- **目标**：能讲清 DDP/FSDP/TP/EP/PD 分离各自调用了哪些集合操作、在哪个层次调用；理解推理框架为什么自带 custom allreduce。
- **阅读材料**：
  - 书 `10-训练系统.md` §10.2.1、§10.3.3~10.3.4、§10.5.2；`09-分布式推理.md` §9.2.2、§9.4.1~9.4.3。
  - `third_party/vllm/vllm/distributed/parallel_state.py`（`GroupCoordinator`）、`device_communicators/pynccl.py`（`PyNcclCommunicator`）、`custom_all_reduce.py`、`symm_mem.py`、`all2all.py`。
  - `third_party/sglang/python/sglang/srt/distributed/device_communicators/`（`custom_all_reduce.py`、`pymscclpp.py`、`torch_symm_mem.py`、`mooncake_transfer_engine.py`）。
  - `third_party/DeepEP/deep_ep/buffers/ep.py`（`dispatch`、`combine`）、`csrc/kernels/ep/`。
  - 待确认：PyTorch 未克隆，`ProcessGroupNCCL` 与 DDP `Reducer` 需另行获取源码；确认方式：克隆 pytorch 后查 `torch/csrc/distributed/c10d/`。
- **产出物**：[05-framework-integration](../05-framework-integration/README.md) 下 vLLM TP 通信链、DeepEP dispatch/combine、训练并行通信表三篇笔记。
- **验收方式**：给定一个模型配置（层数、hidden、TP/EP/DP 度），能手算一个 step / 一个 decode token 的各类通信量与次数，并指出走 NVLink 还是 RDMA。

## S6（第 8 周）通信与计算融合

- **目标**：理解 tile 级通算融合的编程模型（对称内存、signal/wait），能读懂并改动一个 Triton-distributed 的 AllGather+GEMM kernel。
- **阅读材料**：
  - `third_party/Triton-distributed/tutorials/01-distributed-notify-wait.py` ~ `04-deepseek-infer-all2all.py`。
  - `third_party/Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py`、`gemm_reduce_scatter.py`、`gemm_allreduce.py`、`low_latency_all_to_all.py`、`comm_perf_model.py`。
  - 书 `04-加速器架构.md` §4.4.4；`05-算子与运行时.md` §5.5.4；`06-超节点.md` §6.4.3~6.4.4；案例 `remote-ordering-and-completion.md`、`mesh-shape-and-slicing.md`。
- **产出物**：[06-comm-compute-fusion](../06-comm-compute-fusion/README.md) 下编程模型、AG+GEMM 导读、MoE All2All 融合三篇笔记；待建 `labs/overlap/`：对比"先通信后 GEMM"、"stream 级重叠"、"融合 kernel"三种实现的时间。
- **验收方式**：[多GPU] 跑通 tutorial 02 并记录与 NCCL AllGather + cuBLAS 串行的耗时对比；无机器时，用书实验 6-5 的数据解释"重叠收益不能用单独时间相减"。

## S7（第 9 周）性能调优与故障排查

- **目标**：形成"网卡 → 库 → 框架"的分层排查方法；能快速定位慢、hang、带宽不达标三类问题。
- **阅读材料**：
  - `third_party/nccl-tests/src/common.cu`（参数 `-b/-e/-f/-g` 等定义）、`all_reduce.cu`。
  - 关键环境变量的读取点：`third_party/nccl/src/graph/paths.cc`（`NCCL_P2P_DISABLE`/`NCCL_P2P_LEVEL`）、`src/transport/shm.cc`（`SHM_DISABLE`）、`src/transport/net_ib/connect.cc`（`IB_GID_INDEX`、`IB_TIMEOUT`、`IB_QPS_PER_CONNECTION`）、`src/graph/search.cc`（`CROSS_NIC`）、`src/graph/connect.cc`（`MIN_NCHANNELS`）、`src/init.cc`（`BUFFSIZE`）、`src/transport/net_ib/common.h`（`NCCL_NET_GDR_LEVEL` 提示）。
  - `third_party/nccl/src/ras/`、`src/plugin/profiler/`。
  - 书 `07-数据中心网络.md` §7.6.1~7.6.5、`10-训练系统.md` §10.4.5；案例 `collective-paths-and-diagnosis.md`。
- **产出物**：[07-perf-troubleshooting](../07-perf-troubleshooting/README.md) 下 nccl-tests 使用、环境变量速查、排查 checklist、典型故障案例四篇笔记。
- **验收方式**：环境变量速查表中每一项都附 `third_party/nccl/src/` 读取位置；[多GPU/RDMA] 人为制造一次问题（如错误的 `NCCL_SOCKET_IFNAME` 或 `NCCL_IB_HCA`）并用日志定位，记录过程。

分层排查的顺序是自下而上：下一层确认正常，才往上一层找原因（工具出处见上方阅读材料与题库第 47 题）：

```text
3. framework profile                     ← 看现象：步时间、通信占比、各 rank 发起时刻
        ▲ 库层正常再往上
2. nccl-tests (all_reduce_perf)          ← 先单机、再两机；配合 NCCL_DEBUG=INFO 看日志
        ▲ 网卡层正常再往上
1. perftest (ib_write_bw, --use_cuda)    ← 每张网卡点对点线速、GPU 内存路径
```

## S8（第 10 周）面试准备

- **目标**：题库全部能口述；准备 2~3 个可讲 5 分钟的项目表述。
- **阅读材料**：[08-interview/01-question-bank.md](../08-interview/01-question-bank.md)，回看各章节笔记的"面试自测"。
- **产出物**：[08-interview](../08-interview/README.md) 下题库补充个人答案；用 STAR 写 `labs/ccl_sim/` 与一项性能实验的项目表述。
- **验收方式**：随机抽 20 题限时口述，录音回听；[02-jd-breakdown.md](02-jd-breakdown.md) 中 80% 以上知识点达到"能讲清"，至少 5 项达到"能动手"。

## 待深入

- 若时间充裕，追加 MSCCL++ / NVSHMEM 源码阅读。待确认：本地 `third_party/` 未克隆 MSCCL++ 与 NVSHMEM 独立仓库（sglang 中有 `pymscclpp.py` 封装，Triton-distributed 下有 `shmem/nvshmem_bind/`）；确认方式：`ls third_party/`。
