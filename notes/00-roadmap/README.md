# 笔记总入口

> 层次：总览 · 来源：third_party/ai-infra-book@3bdcb4fc、third_party/nccl@3e46c089、third_party/vllm@02653fa2、third_party/DeepEP@93eb6eb2、third_party/DeepEP-hybrid-ep@10d4dd73

## 一句话

本仓库围绕"集合通信库（CCL）研发"岗位组织笔记：自下而上分为互联与网络（下层）、集合通信库（中层）、训练推理框架（上层）三层，加上跨层的通算融合与性能排查。

> **第一次来，先读 [00-start-here.md](00-start-here.md)**：从大模型推理出发，讲清各并行策略对应的通信、EP 通信的主流方案和发展趋势，并把下面每一章挂到这条主线上。

## 三层结构

| 层次 | 关心的问题 | 章节 |
|---|---|---|
| 下层 scale-out | 机间如何高效可靠地搬字节：Verbs、QP、RoCEv2/IB、拥塞控制、GPUDirect RDMA | [01-rdma](../01-rdma/README.md) |
| 下层 scale-up | 机内/超节点内如何搬字节：NVLink/NVSwitch、P2P/SHM、NVLS、CUDA IPC | [02-scale-up](../02-scale-up/README.md) |
| 中层 算法 | 搬哪些字节、分几轮：Ring/Tree/Halving-Doubling、分层、α-β 代价模型 | [03-ccl-algorithms](../03-ccl-algorithms/README.md) |
| 中层 实现 | NCCL 如何把算法落到拓扑和硬件：初始化、拓扑、channel、协议、transport、proxy | [04-nccl-internals](../04-nccl-internals/README.md) |
| 中层 EP 专用库 | MoE dispatch/combine 怎样在 kernel 里落地：DeepEP low-latency、Hybrid-EP、NCCL EP | [09-ep-kernels](../09-ep-kernels/README.md) |
| 上层 | 框架在何处、以何种语义调用 CCL：PyTorch、vLLM、SGLang、DeepEP | [05-framework-integration](../05-framework-integration/README.md) |
| 跨层 | 通信与计算如何融合：Triton-distributed、NVSHMEM、对称内存 | [06-comm-compute-fusion](../06-comm-compute-fusion/README.md) |
| 实践 | 怎么测、怎么调、怎么排查：nccl-tests、perftest、环境变量 | [07-perf-troubleshooting](../07-perf-troubleshooting/README.md) |
| 求职 | 面试题与项目表述 | [08-interview](../08-interview/README.md) |

## 调用链全景

```text
 上层 ┌──────────────────────────────┐   ┌───────────────────────────────┐
      │ 训练：PyTorch DDP / FSDP      │   │ 推理：vLLM / SGLang TP、EP    │
      │  梯度分桶 AllReduce、RS + AG   │   │  GroupCoordinator             │
      └──────────────┬───────────────┘   └───────┬───────────┬───────────┘
                     │                           │           │ 小消息 / MoE
                     v                           v           v
      ┌──────────────────────────────┐   ┌────────────────┐ ┌───────────────────┐
      │ torch.distributed            │   │ PyNcclComm     │ │ custom allreduce  │
      │  ProcessGroupNCCL (c10d)     │   │ (ctypes→NCCL)  │ │ symm mem / DeepEP │
      └──────────────┬───────────────┘   └───────┬────────┘ └─────────┬─────────┘
                     └───────────┬───────────────┘          绕过 NCCL │ 直接用
 中层                            v                       CUDA IPC/NVSHMEM/IBGDA
      ┌──────────────────────────────────────────────────────────┐    │
      │ NCCL                                                     │    │
      │  API(ncclAllReduce/Group) → enqueue → tuning(算法×协议)  │    │
      │  → device kernel(runRing/runTree, LL/LL128/Simple)       │    │
      │  → channel/peer 连接 ←→ proxy 线程(网络收发进度)          │    │
      └───────────────┬───────────────────────────┬──────────────┘    │
                      │ 机内 transport            │ 机间 transport    │
                      v                           v                   │
 下层     ┌───────────────────────┐   ┌──────────────────────────┐    │
          │ P2P(CUDA IPC/NVLink)  │   │ NET: IB verbs / RoCEv2   │    │
          │ SHM(主机共享内存)      │   │ (net_ib, GDR, GDAKI)     │    │
          │ NVLS(NVSwitch 多播归约)│   │ CollNet(SHARP) / Socket  │    │
          └───────────┬───────────┘   └────────────┬─────────────┘    │
                      v                            v                  v
 硬件     ┌───────────────────────┐   ┌──────────────────────────────────┐
          │ GPU HBM / SM / CE     │   │ NIC(ConnectX 等) → ToR/Leaf/Spine │
          │ NVLink / NVSwitch     │   │ PCIe Switch / NUMA / PFC+ECN     │
          └───────────────────────┘   └──────────────────────────────────┘
```

图中本地可核对的入口（均已在克隆中看到）：

| 图中节点 | 本地路径 | 说明 |
|---|---|---|
| vLLM GroupCoordinator | `third_party/vllm/vllm/distributed/parallel_state.py` | `class GroupCoordinator`，持有 `device_communicator` |
| PyNcclComm | `third_party/vllm/vllm/distributed/device_communicators/pynccl.py` | `class PyNcclCommunicator`；ctypes 封装见同目录 `pynccl_wrapper.py` |
| custom allreduce / symm mem | `third_party/vllm/vllm/distributed/device_communicators/custom_all_reduce.py`、`symm_mem.py`；`third_party/sglang/python/sglang/srt/distributed/device_communicators/` | 推理小消息专用路径 |
| DeepEP | `third_party/DeepEP/deep_ep/buffers/ep.py`（`dispatch`/`combine`） | MoE EP 通信 |
| NCCL API | `third_party/nccl/src/collectives.cc`（`ncclAllReduce`）、`src/group.cc`（`ncclGroupStart`） | |
| enqueue / tuning | `third_party/nccl/src/enqueue/`、`src/tuning/` | |
| device kernel | `third_party/nccl/src/device/all_reduce.h`（`runRing`、`runTreeUpDown`）、`prims_ll.h`、`prims_ll128.h`、`prims_simple.h` | |
| proxy | `third_party/nccl/src/proxy.cc`（`ncclProxyProgress`） | |
| transport | `third_party/nccl/src/transport/p2p.cc`、`shm.cc`、`nvls.cc`、`net.cc`、`net_ib/`、`coll_net.cc` | |
| IB verbs | `third_party/rdma-core/libibverbs/examples/rc_pingpong.c` | Verbs 最小示例 |

- 待确认：PyTorch DDP / `ProcessGroupNCCL` 的具体文件与函数；本地 `third_party/` 未克隆 PyTorch；确认方式：克隆 pytorch 后查 `torch/csrc/distributed/c10d/`。
- DeepEP 的依赖方式（已核实）：本地 DeepEP@93eb6eb2 为 V2.5，`third_party/DeepEP/README.md` 写明已移除 V1 与 NVSHMEM 后端，改走 NCCL device API（GIN、`ncclCommWindowRegister`），见 `third_party/DeepEP/csrc/kernels/comm/context.cpp`；图中"NVSHMEM/IBGDA"对应的是 V1 及 Triton-distributed 的路径；V1 源码在 `third_party/DeepEP-hybrid-ep`，kernel 细节见 [09-ep-kernels](../09-ep-kernels/README.md)。详见 [05-framework-integration](../05-framework-integration/README.md) 与 [06-comm-compute-fusion](../06-comm-compute-fusion/README.md)。

## 本目录文件

| 文件 | 内容 |
|---|---|
| [00-start-here.md](00-start-here.md) | **学习入口**：推理并行策略 × 通信、EP dispatch/combine、主流 EP 方案与趋势、主线学习顺序 |
| [01-ai-infra-book-map.md](01-ai-infra-book-map.md) | ai-infra-book 中通信相关章节的阅读地图（章节 → 内容 → JD → notes 目录） |
| [02-jd-breakdown.md](02-jd-breakdown.md) | JD 职责与要求拆成知识点、实践与自评栏 |
| [03-learning-path.md](03-learning-path.md) | 10 周分阶段学习路线、产出物与验收方式 |

这几份文件和进度、题库之间的使用关系（依据 00-start-here 的"怎么使用这份入口"与 03-learning-path 的"路线总览"）：

```text
              ┌─────────────────┐     问题驱动 / 按周排时间   ┌────────────────────┐
              │ 00-start-here   │ ◄─────────────────────────► │ 03-learning-path   │
              └───┬─────────┬───┘                             └─────────┬──────────┘
    每学完一节    │         │ 面试前                                    │ 每阶段的书章节
                  ▼         ▼                                           ▼
      ┌─────────────────┐ ┌──────────────┐                    ┌──────────────────────┐
      │ 02-jd-breakdown │ │ 08-interview │                    │ 01-ai-infra-book-map │
      └────────┬────────┘ └──────────────┘                    └──────────────────────┘
               │ 整体进度与证据
               ▼
        docs/PROGRESS.md
```

## 全部章节

- [00-roadmap](README.md)：本页，总览
- [01-rdma](../01-rdma/README.md)：RDMA 与数据中心网络
- [02-scale-up](../02-scale-up/README.md)：机内与超节点互联
- [03-ccl-algorithms](../03-ccl-algorithms/README.md)：集合通信算法与代价模型
- [04-nccl-internals](../04-nccl-internals/README.md)：NCCL 内部实现
- [05-framework-integration](../05-framework-integration/README.md)：框架中的通信调用链
- [06-comm-compute-fusion](../06-comm-compute-fusion/README.md)：通信与计算融合
- [07-perf-troubleshooting](../07-perf-troubleshooting/README.md)：性能调优与故障排查
- [08-interview](../08-interview/README.md)：面试题与项目表述
- [09-ep-kernels](../09-ep-kernels/README.md)：EP 专用通信库（DeepEP LL、Hybrid-EP、NCCL EP）

## 外部资料与版本

外部源码只读，位于 `third_party/`，本轮笔记引用的 commit：

| 仓库 | commit |
|---|---|
| ai-infra-book | 3bdcb4fc |
| nccl | 3e46c089 |
| nccl-tests | afd59aba |
| perftest | 05200c2f |
| rdma-core | bd3282a1 |
| DeepEP | 93eb6eb2 |
| DeepEP-hybrid-ep | 10d4dd73 |
| Triton-distributed | 7908e4ea |
| vllm | 02653fa2 |
| sglang | d2a43644 |

- 版本权威来源是 `third_party/repos.lock`，说明见 `docs/references/third-party.md`；本表仅为方便查阅，与之不一致时以 lock 文件为准。
