# 04 NCCL 内部实现（章节索引）

> 层次：中层 · 来源：third_party/nccl@3e46c08（`makefiles/version.mk` 显示 NCCL 2.32.3）、third_party/nccl-tests@afd59ab、ai-infra-book@3bdcb4fc 第 6 章 6.4 节

## 一句话

把 NCCL 拆成"建通信组 → 识别拓扑 → 搜索通道 → 建连接 → 入队发射 kernel → GPU primitives 搬数据 / proxy 线程驱动网卡"这几段，每段对应一篇源码导读。

下图是这几段与本章笔记编号的对应：左列是 init 时做一次的事，右列是每次集合调用都走的路径；左下"建连接"产出的 P2P/SHM/NET 连接供右侧 kernel 和 proxy 使用（默认懒连接时，这一步推迟到首次使用，见 01）。

```text
                        frameworks / nccl-tests (05)
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
    ncclCommInitRank (once)                   ncclAllReduce ... (per call)
  ┌────────────────────────────┐           ┌────────────────────────────────────┐
  │ 01 bootstrap ring,         │           │ 03 enqueue / group,                │
  │    AllGather peer info     │           │    tuning, kernel plan             │
  ├────────────────────────────┤           └───────┬─────────────────┬──────────┘
  │ 02 topo XML, paths,        │                   ▼                 ▼
  │    ring/tree/NVLS search   │           ┌───────────────┐  ┌─────────────────┐
  ├────────────────────────────┤           │ 03 GPU kernel │  │ 04 proxy        │
  │ 01 connect transports      │──────────►│    primitives │◄►│ ncclNet_t → NIC │
  │    P2P / SHM / NET         │           └───────────────┘  └─────────────────┘
  └────────────────────────────┘
```

## 在调用链中的位置

- 上层：PyTorch ProcessGroupNCCL、vLLM/SGLang 的通信封装调用 `ncclCommInitRank`、`ncclAllReduce`、`ncclGroupStart/End`（见 05-framework-integration，另行编写）。
- 本章：NCCL 主机侧（`src/init.cc`、`src/graph/`、`src/enqueue/`、`src/group.cc`、`src/proxy.cc`）和设备侧（`src/device/`）。
- 下层：scale-up 传输 P2P/SHM/NVLS 见 [../02-scale-up/README.md](../02-scale-up/README.md)；scale-out 的 IB verbs 与 `net_ib` 见 [../01-rdma/README.md](../01-rdma/README.md)。
- 算法本身（Ring/Tree 的步骤与 α-β 代价）见 [../03-ccl-algorithms/README.md](../03-ccl-algorithms/README.md)，本章只讲它们在 NCCL 里落到哪些文件。

## 本章笔记

| 文件 | 主题 | 核心源码 |
|---|---|---|
| [01-init-bootstrap.md](01-init-bootstrap.md) | `ncclGetUniqueId` / `ncclCommInitRank`、bootstrap 环、`initTransportsRank` 全流程 | `src/init.cc`、`src/bootstrap.cc`、`src/misc/socket.cc` |
| [02-topology-graph.md](02-topology-graph.md) | 拓扑 XML、路径类型、ring/tree/NVLS 通道搜索 | `src/graph/*.cc`、`src/include/graph.h` |
| [03-channels-protocols-kernels.md](03-channels-protocols-kernels.md) | channel、LL/LL128/Simple、primitives、enqueue、kernel launch、group 语义 | `src/device/*.h`、`src/enqueue/enqueue.cc`、`src/group.cc` |
| [04-proxy-and-net-plugin.md](04-proxy-and-net-plugin.md) | proxy 线程、net plugin 接口、与 IB transport 的关系 | `src/proxy.cc`、`src/transport/net.cc`、`src/plugin/net.cc`、`src/include/plugin/net/net_v12.h` |
| [05-nccl-tests.md](05-nccl-tests.md) | nccl-tests 结构、`all_reduce_perf` 参数、algbw/busbw | `third_party/nccl-tests/src/common.cu`、`src/all_reduce.cu`、`doc/PERFORMANCE.md` |

## 建议阅读顺序

1. 先读 01，建立"一次 init 做了哪几次 allgather、在哪一步建拓扑和连接"的骨架。
2. 读 02，理解 `NCCL_DEBUG=INFO` 日志里 `Pattern ... type NVL/PIX` 和 `Channel 00/16 : 0 1 2 ...` 的来源。
3. 读 03，把一个 `ncclAllReduce` 从 API 跟到 `ncclDevKernel_*` 和 `runRing` 里的 `directRecvReduceDirectSend`。
4. 读 04，理解跨节点数据为什么需要 CPU 线程，以及外部网络插件从哪里接入。
5. 读 05，跑基线并会解读 busbw。之后去 [../07-perf-troubleshooting/README.md](../07-perf-troubleshooting/README.md)。

## 版本差异提醒

本地 NCCL 是 2.32.3，与网上大量 2.1x 版本的源码解读相比，目录布局变化较大：

- `src/enqueue.cc` 已移到 `src/enqueue/enqueue.cc`，并新增 `src/enqueue/task_prep/`、`task_sched/`。
- 算法/协议选择移到 `src/tuning/`（`tuning.cc`、`cost_model.cc`、`ring.cc`、`tree.cc`、`nvls.cc`、`pat.cc`）。
- 网络插件接口在 `src/include/plugin/net/net_v6.h` ~ `net_v12.h`，参考实现在 `plugins/net/example/`；本地没有 `ext-net/` 目录。
- 环境变量有两种定义方式：老的 `NCCL_PARAM(...)` 宏（`src/include/param.h`）和新的 `DEFINE_NCCL_PARAM(...)`（`src/include/param/param.h`，目前用于 `NCCL_DEBUG*` 等）。
- 新增算法 `PAT`、对称内存 kernel（`src/sym_kernels.cc`、`src/device/symmetric/`）、copy engine 集合（`src/ce_coll.cc`）、RAS 子系统（`src/ras/`）、内置诊断（`src/diagnostics/`）。

## 面试自测

1. 问：一次 `ncclAllReduce` 从调用到数据到达对端，经过 NCCL 的哪些模块？
   答：`collectives.cc` 打包 `ncclInfo` → `enqueue.cc::ncclEnqueueCheck` 做参数检查并 `taskAppend` → `group.cc::ncclGroupEndInternal` 在 group depth 归零时触发 → tuning 选算法/协议/通道 → 生成 kernel plan 和 proxy op → `ncclLaunchKernel` 用 `cuLaunchKernelEx` 发射（grid = 通道数）→ 设备侧 `ncclKernelMain` → `RunWorkColl::run` → primitives 通过 P2P/SHM 缓冲或 net 缓冲收发；跨节点部分由 proxy progress 线程调用 `ncclNet->isend/irecv/test`。
2. 问：NCCL 为什么要有自己的拓扑探测，而不是直接用 MPI 的进程布局？
   答：带宽和延迟取决于 GPU、NVSwitch、PCIe 交换芯片、CPU 插槽、网卡之间的物理路径；NCCL 要据此决定 P2P 是否可用、GDR 是否可用、每个通道经过哪块网卡、ring 怎么排，这些信息只有 PCI/NVML/网卡层面的探测才能给出。

## 待深入

- 待确认：`ENQUEUE_REARCH_ENABLE`（`src/enqueue/enqueue.cc:33`，默认 0）开启后的新入队架构与 legacy 路径的完整差异；确认方式：对比 `src/group.cc` 中 `groupLaunchEnqueueRearch` 与 `groupLaunchLegacy`。
- PAT 算法与对称内存 kernel 的设备侧实现：`src/device/symmetric/*.cuh`、`src/tuning/pat.cc`。
