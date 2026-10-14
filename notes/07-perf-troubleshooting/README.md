# 07 性能调优与故障排查（章节索引）

> 层次：实践 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3，含 `docs/userguide/source/troubleshooting/`）、third_party/nccl-tests@afd59ab、ai-infra-book@3bdcb4fc 第 6 章 6.4 节、第 7 章 7.6 节

## 一句话

性能和故障问题都先回到三件事：NCCL 实际选了什么（transport、通道、算法、协议、网卡、GDR）、硬件理论上能给多少（链路、拓扑）、两者差在哪一层；本章给出环境变量速查、性能分析方法和故障排查清单。

## 在调用链中的位置

- 依赖：NCCL 内部机制（[../04-nccl-internals/README.md](../04-nccl-internals/README.md)）、scale-up 传输（[../02-scale-up/README.md](../02-scale-up/README.md)）、RDMA 与 `net_ib`（[../01-rdma/README.md](../01-rdma/README.md)）、算法代价模型（[../03-ccl-algorithms/README.md](../03-ccl-algorithms/README.md)）。
- 工具：nccl-tests（[../04-nccl-internals/05-nccl-tests.md](../04-nccl-internals/05-nccl-tests.md)）、NCCL 自带的 `ncclras`、`ncclparam`、`NCCL_RUN_DIAGNOSTICS`。

## 本章笔记

| 文件 | 主题 |
|---|---|
| [01-env-vars.md](01-env-vars.md) | 从源码 grep 的关键环境变量与定义位置（调试、算法/协议/通道、P2P/SHM/NVLS、网络/IB/GDR、拓扑、RAS） |
| [02-perf-methodology.md](02-perf-methodology.md) | 基线带宽、busbw 对比理论值、小消息延迟、拓扑核查、`NCCL_DEBUG_SUBSYS`、profiler 插件、nsys/NVTX |
| [03-troubleshooting.md](03-troubleshooting.md) | hang、timeout、性能掉档、IB 链路、RAS 与内置诊断：症状 → 只读检查 → 可能原因 → 处理 |

## 速查：最先打开的三个开关

```shell
NCCL_DEBUG=INFO                                    # 版本、拓扑、通道、transport 选择
NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET,P2P,TUNING   # 按问题选子系统（默认 INIT,BOOTSTRAP,ENV）
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log               # 每进程一个文件（%h 主机名，%p pid）
```

## NCCL 自带工具（本地源码中可见）

| 工具 | 位置 | 用途 |
|---|---|---|
| `ncclras` | `src/ras/client.cc`，`src/CMakeLists.txt` 构建目标 | 连接运行中作业的 RAS（默认 `localhost:28028`），查看各 rank/communicator 状态，`-D` 运行 RAS 诊断，`-m` 监控模式 |
| `ncclparam` | `src/param/ncclparam.cc` | 列出 NCCL 参数，`-l` 显示值与说明，`-a` 包括私有参数 |
| `NCCL_RUN_DIAGNOSTICS=1` | `src/diagnostics.cc:31` | init 时运行主动诊断（P2P 访问、IB 带宽等），输出前缀 `NCCL DIAG` |
| `NCCL_PROXY_DUMP_SIGNAL=<signo>` | `src/proxy.cc:922` | 发信号打印 proxy op 状态 |
| `NCCL_TOPO_DUMP_FILE` / `NCCL_GRAPH_DUMP_FILE` | `src/init.cc:1385`、`src/graph/search.cc:1435` | 导出拓扑与通道 |

## 面试自测

1. 问：拿到一个"NCCL 性能差"的问题，你的前三步是什么？
   答：（1）用 nccl-tests 在同样的 rank 布局下跑全尺寸曲线，拿到 busbw 与时间，和链路理论值对比，确定是带宽问题还是延迟问题、是节点内还是节点间（`NCCL_TESTS_SPLIT`）；（2）`NCCL_DEBUG=INFO` 看实际 transport（P2P/SHM/NET、GDRDMA）、通道数、ring/tree 图、网卡选择；（3）核对拓扑（`nvidia-smi topo -m`、NUMA、网卡与 GPU 的 PCIe 距离）和链路状态（NVLink、IB 端口速率），再决定是否用环境变量做对比实验。
2. 问：作业 hang 了，NCCL 有哪些内置手段？
   答：RAS（默认开启，`ncclras` 或 `echo verbose status | nc localhost 28028` 查看哪些 rank 缺失/卡住）、`NCCL_PROXY_DUMP_SIGNAL` 打印 proxy 状态、`NCCL_DEBUG_SUBSYS=COLL` 比对各 rank 调用序列、`NCCL_RUN_DIAGNOSTICS` 在下次启动时验证链路；再结合 py-spy/gdb 看主机栈（详见 [03-troubleshooting.md](03-troubleshooting.md)）。

## 待深入

- `docs/userguide/source/troubleshooting/performance_and_tuning.rst` 中 RoCE 与 CPU 亲和（Slurm/Open MPI）章节的具体建议。
- profiler 插件示例 `plugins/profiler/example/` 的事件模型与 `src/include/plugin/nccl_profiler.h` 的版本差异。
