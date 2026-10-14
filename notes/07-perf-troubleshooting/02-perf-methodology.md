# NCCL 性能分析方法

> 层次：实践 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3，含 `docs/userguide/source/troubleshooting/performance_and_tuning.rst`、`logging.rst`）、third_party/nccl-tests@afd59ab（`doc/PERFORMANCE.md`）、ai-infra-book@3bdcb4fc 第 6 章 6.4.3 节、第 7 章 7.6 节

## 一句话

先用 nccl-tests 建立"可复现的基线曲线"，用 busbw 对比链路理论值判断带宽问题、用小尺寸 time 判断延迟问题；再用 NCCL 日志确认实际路径（transport、通道、算法/协议、网卡、GDR），用拓扑与链路检查解释差距；最后放回真实负载里用 nsys/NVTX/profiler 插件看通信与计算的重叠和就绪偏差。

## 在调用链中的位置

- 工具与依据：[../04-nccl-internals/05-nccl-tests.md](../04-nccl-internals/05-nccl-tests.md)（busbw 定义）、[01-env-vars.md](01-env-vars.md)（开关）、[../04-nccl-internals/02-topology-graph.md](../04-nccl-internals/02-topology-graph.md)（日志含义）。
- 代价模型背景：α-β 模型与 Ring/Tree 步数见 [../03-ccl-algorithms/README.md](../03-ccl-algorithms/README.md)。
- 出现异常（hang/报错/数据错）转到 [03-troubleshooting.md](03-troubleshooting.md)。

## 核心概念

### 1. 两类指标

| 区间 | 看什么 | 主要决定因素 |
|---|---|---|
| 小消息（B ~ 几十 KB） | `time`（us） | kernel launch、CPU 提交、LL 协议轮询、算法步数（ring 2(n-1) 步 vs tree log 级）、网络单程延迟、proxy 处理 |
| 中等消息 | 曲线拐点 | 协议/算法切换点（LL → LL128 → Simple；Tree/NVLS → Ring） |
| 大消息（几十 MB 以上） | `busbw`（GB/s） | 链路带宽、通道数、是否 GDR、NVLS/PXN、PCIe/NUMA 路径 |

`doc/PERFORMANCE.md`：小尺寸下时间反映常数开销，大尺寸下时间 ≈ 开销 + size/bw，此时看带宽更有意义。

### 2. 理论值从哪里来

- 节点内：每 GPU 的 NVLink 单方向带宽（ai-infra-book 第 7 章 H100 例为 450 GB/s，NVLink 规格表见 [../02-scale-up/01-nvlink-nvswitch.md](../02-scale-up/01-nvlink-nvswitch.md)）。
- 节点间：每 GPU 对应网卡带宽（书中 400 Gbit/s ≈ 每方向 50 GB/s）× 该 GPU 可用网卡数。
- 中间段：PCIe（书中 Gen5 x16 每方向 64 GB/s，Gen4 32 GB/s），以及 GPU-NIC 是否在同一 PCIe 交换芯片下（PIX/PXB）还是要过 CPU（PHB/SYS）。
- busbw 是归一化后的"每 rank 链路带宽"，应和单 rank 的瓶颈链路比较；书中提醒它不等于每张网卡的实际吞吐，分层通信、交换机内归约（NVLS/SHARP）会改变实际流量，所以 NVLS 下 busbw 可能高于单 GPU NVLink 单向带宽的直觉值。

### 3. 就绪偏差与重叠

- ai-infra-book 7.6.1："关键路径从数据就绪开始"：一张卡早进入集合通信调用、另一张卡梯度还没算完，等待会被记到通信耗时里，根源却在前面的计算或主机操作（书中引用 MegaScale 的观察）。在 profile 中看到某 rank 的 AllReduce 特别长，先看其他 rank 是否晚到。
- 6.4.3：通信与计算重叠时互相争用 SM、HBM 和互联，"只看单独运行的通信带宽来选择配置，可能让整体变慢"；评价重叠要比较同时运行的时间，不能把各自单独测得的时间相减。

## 源码导读：分析时依赖的日志与钩子

| 想知道 | 日志/钩子 | 源码位置 |
|---|---|---|
| 版本、CUDA 驱动 | `NCCL version ...`（`showVersion`） | `src/init.cc:861` |
| 每阶段 init 耗时 | `Init timings - ... total ... (kernels, alloc, bootstrap, allgathers, topo, graphs, connect ...)` | `src/init.cc::ncclCommInitRankFunc` |
| 拓扑与路径 | `NCCL_DEBUG_SUBSYS=GRAPH`：`ncclTopoPrint`、`ncclTopoPrintPaths`、`Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d` | `src/graph/topo.cc:347`、`paths.cc:189`、`search.cc:1399` |
| 通道 ring 顺序 | `Channel %02d/%02d : r0 r1 ...` | `src/graph/rings.cc:48` |
| 通道总数 | `%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer` | `src/init.cc:1901` |
| transport | `Channel xx/y : a[x] -> b[y] via P2P/CUMEM`、`via P2P/IPC`、`via SHM`、`[send] via NET/IB/0/GDRDMA`、`NET/IB/0(proxyRank)`（PXN） | `src/transport/p2p.cc::p2pSendSetup`、`net.cc::sendSetup` |
| NVLS 是否可用 | `NVLS multicast support is ... available` | `src/transport/nvls.cc:197` |
| 每次调用参数 | `NCCL_DEBUG_SUBSYS=COLL`：`%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p` | `src/enqueue/enqueue.cc::ncclEnqueueCheck` |
| 算法/协议选择 | `NCCL_DEBUG_SUBSYS=TUNING`；nccl-tests `-U 1` | `src/tuning/` |
| NVTX 区间 | `NVTX3_FUNC_WITH_PARAMS(AllReduce, ...)` 等，带 commHash、字节数、op；`NCCL_NVTX_DISABLE=1` 关闭 | `src/collectives.cc`、`src/init_nvtx.cc` |
| profiler 插件 | `NCCL_PROFILER_PLUGIN`；事件类型见 `ncclras` 帮助：`group,coll,p2p,proxyop,proxystep,proxyctrl,kernelch,netplugin,groupapi,collapi,p2papi,kernellaunch,cecoll,cesync,cebatch` | `src/plugin/profiler.cc:109`、`src/ras/client.cc:printUsage`、`plugins/profiler/example/` |
| 运行中开关 profiler | `ncclras CONTROL PROFILER_MASK <value>` | `src/ras/client.cc` |

## 方法步骤

### Step 1 固定环境，跑基线

1. 记录：NCCL/驱动/CUDA 版本、GPU 型号、网卡型号与固件、rank 布局（每节点几 rank、每 rank 绑哪块 GPU 和 CPU 核）。
2. 单机：`./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8`（或 `mpirun -np 8` + `-g 1`，更接近框架的每进程一卡）。
3. 多机：`mpirun -np N -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1`；再用 `NCCL_TESTS_SPLIT="MOD 8"` 单测跨节点、`"DIV 8"` 单测节点内（[05-nccl-tests.md](../04-nccl-internals/05-nccl-tests.md)）。
4. 除 AllReduce 外至少再跑 AllGather、ReduceScatter、AlltoAll、SendRecv（MoE/PP 场景）。
5. 加大 `-w` 消除懒连接影响；`-I 1` 看迭代抖动；`-a 2/3` 找慢 rank。

### Step 2 对比理论值，定位哪一段

- 大消息 busbw 明显低于节点内 NVLink 或节点间网卡的理论值 → 带宽问题：查 transport（是否误走 SHM/Socket/无 GDR）、通道数、网卡选择、PCIe 路径。
- 小消息 time 明显偏高 → 延迟问题：查算法（Tree/NVLS/PAT 是否可用）、协议（LL/LL128 是否被禁用）、CPU 绑核、proxy 线程是否被抢、是否走了 host 内存。
- 单机正常、跨机差 → 网络侧：IB 端口速率/错误计数、GDR、PXN、交换网络（ECMP、拥塞、PFC），见 [03-troubleshooting.md](03-troubleshooting.md) 与 [../01-rdma/README.md](../01-rdma/README.md)。
- 少数节点拖慢整体 → `-a 2/3`、逐对节点 sendrecv/all_reduce 二分定位。

### Step 3 拓扑核查

- `nvidia-smi topo -m`：看 GPU-GPU 是 NV#（NVLink 条数）还是 PIX/PXB/PHB/SYS/NODE，GPU-NIC 的距离，以及 CPU/NUMA 亲和列。（本地无 GPU，未实测输出。）
- 对照 NCCL 自己的判定：`NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` 导出 XML，`NCCL_DEBUG_SUBSYS=GRAPH` 打印路径；两者不一致（例如 nvidia-smi 显示 PIX 而 NCCL 判成 SYS）多见于虚拟化/容器 sysfs 不完整，用 `NCCL_TOPO_FILE` 修正。
- NUMA：确认进程绑核与 GPU 同 NUMA。userguide `performance_and_tuning.rst` 的 "How NCCL handles affinity" 节：NCCL 默认取"启动器继承的 CPU 亲和"与"GPU 的 CPU 亲和"的交集，交集为空时保持继承值；`NCCL_IGNORE_CPU_AFFINITY=1` 只用 GPU 亲和；cgroup/cpuset 排除的 CPU 仍不可用。检查日志 `ncclTopoGetCpuAffinity: Affinity for GPU ...`，并用 `nvidia-smi topo -m`、`lscpu --extended=CPU,NODE,SOCKET,CORE` 核对（文档给出的命令）。
- 网卡：每 GPU 是否有同 PCIe 交换芯片下的网卡（rail 对齐）；多端口网卡是否被合并为虚拟设备（`NCCL_IB_MERGE_NICS`）。

### Step 4 用日志确认 NCCL 实际选择

推荐组合：

```shell
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,P2P,SHM,NET,TUNING NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log
```

逐项核对：通道数是否符合预期、每个通道走的 transport、网卡与 GDRDMA、NVLS 是否启用、ring 是否跨 rail（`crossNic`）、tuning 选择。

### Step 5 受控 A/B 实验（只用于定位）

| 假设 | 实验 |
|---|---|
| 协议选择不佳 | `NCCL_PROTO=Simple` / `^LL128` |
| 算法选择不佳 | `NCCL_ALGO=Ring` / `Tree` / `^NVLS` |
| 通道不足或过多 | `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`、`NCCL_MIN_CTAS`/`MAX_CTAS` |
| NVLS 影响 | `NCCL_NVLS_ENABLE=0` |
| PXN 影响 | `NCCL_PXN_DISABLE=1` |
| GDR 影响 | `NCCL_NET_GDR_LEVEL=SYS`（放宽）或 `LOC`（禁用）对比 |
| IB 多 QP | `NCCL_IB_QPS_PER_CONNECTION=2/4` |
| 拓扑误判 | `NCCL_TOPO_FILE=<修正后的 xml>` |

每次只改一个变量，并用全尺寸曲线而不是单点比较。结论如果是"某个强制设置更快"，应反馈为 tuning 模型或拓扑问题（或用 tuner 插件），而不是直接写死。

### Step 6 回到真实负载

- nsys：采集 CUDA kernel 与 NVTX，看 `ncclDevKernel_*` 与计算 kernel 的重叠、各 rank 进入集合通信的时间差（就绪偏差）。命令形如 `nsys profile -t cuda,nvtx ...`（参数以本机 `nsys --help` 为准，未在本地验证）。
- profiler 插件：`NCCL_PROFILER_PLUGIN` 加载 `plugins/profiler/example/` 编译出的插件，可得到 coll/proxyOp/proxyStep/kernel channel 级事件，定位是 GPU 在等还是 proxy/网络在等。
- 框架侧：通信是否分桶、是否与 backward 重叠（参考 ai-infra-book 7.6.1 的分桶讨论；框架实现见 05-framework-integration，另行编写）。
- 与计算争用：调小 `NCCL_MAX_CTAS`/通道数或使用 CE/NVLS 路径以节省 SM，评估整体 step time 而不是只看通信带宽（6.4.3）。

## 性能要点与常见坑

- 只跑一个尺寸（nccl-tests 默认 32M）得不出结论。
- 用 algbw 对比链路带宽会低估 AllReduce（应乘 2(n-1)/n 看 busbw）。
- 首次迭代包含懒连接与注册开销；warmup 不足会让小规模测试看起来"很慢"。
- `NCCL_DEBUG=INFO` 本身在高频小消息下有开销（COLL 子系统每次调用打印一行），性能测量时关闭或只开 INIT。
- 多 comm 并发（如 TP 与 DP 同时通信）时单 comm 基线不能直接外推，需要用 `NCCL_TESTS_SPLIT` 或多 comm 测试模拟。
- 不同 rank 进程绑核不一致，会导致 proxy 线程跨 NUMA，跨机带宽波动。

## 面试自测

1. 问：8 节点 64 卡 H100，AllReduce 8GB busbw 只有预期一半，怎么分析？
   答：先用 `NCCL_TESTS_SPLIT` 分别测节点内和节点间，确定瓶颈侧；节点间侧看日志里每通道是否 `GDRDMA`、每 GPU 是否用了本地网卡（PXN/crossNic）、通道数与网卡数是否匹配、IB 端口速率与错误计数；节点内侧看是否 P2P/NVLS；再用 `-a 2/3` 和逐对测试找慢节点；最后 A/B `NCCL_IB_QPS_PER_CONNECTION`、`NCCL_ALGO` 验证假设。
2. 问：小消息 AllReduce 延迟高，有哪些可能？
   答：算法只能用 Ring（Tree/NVLS 不可用）导致步数多；LL/LL128 被禁用或不适用（PCIe 路径不开 LL128）；CPU launch 开销（可用 CUDA graph 对比 `-G`）；proxy 线程被抢占或跨 NUMA；网络单程延迟高（RoCE 拥塞、交换机层数）；就绪偏差被算进通信时间。
3. 问：如何判断通信与计算重叠是否有效？
   答：在 nsys 时间线上看通信 kernel 是否与计算 kernel 并行、step time 是否接近 max(计算, 通信) 而不是两者之和；对比"同时运行"时两者的实际耗时（书中 6.4.3 强调不能用单独运行时间相减）；必要时减少通信 CTA 或改用 NVLS/CE 降低 SM 占用。
4. 问：busbw 超过单卡 NVLink 单向带宽可能吗？
   答：在 NVLS 等交换机内归约下可能：busbw 按 2(n-1)/n 的点对点模型换算，而 NVLS 实际每 GPU 链路流量更少；所以要按实际算法的物理流量解释，不能机械比较。
5. 问：你会怎样向用户证明"不是 NCCL 的问题"？
   答：给出同一环境下 nccl-tests 的基线曲线与理论值对比、`NCCL_DEBUG` 证明路径正确；用 nsys 证明通信 kernel 时间正常、等待来自某 rank 晚到（就绪偏差）或计算争用；用 RAS/诊断证明链路健康。

## 待深入

- userguide "RoCE considerations" 只给出一条：RoCE 上可能需要每连接多个 QP 才能打满网络，设置 `NCCL_IB_QPS_PER_CONNECTION` > 1。TC/GID/PFC 等配置需结合 [../01-rdma/README.md](../01-rdma/README.md) 与交换机侧设置。
- profiler 插件事件结构（`src/include/plugin/nccl_profiler.h`、`plugins/profiler/example/event.h`）与如何从 proxyStep 事件计算网络单程时间。
- `NCCL_IB_WQE_LATENCY_THRESHOLD_NS`（`src/transport/net_ib/wqe_lat_mon.cc`）用于在线发现慢 WQE。
