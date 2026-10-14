# NCCL 初始化与 bootstrap

> 层次：中层 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）、ai-infra-book@3bdcb4fc 第 6 章 6.4.3 节

## 一句话

`ncclGetUniqueId` 在 rank 0 开一个 TCP 监听（bootstrap root），所有 rank 带着这个 id 调 `ncclCommInitRank`，先经 root 拼成一个 TCP 环（bootstrap 网络），再用这个环做几轮 AllGather 交换 peer 信息和拓扑结果，最后建立真正的数据通道（P2P/SHM/NET）。

## 在调用链中的位置

- 上层：框架（如 PyTorch）在 rank 0 调 `ncclGetUniqueId`，通过自己的 store/MPI 广播 128 字节的 `ncclUniqueId`，然后每个 rank 调 `ncclCommInitRank` 或 `ncclCommInitRankConfig`。
- 本模块：`src/init.cc`（API 与 `initTransportsRank` 编排）、`src/bootstrap.cc`（OOB 网络）、`src/misc/socket.cc`（网卡选择、socket 封装）。
- 下层：拓扑与搜索（[02-topology-graph.md](02-topology-graph.md)）、transport 连接（[../02-scale-up/02-nccl-p2p-shm-nvls.md](../02-scale-up/02-nccl-p2p-shm-nvls.md)、[04-proxy-and-net-plugin.md](04-proxy-and-net-plugin.md)）。

## 核心概念

| 概念 | 含义 |
|---|---|
| `ncclUniqueId` | 128 字节不透明 id；内部是 `struct ncclBootstrapHandle`（root 监听地址 `addr` + 随机 `magic`），见 `ncclGetUniqueId` 中的 `memcpy(out, &handle, sizeof(handle))` |
| bootstrap root | `bootstrapCreateRoot` 起的一个分离线程 `bootstrapRoot`（线程名 `NCCL BootstrapR`），收集所有 rank 的监听地址，再把"下一个 rank 的地址"发回去 |
| bootstrap 环 | 每个 rank 只连自己的 next，形成 TCP 环；`bootstrapAllGather` 用 ring allgather 在这个环上交换数据 |
| OOB 网卡 | `bootstrapNetInit` 选的那块 IP 网卡，受 `NCCL_SOCKET_IFNAME`、`NCCL_SOCKET_FAMILY`、`NCCL_COMM_ID` 影响 |
| 异步 init | `ncclCommInitRankDev` 只做参数检查和分配，真正工作 `ncclCommInitRankFunc` 作为 async job 执行；非阻塞模式下返回 `ncclInProgress` |

## 源码导读

### 1. 全局一次性初始化

- `src/init.cc::ncclInitEnv` → `ncclEnvPluginInit`：加载环境变量插件（env plugin）。
- `src/init.cc::ncclInit` → `initOnceFunc`：`ncclOsInitialize`、`initGdrCopy`、`bootstrapNetInit`。注释写着 "Always initialize bootstrap network"。
- `src/misc/param.cc::initEnvFunc`：先读 `NCCL_CONF_FILE` 指定的文件，否则读 `~/.nccl.conf`，再读 `/etc/nccl.conf`，把 `KEY=VALUE` 写入环境变量。排查"明明没设却生效"的参数时要看这两个文件。

### 2. `ncclGetUniqueId`

`src/init.cc::ncclGetUniqueId` → `src/bootstrap.cc::bootstrapGetUniqueId`：

- 若设置 `NCCL_COMM_ID`（格式 `<ipv4>:<port>`、`[<ipv6>]:<port>` 或 `<hostname>:<port>`），handle 直接用这个地址，magic 固定为 `NCCL_MAGIC`，root 由 rank 0 在 `ncclCommInitRankDev` 里起。
- 否则 `getRandomData` 生成 magic，地址取 `bootstrapNetIfAddr`，并立刻 `bootstrapCreateRoot` 起 root 线程（所以调 `ncclGetUniqueId` 的进程必须活到所有 rank 连上 root）。

### 3. bootstrap 网卡选择

`src/bootstrap.cc::bootstrapNetInit` → `src/misc/socket.cc::ncclFindInterfaces`，自动选择顺序（未设 `NCCL_SOCKET_IFNAME` 时）：

1. 名字以 `ib` 开头的接口（IPoIB）；
2. 与 `NCCL_COMM_ID` 同子网的接口；
3. 除 `docker`、`lo`、`virbr` 外的任意接口；
4. 依次退回 `docker`、`lo`、`virbr`。

`NCCL_SOCKET_IFNAME` 的值交给 `src/os/linux.cc::ncclOsFindInterfaces`：默认按前缀匹配，开头 `^` 表示排除（`searchNot`），开头 `=` 表示精确匹配（`searchExact`），列表由 `parseStringList` 解析（逗号分隔）。`NCCL_IB_HCA` 在 `src/transport/net_ib/init.cc` 中用同样的 `^`/`=` 约定。日志关键字：`Bootstrap: Using <ifname>:<ip>`（`NCCL_BOOTSTRAP` 子系统）。

### 4. `ncclCommInitRank` → 异步 job

`src/init.cc`：

- `ncclCommInitRank`：`ncclInitEnv` → `ncclCudaLibraryInit` → `ncclGroupStartInternal` → `ncclCommInitRankDev` → `ncclGroupEndInternal`。也就是说 init 本身是包在 group 里的，这是可以把多个 init 放进用户 `ncclGroupStart/End` 一起完成的原因。
- `ncclCommInitRankDev`：`ncclInit`、`cudaFree(NULL)` 强制创建 CUDA context、分配 `comm`、`abortFlag`、`parseCommConfig`，设置 `comm->initState = ncclInProgress`；把 `ncclCommInitRankFunc` 交给 `ncclAsyncLaunch`（或 `ENQUEUE_REARCH_ENABLE` 下的 `ncclMgmtTaskEnqueue`）。
- `ncclCommInitRankFunc`：`ncclInitKernelsForDevice` → `commAlloc` → `bootstrapInit`（或 split/shrink 时 `bootstrapSplit`）→ `initTransportsRank` → `ncclProgressCounterMonitorInit` → `initState = ncclSuccess`，并打印 `Init COMPLETE` 与 `Init timings`（kernels/alloc/bootstrap/allgathers/topo/graphs/connect 各阶段耗时）。

### 5. `bootstrapInit`：拼环

`src/bootstrap.cc::bootstrapInit` 主要步骤（按源码注释）：

1. 为环上邻居创建监听：默认 socket（`createListenSocket`）；若 `NCCL_OOB_NET_ENABLE=1`（`NCCL_PARAM(BootstrapNetEnable, "OOB_NET_ENABLE", 0)`），改用 `state->net->listen`，即用数据网络插件做 bootstrap。
2. 为 root 回连创建监听 `listenSockRoot`。
3. 大规模时错峰连接 root：`UID_STAGGER_RATE`、`UID_STAGGER_THRESHOLD`。
4. `sendToRoot` 把自己的地址发给 root；root 线程 `bootstrapRoot` 收齐后把每个 rank 的 next 地址回发。
5. `netRingConnect` / `socketRingConnect` 连上 next，环成形。
6. 创建 proxy 监听 socket、UDS（用于 cuMem fd 传递）、P2P socket，并初始化 RAS（`ncclRasAddRanks`）。
7. `ringAllInfo` 在环上交换所有 rank 的 proxy/P2P 地址。
8. `ncclProxyInit` 建 proxy 服务的基础状态。

`bootstrapAllGather` 根据 `ncclParamBootstrapNetEnable()` 选 `netRingAllGather` 或 `socketRingAllGather`；另有 `bootstrapSend/Recv`（点对点，带 tag 和 unexpected 队列）、`bootstrapBarrier`、`bootstrapIntraNodeAllGather`、`bootstrapIntraNodeBarrier`。

### 6. `initTransportsRank`：init 的主体

`src/init.cc::initTransportsRank`（约 1163 行起），按源码注释顺序：

| 阶段 | 关键调用 | 作用 |
|---|---|---|
| AllGather1 | `fillInfo` + `bootstrapAllGather(peerInfo)` | 交换 `ncclPeerInfo`：hostHash、busId、nvmlDev、gdrSupport、fabric 信息等；据此算 `nNodes`、intra-process ranks |
| MNNVL | `ncclMnnvlCheck` | 判断是否跨节点 NVLink（多节点 NVLink 域） |
| 拓扑 | `ncclTopoGetSystem` → `ncclTopoComputePaths` → `ncclTopoTrimSystem` → 再 `ComputePaths` → `ncclTopoSearchInit` → `ncclTopoPrint` | 建拓扑图、算路径（[02-topology-graph.md](02-topology-graph.md)） |
| CPU 亲和 | `ncclTopoGetCpuAffinity` + `ncclOsSetAffinity` | 绑定到离 GPU 近的 CPU，使后续 host 内存分配落在本地 NUMA |
| 图搜索 | `ncclTopoCompute(ringGraph/treeGraph/collNet*/nvlsGraph)` + `ncclTopoPrintGraph` | 搜索 ring、tree、CollNet、NVLS 通道 |
| AllGather3 | `ncclTopoPreset` + `bootstrapAllGather(allGather3Data)` | 交换每个 rank 的 graph 信息、topoRanks、网卡数，检查网卡数一致（`NCCL_IGNORE_NET_MISMATCH`） |
| 对齐与连接图 | `ncclTopoPostset` | 跨节点把各节点的通道拼成全局 ring/tree，取各 rank 的最小通道数保证一致 |
| buffer 与 p2p 通道 | `computeBuffSizes`、`ncclTopoComputeP2pChannels` | 确定 LL/LL128/Simple 缓冲大小（`NCCL_BUFFSIZE` 等）与 p2p 通道数 |
| proxy | `ncclProxyCreate` | 起 proxy service 线程 |
| 连接 | `comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect()`；为真时只 `setupChannel` + `ncclNvlsSetup`，连接推迟到首次使用（`enqueue.cc` 中检查 `initAlgoChannels`，`group.cc::ncclCollPreconnect`）；为假时立即 `ncclTransportRingConnect` → `ncclTransportTreeConnect` → `ncclNvlsSetup/BufferSetup` → `ncclTransportPatConnect` → `ncclNvlsTreeConnect` → CollNet | 默认 `NCCL_RUNTIME_CONNECT=1`，且需要 cuMem 支持才走懒连接 |
| tuning | `ncclTuningInit` | 初始化代价模型、读取 `NCCL_ALGO/PROTO` |
| 设备侧 | `devCommSetup` | 把 channel、peer 连接信息拷到 GPU；注释说明必须在最后一个 barrier 前做，否则可能死锁 |
| 收尾 | `bootstrapIntraNodeBarrier` | 节点内同步，恢复 CPU 亲和 |

## 性能要点与常见坑

- init 卡住时先看日志停在哪一步：`Init START` 之后没有 `Bootstrap: Using` 说明 OOB 网卡选择出问题；有 bootstrap 但无 `Init COMPLETE` 多半是某个 rank 没到（rank 数、unique id 不一致、防火墙/端口不通）。
- 容器里常见问题：自动选到了 `docker0` 或不同节点选到不同子网的网卡。显式设置 `NCCL_SOCKET_IFNAME`。
- 大规模 init 慢：`Init timings` 会拆出 bootstrap、allgathers、topo、graphs、connect；root 连接风暴由 `UID_STAGGER_*` 缓解；`ncclCommInitRankScalable` 支持多个 unique id（多 root，`nId > 1`）。
- 默认 `NCCL_RUNTIME_CONNECT=1`（且 cuMem 可用）时，首次使用某种算法的集合调用才建连，所以首个 iteration 偏慢是正常现象，测性能要 warmup；想把建连开销挪回 init 可设 `NCCL_RUNTIME_CONNECT=0`。
- `NCCL_COMM_ID` 与多 unique id 互斥：源码 `"NCCL_COMM_ID cannot be used with more than one ncclUniqueId"`。

## 面试自测

1. 问：`ncclUniqueId` 里装了什么？为什么调用 `ncclGetUniqueId` 的进程不能马上退出？
   答：装的是 bootstrap root 的 socket 地址和一个随机 magic（`ncclBootstrapHandle`）。`bootstrapGetUniqueId` 在本进程里起了 root 线程，所有 rank 的 `bootstrapInit` 都要连它交换地址；进程退出 root 就没了。
2. 问：bootstrap 网络和数据网络是什么关系？
   答：bootstrap 默认走 TCP socket（OOB 网卡），只用于 init 期间的地址/拓扑交换、barrier，以及后续 RAS、proxy 控制消息；数据走 P2P/SHM/NET transport。`NCCL_OOB_NET_ENABLE=1` 时 bootstrap 环可改用 net 插件。
3. 问：`initTransportsRank` 里做了几次全局 AllGather，各交换什么？
   答：代码注释写 "We use 2 AllGathers"：第一次交换 `ncclPeerInfo`（host、busId、能力），第二次（代码里叫 AllGather3）交换各 rank 搜出来的 graph 信息和 `topoRanks`，用于 `ncclTopoPostset` 拼全局 ring/tree 并对齐通道数。拓扑 XML 还会在节点内做一次 `bootstrapIntraNodeAllGather` 融合。
4. 问：为什么 init 内部也包了 `ncclGroupStartInternal/EndInternal`？
   答：init 的真正工作是 async job，放在 group 里可以让一个线程在 `ncclGroupStart/End` 中同时初始化多个 comm（单进程多 GPU），并支持非阻塞 init（`config.blocking=0` 时返回 `ncclInProgress`，用 `ncclCommGetAsyncError` 轮询）。
5. 问：init 卡死，如何最快定位？
   答：`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV`（默认子系统就是 INIT|BOOTSTRAP|ENV），对比所有 rank 的最后一行；核对 nranks/rank/uniqueId 广播、网卡选择、端口连通；必要时 `ncclras` 看哪些 rank 未加入（见 [../07-perf-troubleshooting/03-troubleshooting.md](../07-perf-troubleshooting/03-troubleshooting.md)）。

## 待深入

- `ncclCommSplit`/`ncclCommShrink`/`ncclCommGrow` 的 `bootstrapSplit` 与 `bcastGrowHandle` 路径。
- 懒连接路径细节：`src/enqueue/enqueue.cc::ncclPrepareTasks` 中 `algoNeedConnect` 的计算，以及 `src/group.cc::ncclCollPreconnectFunc`。
- `devCommSetup` 中 `ncclDevComm` 的字段布局：`src/include/device.h` 中 `struct ncclDevComm`。
