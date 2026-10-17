# NCCL 的 scale-up transport：P2P、SHM、NVLS

> 层次：下层 scale-up（NCCL 实现） · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）、ai-infra-book@3bdcb4fc 第 6 章 6.4.4 节、第 7 章 7.2.1 节（在网归约）

## 一句话

同一 scale-up 域内，NCCL 优先用 P2P：把对端（或自己）分配的缓冲通过 CUDA IPC / cuMem 句柄映射进本 GPU 地址空间，kernel 直接读写；P2P 不可用时退到 SHM（经主机共享内存中转）；在支持 multicast 的 NVSwitch 上，NVLS 把所有 GPU 的缓冲绑定到一个 multicast 对象，用 `multimem.ld_reduce`/`multimem.st` 让交换机完成归约和广播。

## 在调用链中的位置

- 上层：`src/transport.cc::selectTransport` 按 `ncclTransports[] = {&p2pTransport, &shmTransport, &netTransport, &collNetTransport}` 依次 `canConnect`；NVLS 在 init 中单独 `ncclNvlsInit/ncclNvlsSetup/ncclNvlsBufferSetup` 建立（[../04-nccl-internals/01-init-bootstrap.md](../04-nccl-internals/01-init-bootstrap.md)）。
- 本篇：`src/transport/p2p.cc`、`shm.cc`、`nvls.cc`、`multicast.cc`、`mc_arena.cc`、`nvls_ub.cc`，`src/mnnvl.cc`，`src/ce_coll.cc`，`src/misc/cudawrap.cc`。
- 下层：CUDA driver API（`cuMemCreate/cuMemExportToShareableHandle/cuMemImportFromShareableHandle/cuMemMap`、`cuMulticast*`）、legacy `cudaIpcGetMemHandle`；设备侧 PTX `multimem.*`。
- 跨节点部分见 [../04-nccl-internals/04-proxy-and-net-plugin.md](../04-nccl-internals/04-proxy-and-net-plugin.md) 与 [../01-rdma/README.md](../01-rdma/README.md)。

`selectTransport` 对每个 (channel, peer, send/recv) 连接按数组顺序试，第一个 `canConnect` 为真的 transport 立即调用自己的 `setup` 并返回；全部失败就是常见报错 `No transport found for rank X -> rank Y`（`src/transport.cc`）：

```text
  ncclTransports[] = { p2p, shm, net, collNet }

  ┌─────────┐  no  ┌─────────┐  no  ┌─────────┐  no  ┌─────────┐  no
  │ p2p     ├─────►│ shm     ├─────►│ net     ├─────►│ collNet ├─────► WARN "No transport found"
  └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘       return ncclSystemError
       │ yes            │ yes            │ yes            │ yes
       ▼                ▼                ▼                ▼
   p2p setup        shm setup        net setup       collNet setup

  NVLS is not in this array: built separately at init
  (ncclNvlsInit -> ncclNvlsSetup -> ncclNvlsBufferSetup)
```

## 核心概念

### CUDA IPC 与 cuMem

| 机制 | NCCL 中的入口 | 句柄如何传给对端 | 备注 |
|---|---|---|---|
| legacy CUDA IPC | `cudaIpcGetMemHandle`（`p2p.cc::ncclP2pAllocateShareableBuffer`） | 64B `cudaIpcMemHandle_t` 直接经 bootstrap/proxy 消息传 | 只能同 host；`p2pCanConnect` 会探测 legacy IPC 是否可用（注释 "WSL WAR"） |
| cuMem + POSIX fd | `ncclCuMemAlloc` + `cuMemExportToShareableHandle(..., CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)` | fd 不能直接传字节，需经 proxy 的 UDS 线程（`ncclProxyMsgGetFd/QueryFd`、`ncclProxyClientGetFdBlocking`）用 Unix socket 传 fd | `ncclCuMemHandleType` 默认值（`misc/cudawrap.cc:19`） |
| cuMem + FABRIC | `CU_MEM_HANDLE_TYPE_FABRIC` | fabric handle 是可直接传的字节 | MNNVL 跨节点 NVLink 域使用（`mnnvl.cc:89` 切换类型） |

`ncclCuMemEnable()`（`misc/cudawrap.cc:47`）：`NCCL_CUMEM_ENABLE` 默认 -2 表示自动检测。源码注释提醒 legacy IPC 检测"will always fail when using NCCL_CUMEM_ENABLE=1"。

三种句柄从"分配方"到"导入方"的路径（`p2p.cc::ncclP2pAllocateShareableBuffer` 与 `ncclP2pImportShareableBuffer`）。POSIX fd 不能当字节传，所以导入方要把原生 cuMem handle 发回分配方的 proxy，由它转成 fd 再经 Unix socket 送回：

```text
  owner (p2pSendProxySetup / p2pRecvProxySetup)      importer (p2pMap)
  ncclP2pAllocateShareableBuffer                     ncclP2pImportShareableBuffer
  │
  ├─ legacy IPC: ncclCudaCalloc
  │              cudaIpcGetMemHandle ───── 64B handle ─────► cudaIpcOpenMemHandle
  │
  ├─ cuMem FABRIC: ncclCuMemAlloc
  │                cuMemExportToShareableHandle ── bytes ──► cuMemImportFromShareableHandle
  │
  └─ cuMem POSIX fd: ncclCuMemAlloc
                     keep native handle ◄── handle ───────── ncclProxyClientGetFdBlocking
                     (proxy UDS thread) ─── fd via UDS ────► cuMemImportFromShareableHandle(fd)

  both cuMem paths, importer side:  cuMemAddressReserve ──► cuMemMap ──► cuMemSetAccess
  then: kernel load/store directly on the mapped peer buffer
```

### P2P 四种类型（`p2p.cc` 中 `enum p2pType`）

| 类型 | 条件 | 日志（`p2pSendSetup`） |
|---|---|---|
| `P2P_DIRECT` | 两 rank 在同一进程（`P2P_SAME_PID`）且未设 `NCCL_P2P_DIRECT_DISABLE`、未用 CE | `via P2P/direct pointer` |
| `P2P_CUMEM` | `ncclCuMemEnable()` | `via P2P/CUMEM` 或 `via P2P/MNNVL` |
| `P2P_IPC` | 否则 legacy IPC | `via P2P/IPC` |
| `P2P_INTERMEDIATE` | 需要经中间 GPU（`intermediateRank != -1`，对应 NVB 路径） | `via P2P/indirect/<rank>` |

后缀 `/read` 表示 P2P read 模式（接收方从发送方显存读，`NCCL_P2P_READ`），否则 write 模式（`NCCL_P2P_WRITE`）；`/CE` 表示用 copy engine（`NCCL_P2P_USE_CUDA_MEMCPY=1`，由 proxy 线程 `p2pSendProxyProgress` 调 `cudaMemcpyAsync`，不占 SM）。

两种模式下缓冲放在哪一侧（`p2pSendSetup` 的 `sendSize` 与 `p2pRecvSetup` 的 `recvSize`）：read 模式只把 SIMPLE 缓冲挪到发送方 `ncclSendMem` 之后，LL/LL128 仍在接收方。

```text
  write mode (NCCL_P2P_WRITE)
     sender GPU                          receiver GPU
     ┌───────────────┐                   ┌───────────────┐
     │ ncclSendMem   │                   │ ncclRecvMem   │
     └───────────────┘                   │ LL buffer     │
       sender kernel ────── store ─────► │ LL128 buffer  │
                                         │ SIMPLE buffer │
                                         └───────────────┘

  read mode (NCCL_P2P_READ)
     sender GPU                          receiver GPU
     ┌───────────────┐                   ┌───────────────┐
     │ ncclSendMem   │                   │ ncclRecvMem   │
     │ SIMPLE buffer │                   │ LL buffer     │
     └───────┬───────┘                   │ LL128 buffer  │
             │                           └───────────────┘
             └──────── load (SIMPLE) ─── receiver kernel
       (LL / LL128: sender still stores into receiver buffers)
```

### SHM

- `shmCanConnect`：`NCCL_SHM_DISABLE=1` 时不用；`ncclTopoCheckNet` 判定走网络时不用；要求同 `hostHash` 且同 `shmDev`（同一个 `/dev/shm`，容器间要共享）。
- 缓冲：`ncclShmAllocateShareableBuffer` 在 cuMem host 可用（`ncclCuMemHostEnable()`）时用 `ncclCuMemHostAlloc`，否则 `ncclShmOpen` → `src/os/linux.cc::ncclOsShmOpen` 建共享内存文件并用 `cudaHostRegister(..., cudaHostRegisterPortable | cudaHostRegisterMapped)` 注册给 GPU（日志 `SHM legacy: sharing buffer with GPU via cudaHostRegister + cudaHostGetDevicePointer`）；`NCCL_SHM_LOCALITY` 决定缓冲放在发送端还是接收端 NUMA（默认 `SHM_RECV_SIDE`）。
- 数据路径：GPU → host 共享内存 → GPU，两次跨 PCIe，带宽和延迟都明显差于 P2P；常见于 P2P 被禁用（跨 NUMA、ACS、虚拟化）的机器。

### NVLS（NVLink SHARP）

- 启用：`ncclNvlsInit`（`nvls.cc:166`）要求 `NCCL_NVLS_ENABLE` 非 0（默认 2 = 自动）、本 comm 至少 2 块 GPU、驱动有 `cuMulticastCreate`，且设备属性 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` 为真；一块 GPU 上有多个 rank 时不支持。日志：`NVLS multicast support is [not ]available on dev %d, transport ..., symmetric multimem ...`。
- multicast 组：`multicast.cc::ncclMcGroupBuildPartitions`：本地 rank 0 `ncclMcCreate`（`cuMulticastCreate`，FABRIC 时导出句柄）→ `bootstrapIntraNodeBroadcast` 分发 → 其他 rank `ncclMcImport` → 每个 rank `cuMulticastAddDevice` → 节点内 barrier（注释：`cuMemMap` 会阻塞到所有设备加入，用 abort 感知的 barrier 防止对端失败导致挂死）→ `cuMemAddressReserve/cuMemMap/cuMemSetAccess` 映射 multicast VA。每块 GPU 的单播（UC）物理内存再通过 `cuMulticastBindMem`（`ncclMcPartitionBindMem`）绑到组里。
- 通道布局：`ncclNvlsSetup` 中每个 NVLS head、每个 channel 有两对连接，注释 "Reduce UC -> MC" 与 "Broadcast MC -> UC"；`NCCL_NVLS_NCHANNELS`、`NCCL_NVLS_CHUNKSIZE`（默认 128K）可调。
- 设备侧：`all_reduce.h` 中 `RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>` 把线程分成 scatter / gather / reduce 几组，reduce 组使用 `ProtoSimple<1, 1, COLL_UNROLL, 1, 1>`（模板参数 `MultimemSrcs=1, MultimemDsts=1`）；`common_kernel.h` 中 `applyLoadMultimem` 发出 `multimem.ld_reduce.relaxed.sys.global.add...`（`reduce_kernel.h:1131`）从 multicast 地址读出"所有 GPU 对应位置之和"，`multimem_st_global`（`op128.h`）发出 `multimem.st.global...` 一次写到所有 GPU。归约在 NVSwitch 中完成，GPU SM 只发起访存。

单节点 NVLS AllReduce 的数据流（`all_reduce.h` 的 `work->oneNode` 分支 + `ncclNvlsSetup` 的缓冲布局）。每个 head h、每个 channel 在同一块已绑定到 multicast 的物理内存上有两个槽，同一块内存既可用 UC 指针（只访问本卡）也可用 MC 指针（访问组内所有卡）：

```text
  per head h, per channel (nvls.cc: dataUc = UC view, dataPartition = MC view)
    slot "Reduce UC -> MC"    : written via UC by scatter,  read via MC by reduce
    slot "Broadcast MC -> UC" : written via MC by reduce,   read via UC by gather

  every GPU                          GPU whose headRank == h          every GPU
  ─────────                          ───────────────────────          ─────────
  sendbuff
     │ scatter warps (UC store)
     ▼
  reduce slot h ─────────────►  multimem.ld_reduce(MC addr)
  (on all GPUs)                   NVSwitch sums slot h of all GPUs
                                         │ result in registers
                                         ▼
                                multimem.st(MC addr) ─────────────► bcast slot h
                                  NVSwitch writes every GPU         (on all GPUs)
                                                                       │ gather warps
                                                                       ▼ (UC load)
                                                                    recvbuff
```
- NVLS_TREE：节点内 NVLS + 节点间 tree，用于多节点。
- 用户缓冲注册：`ncclCommRegister` 后，NVLS 可直接把用户缓冲绑到 multicast（`nvls_ub.cc::ncclNvlsUbRegister`、`ncclNvlsLocalRegisterBuffer`、`ncclNvlsGraphRegisterBuffer`），省掉拷入/拷出 NCCL 内部缓冲；源码注释"MNNVL does not support NVLS buffer registration"。

### Copy engine 集合（`src/ce_coll.cc`）

`ncclCeAllGather/ncclCeAlltoAll/ncclCeScatter/ncclCeGather` 用 `cudaMemcpyBatchAsync` 等 copy engine 接口搬数据（不占 SM），用 `CUstreamBatchMemOpParams` 写对端 flag 做同步（`ncclPrepMCSync`/`ncclPrepUCSync`）；参数 `NCCL_CE_CHUNK_SIZE`（默认 8M）、`NCCL_CE_COLL_AG_MULTICAST_THRESHOLD`。这是 ai-infra-book 6.4.4 所说"NCCL 的部分路径用节点内复制引擎承担数据搬移"的实现位置。

## 源码导读

| 路径 | 函数 | 作用 |
|---|---|---|
| `src/transport/p2p.cc` | `p2pCanConnect` | `ncclTopoCheckP2p`（路径与 `NCCL_P2P_LEVEL`）→ `ncclTopoCheckNet` → 同 host 检查 → `cudaDeviceCanAccessPeer` → legacy IPC 探测 |
| | `p2pGetInfo` | 是否 P2P read（Ampere + NVLink 默认开，`NCCL_P2P_READ_ENABLE` 覆盖） |
| | `p2pSendSetup` / `p2pRecvSetup` | 选 `P2P_DIRECT/CUMEM/IPC/INTERMEDIATE`，计算缓冲大小（read 模式下 Simple 缓冲挂在 `ncclSendMem` 之后） |
| | `p2pSendProxySetup` / `p2pRecvProxySetup` | 在 proxy 中 `ncclP2pAllocateShareableBuffer`（缓冲由目标 GPU 的 proxy 分配） |
| | `p2pMap` → `ncclP2pImportShareableBuffer` | 导入对端句柄、映射到本地 VA |
| | `ipcRegisterBuffer`、`ncclIpcLocalRegisterBuffer`、`ncclIpcGraphRegisterBuffer` | 用户缓冲 IPC 注册（零拷贝） |
| `src/transport/shm.cc` | `shmCanConnect`、`shmSendSetup`、`ncclShmAllocateShareableBuffer`、`ncclShmImportShareableBuffer` | SHM 判定与共享内存缓冲 |
| `src/transport/nvls.cc` | `ncclNvlsInit`、`ncclNvlsSetup`、`ncclNvlsBufferSetup`、`ncclNvlsTreeConnect`、`ncclNvlsTuning` | NVLS 能力检测、通道与缓冲建立 |
| `src/transport/multicast.cc` | `ncclMcCreate`、`ncclMcImport`、`ncclMcGroupBuildPartitions`、`ncclMcPartitionBindMem` | multicast 对象的创建、导入、绑定 |
| `src/mnnvl.cc` | `ncclMnnvlCheck` | 多节点 NVLink 域检测，切换 FABRIC 句柄 |
| `src/misc/cudawrap.cc` | `ncclCuMemEnable`、`ncclCuMemGdrSupport`、`ncclCuMemHostEnable` | cuMem 开关与能力 |

## 性能要点与常见坑

- 首先看日志里每个 channel 走的是 `P2P/CUMEM`、`P2P/IPC`、`SHM` 还是 `NET`；8 卡 NVLink 机器上出现 `SHM` 说明 P2P 被禁用（查 `NCCL_P2P_DISABLE/LEVEL`、ACS、NVML 报错）。
- 容器内 SHM 需要足够大的 `/dev/shm` 且多容器共享同一个 `/dev/shm` 才能互通（`shmDev` 比较）；Docker 默认 `/dev/shm` 很小，常见报错与 `ncclShmOpen` 相关（userguide `troubleshooting/runtime_and_mpi_issues.rst` 的 "Shared memory" 节）。
- cuMem fd 传递依赖 proxy 的 UDS 线程；某些容器/安全策略限制 Unix domain socket 时可能失败，可对比 `NCCL_CUMEM_ENABLE=0`（待确认：具体报错形态；确认方式：`grep -n "WARN" third_party/nccl/src/proxy.cc` 中 UDS 相关部分）。
- NVLS 会预留 multicast 资源和额外显存（buffer + credit），可用 `NCCL_NVLS_ENABLE=0` 对比内存与性能；NVLS 对大消息 AllReduce/ReduceScatter/AllGather 收益大，小消息延迟受 multicast 设置开销影响，由 tuning 自动选择。
- P2P read vs write：默认只在两端都是 cc 80（Ampere）且经 NVLink 时开启 read（`paths.cc::ncclTopoCheckP2p`），其他情况默认 write；可用 `NCCL_P2P_READ_ENABLE` 强制对比。
- CE 路径（`P2P_USE_CUDA_MEMCPY`、ce_coll）不占 SM，适合与计算重叠，但依赖 proxy/stream 调度，小消息延迟更高。

## 面试自测

1. 问：在同机两 GPU 间建立 NCCL P2P 连接，句柄如何交换？cuMem 和 legacy IPC 有何不同？
   答：缓冲由一侧的 proxy 分配，legacy 模式用 `cudaIpcGetMemHandle` 得到 64B 句柄经 proxy/bootstrap 消息交给对端，对端 `cudaIpcOpenMemHandle` 映射；cuMem 模式用 `cuMemCreate` + `cuMemExportToShareableHandle`，POSIX fd 需经 Unix domain socket 传递（NCCL 的 UDS proxy 线程），FABRIC 句柄可直接按字节传并支持跨节点 NVLink；导入后 `cuMemMap`+`cuMemSetAccess` 映射。cuMem 支持更细粒度的 VA 管理、多节点和 multicast 绑定。
2. 问：NVLS AllReduce 的数据路径是什么？与 ring 相比好在哪？
   答：各 GPU 的缓冲绑到同一个 multicast 对象；reduce 线程对 multicast 地址做 `multimem.ld_reduce`，NVSwitch 从所有 GPU 读出对应数据并归约后返回，再用 `multimem.st` 一次写回所有 GPU。按每块 GPU 负责 S/n 切片推算（非源码原文）：每块 GPU 每方向的 NVLink 流量约为 S，而 ring 约为 2(n-1)S/n≈2S；步数是常数而不是 2(n-1)；归约在交换机里做，节省 SM 的读-加-写。
3. 问：什么时候 NCCL 会在同一台机器内走 SHM 甚至 NET？
   答：P2P 不满足（路径超过 `NCCL_P2P_LEVEL`，默认 PXB；NVML 报 P2P 不可用；`cudaDeviceCanAccessPeer` 失败；legacy IPC 不可用；用户禁用）时走 SHM；SHM 也不满足（不同 `/dev/shm`、`NCCL_SHM_DISABLE`）或 `ncclTopoCheckNet` 判定（如 `NCCL_NET_DISABLE_INTRA` 相关逻辑）时走 NET。
4. 问：`P2P/indirect` 是什么情况？
   答：两 GPU 无直接 NVLink，但都与某块中间 GPU 有 NVLink（PATH_NVB），NCCL 让数据经中间 GPU 的缓冲转发（`P2P_INTERMEDIATE`），比走 PCIe 快。
5. 问：如果你要为一个新的 scale-up 互联（支持 load/store 和交换机归约）做 NCCL 适配，需要改哪些地方？
   答：拓扑：在 XML/`topo.cc` 中加入新链路类型与带宽，让 `ncclTopoCheckP2p` 和搜索识别；内存：提供导出/导入对端显存的句柄机制（类似 cuMem FABRIC）；transport：在 `p2p.cc` 风格的 transport 中实现 setup/connect/map；交换机归约：类似 `multicast.cc`+NVLS kernel 的 multicast 绑定与 `ld_reduce` 指令路径；tuning：在 `src/tuning/` 加代价模型；验证：nccl-tests busbw 对比理论值。也可以先通过 net 插件或 device API 接入，作为最小侵入方案。

## 待深入

- `ncclNvlsBufferSetup` 中 buffer 与 credit 两类分区的大小计算，以及 `ncclNvlsTuning` 选 NVLS 通道数的逻辑。
- NCCL 对称内存（`ncclCommWindowRegister` 一类窗口 API）与 `src/device/symmetric/` 中基于 LSA/multimem 的 kernel。
- `P2P_SAME_PID`（`p2p.cc:334`）= 同 `hostHash` 且同 `pidHash`，即同一进程内的多 GPU 才走 direct pointer；多进程（每进程一卡，最常见）走 CUMEM/IPC。
