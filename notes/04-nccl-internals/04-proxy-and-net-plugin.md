# proxy 线程、net plugin 接口与 IB transport

> 层次：中层（连接下层 scale-out） · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）、ai-infra-book@3bdcb4fc 第 6 章 6.4.4 节、第 7 章 7.3 节

## 一句话

GPU kernel 只负责把数据写进（或读出）一块"网络缓冲"并更新计数，真正调用网卡收发的是每个 GPU 对应的 CPU proxy progress 线程：它轮询这些计数，调用 net 插件的 `isend/irecv/test/iflush`；内置插件是 `IB`（verbs）和 `Socket`，外部插件（`libnccl-net*.so`）通过同一张 `ncclNet_vN_t` 函数表接入。

## 在调用链中的位置

- 上层：`src/enqueue/enqueue.cc` 在生成 kernel plan 时为跨节点（或 PXN/CollNet）连接生成 `ncclProxyOp`（`ncclAddProxyOpIfNeeded` → `ncclProxySaveOp`），kernel 发射后由 `hostStreamPlanTask` → `uploadProxyOps` + `ncclProxyStart` 交给 proxy 线程。
- 本模块：`src/proxy.cc`（线程与 op 池）、`src/transport/net.cc`（NET transport：setup/connect/progress）、`src/plugin/net.cc` + `src/plugin/plugin_open.cc`（插件加载）、接口定义 `src/include/plugin/nccl_net.h`、`src/include/plugin/net/net_v6.h ~ net_v12.h`。
- 下层：`src/transport/net_ib/`（`ncclNetIb`，verbs 实现，细节见 [../01-rdma/README.md](../01-rdma/README.md)）、`src/transport/net_socket.cc`（`ncclNetSocket`）、外部插件（参考 `plugins/net/example/plugin.c`）。

## 核心概念

### 三类 proxy 相关线程（`src/proxy.cc`）

| 线程名（`ncclSetThreadName`） | 函数 | 职责 |
|---|---|---|
| `NCCL Service %2d` | `ncclProxyService` | 接收本节点 rank 发来的控制消息（`ncclProxyMsgInit/SharedInit/Setup/Connect/Register/...`，枚举在 `src/include/proxy.h:424`），代表它们分配网络缓冲、`listen/connect/accept`、注册内存 |
| `NCCL UDS Svc%2d` | `ncclProxyServiceUDS` | 通过 Unix domain socket 传递 cuMem 文件描述符（`ncclProxyMsgGetFd/QueryFd`） |
| `NCCL Progress%2d` | `ncclProxyProgress` | 数据面：循环 `progressOps`（逐个调用 op 的 `progress` 回调，如 `sendProxyProgress`/`recvProxyProgress`），空闲或每 `NCCL_PROGRESS_APPENDOP_FREQ`（默认 8）轮取一次新 op（`ncclProxyGetPostedOps`） |

每个 rank 有自己的 `proxyState`（源码注释：parent comm 与 split 出的子 comm 共享同一个 proxyState，`ncclProxyCreate` 只在 `refCount == 1` 时起线程），线程运行在该 rank 的 GPU 上下文；控制消息通过 `ncclProxyConnect` + `ncclProxyCallBlocking/Async` 发给目标 rank 的 proxy（PXN 时是同节点另一块 GPU 所属 rank 的 proxy）。

下图把三类线程和 GPU kernel 放在一起：用户线程只负责把 proxy op 放进 op 池并 post（CUDA graph 模式下这一步由 `cudaLaunchHostFunc` 回调在每次 replay 时执行），progress 线程与 kernel 之间不直接调用，只通过共享的 `tail/head/connFifo` 内存互相轮询。

```text
 ┌────────────────────┐                  ┌────────────────────┐
 │ user thread        │ cuLaunchKernelEx │ GPU kernel         │
 │ (enqueue / launch) │ ───────────────► │ (primitives)       │
 └─────────┬──────────┘                  └─────────┬──────────┘
           │ uploadProxyOps → op pool              │ tail / head / connFifo
           │ ncclProxyStart → ncclProxyPost        │ (memory polled by proxy)
           ▼                                       ▼
 ┌────────────────────────────────────────────────────────────┐
 │ "NCCL Progress" thread: ncclProxyProgress                  │
 │   ncclProxyGetPostedOps → ProxyAppend → progressOps        │
 │   sendProxyProgress / recvProxyProgress                    │
 └─────────────────────────────┬──────────────────────────────┘
                               │ ncclNet->isend / irecv / test / iflush
                               ▼
                        net plugin → NIC

 control plane (local ranks, or PXN peers):
   ncclProxyCallBlocking/Async ──► "NCCL Service"  Setup / Connect / Register ...
   cuMem fd over UDS           ──► "NCCL UDS Svc"  GetFd / QueryFd
```

### 为什么需要 CPU proxy

- 传统 verbs 的 post send/recv 和 poll CQ 由 CPU 完成；GPU kernel 不能直接敲网卡门铃（GPU 发起网络是另一条路线，如 GIN/GDAKI，`src/gin/`、`src/transport/net_ib/gdaki/`）。
- ai-infra-book 6.4.4 把这种模式概括为"CPU 代理线程：GPU 写就绪标志 → CPU 轮询 → 构造描述符、敲门铃 → 完成后再告诉 GPU"，控制路径要多次穿越 PCIe，且 CPU 线程的请求处理能力有限；第 7 章 7.3.2 讨论由加速器直接发起通信的替代方案。

### GPU 与 proxy 的握手

- 发送：kernel 把数据写入 `buffs[p]` 的某个 slot 并更新 `tail`（以及 `connFifo[slot].size`）；proxy 在 `sendProxyProgress` 中看到 slot 就绪后 `ncclNet->isend`，`test` 完成后更新 `head`，让 GPU 复用该 slot。
- 接收：proxy 预先 `irecv` 到 slot；`test` 完成后，若用了 GDR 且需要（`ncclTopoNeedFlush`），调用 `iflush`（或 GDRCopy 的 PCIe 读，源码注释 "Force a PCIe read from GPU memory"）保证网卡写入对 GPU 可见，然后更新 `tail` 通知 kernel。
- 深度受 `NCCL_STEPS`（8）限制；`NCCL_NET_SHARED_BUFFERS`（`net.cc:170`）决定是否多连接共享 buffer pool。

下图按 `src/transport/net.cc` 中 `sendProxyProgress` / `recvProxyProgress` 的步进游标（`sub->posted/received/transmitted/done`）展开上面两条，左右两侧的 head/tail 就是 [03 第 7 节](03-channels-protocols-kernels.md) 里 `waitPeer/postPeer` 用的那组计数：

```text
 send side  (Simple, non-shared buffers)
 send kernel                          send proxy: sendProxyProgress
 ───────────                          ─────────────────────────────
 waitPeer(head): free slot?
   connFifo[slot].size = n
 write data into slot
 postPeer: tail = step ─────────────► transmitted: tail > base + transmitted && size != -1
                                        └ ncclNet->isend(slot)  ═══ to remote ═══►
                                      done: ncclNet->test() reports done
 head moves, slot reusable ◄───────────  └ size = -1; head = base + done

 recv side
 recv proxy: recvProxyProgress                          recv kernel
 ─────────────────────────────                          ───────────
 posted:      ncclNet->irecv(slot)
 received:    ncclNet->test() reports done  ◄═══ from remote
 transmitted: iflush if needed (GDR)
              tail = base + transmitted ──────────────► waitPeer(tail): data ready
                                                        read / reduce slot
 done:        head > base + done      ◄──────────────── postPeer: head = step
```

## 源码导读

### 1. proxy op 生命周期

- `src/enqueue/enqueue.cc::ncclAddProxyOpIfNeeded`：`ncclProxySaveOp(comm, op, &needed)` 先询问是否需要 proxy（纯 NVLink/SHM 的 op 不需要）。
- `uploadProxyOps` → `ncclProxySaveOp`（`proxy.cc:591`）→ `SaveProxy` → `ncclLocalOpAppend`：按 channel、peer 把 op 写入共享 op 池。
- `hostStreamPlanTask` → `ncclProxyStart`（`proxy.cc:1013`）→ `ncclProxyPost`：通知 progress 线程。非 graph 模式直接调用（`ncclLaunchKernelAfter_NoCuda`）；graph 捕获时通过 `cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan)`，每次 replay 执行。
- `ncclProxyProgress`（`proxy.cc:951`）：`progressOps` → `ncclProxyGetPostedOps` → `ProxyAppend`；出错时把错误写入 `proxyState->asyncResult` 并退出循环，主线程通过 `ncclCommGetAsyncError` 感知。
- 调试：`NCCL_PROXY_DUMP_SIGNAL`（`proxy.cc:922`）设一个信号号，向进程发该信号会调用 `ncclDumpProxyState` 打印当前所有 proxy op 的状态（hang 时看哪个 op 卡在哪个 step）。

### 2. NET transport（`src/transport/net.cc`）

- `netTransport`（`net.cc:2126`）实现 `canConnect`、`sendSetup/recvSetup`、`sendConnect/recvConnect`、`sendProxySetup/recvProxySetup`、`sendProxyConnect/recvProxyConnect`、`sendProxyProgress/recvProxyProgress`、注册函数等。
- `sendSetup`：`ncclTopoGetNetDev` 选网卡 → `ncclTopoCheckGdr` 决定 `useGdr` → 打印 `Channel %02d/%d : a[x] -> b[y] [send] via NET/<plugin>/<dev>[/GDRDMA][/Shared]`；若经 PXN，则写成 `NET/<plugin>/<dev>(<proxyRank>)`，表示由另一块 GPU 的 proxy 代发。
- `sendProxyConnect`：在 proxy 线程中调用 `ncclNet->connect`、`regMr`（GDR 时注册 GPU 内存，可能走 `regMrDmaBuf`）。
- 选择 transport 的顺序由 `src/transport.cc` 的 `ncclTransports[] = {&p2pTransport, &shmTransport, &netTransport, &collNetTransport}` 决定：`selectTransport` 依次调用 `canConnect`，第一个返回可用者胜出。所以同机 GPU 只有在 P2P、SHM 都不可用（或 `ncclTopoCheckNet` 判定走网络更好）时才走 NET。

### 3. net 插件接口（`src/include/plugin/net/net_v12.h`）

`ncclNet_v12_t` 主要成员（当前 `ncclNet_t` 即 v12，`nccl_net.h:54`）：

| 函数 | 作用 |
|---|---|
| `init(ctx, commId, config, logFunction, ...)` / `finalize(ctx)` | 每个 comm 一个上下文 |
| `devices(&ndev)` / `getProperties(dev, props)` | 枚举网卡与属性：`name`、`pciPath`、`guid`、`ptrSupport`（`NCCL_PTR_HOST/CUDA/DMABUF`）、`speed`（Mbps）、`latency`、`maxComms`、`maxRecvs`、`regIsGlobal`、`forceFlush`、`railId/planeId` 等 |
| `listen` / `connect` / `accept` | 建连：handle（最大 `NCCL_NET_HANDLE_MAXSIZE` 128B）经 bootstrap 交换 |
| `regMr` / `regMrDmaBuf` / `deregMr` | 内存注册 |
| `isend` / `irecv`（可多接收 `n`，带 tag） / `test` | 异步收发与完成查询 |
| `iflush` | GDR 接收后的可见性 flush |
| `getDeviceMr` / `irecvConsumed` | 设备侧网络卸载（`netDeviceType`） |
| `makeVDevice` | 多网卡合成虚拟设备 |
| `setNetAttr` | 按 op/algo/proto 给插件提示 |

另有 `ncclCollNet_v12_t`（`iallreduce`、`iallgather`、`ireducescatter`、`reduceSupport`），用于 SHARP 类网络内归约（CollNet 算法）。

### 4. 插件加载（`src/plugin/net.cc`、`plugin_open.cc`）

- `initPluginLibsOnceFunc`：
  - 未设 `NCCL_NET_PLUGIN`：先尝试默认 `libnccl-net.so`；
  - 设了：按逗号拆成列表逐个尝试；值为 `none` 时不加载外部插件；
  - 之后总是注册内置的 `ncclNetIb`、`ncclNetSocket`（Linux；Windows 是 `ncclNetNd` 与 Socket）。
- `openPluginLib`：名字不是路径也不是 `lib*.so` 时，会尝试 `libnccl-net-<name>.so`。
- `ncclNetPluginLoad`：按版本 `{12, 11, 10, 9, 8, 7, 6}` 依次 `getNcclNet_vN(dlHandle)` 找符号（如 `NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v12`），并顺带加载 CollNet；版本适配层在 `src/plugin/net/net_v6.cc ~ net_v12.cc`。
- 选定插件：插件 `init/devices` 成功且有设备时使用；用户可通过 `NCCL_NET`（`init.cc:2426`，写入 `comm->config.netName`）指定名字，例如强制 `Socket`。日志：`Successfully loaded external network plugin ...`、`Initialized NET plugin %s`、`Assigned NET plugin %s to comm`。

### 5. 与 IB transport 的关系

- `ncclNetIb`（`src/transport/net_ib/common.cc:244`）就是一个实现了 `ncclNet_t` 的"内置插件"：`ncclIbInit/Devices/GetProperties/Listen/Connect/Accept/RegMr/RegMrDmaBuf/DeregMr/Isend/Irecv/Iflush/Test/...`。
- 分层：NET transport（`net.cc`，与网络无关的缓冲/step/proxy 逻辑）→ `ncclNet_t` 函数表 → `net_ib`（QP 建立 `connect.cc`、收发 `p2p.cc` 中 `wrap_ibv_post_send`/`wrap_ibv_poll_cq`、GDR 检测 `gdr.cc`）。
- 接收端匹配：IB 插件用 RDMA write 把数据直接写到对端 `irecv` 提供的地址，接收方通过一个 FIFO（`ncclIbPostFifo`）先把自己的地址/rkey 告诉发送方，类似 CTS。
- 外部插件（如 `plugins/net/google-fastsocket`、厂商 RoCE/EFA 插件）替换的是 `net_ib` 这一层，NET transport 与 proxy 逻辑不变。

三层关系如下：NET transport 只认 `ncclNet_t` 函数表；内置 IB/Socket 与外部 `.so` 插件是这张表的不同实现，外部插件先于内置插件尝试加载（第 4 节）：

```text
 ┌───────────────────────────────────────────────────┐
 │ NET transport          src/transport/net.cc       │
 │ setup / connect / sendProxyProgress / recv...     │
 └─────────────────────────┬─────────────────────────┘
                           │ proxyState->ncclNet->isend / irecv / test / iflush ...
 ┌─────────────────────────┴─────────────────────────┐
 │ ncclNet_t  (= ncclNet_v12_t)                      │
 │ per-version adapters: src/plugin/net/net_v*.cc    │
 └──────┬─────────────────────┬──────────────────┬───┘
        ▼                     ▼                  ▼
 ┌──────────────┐   ┌────────────────┐   ┌─────────────────────────┐
 │ ncclNetIb    │   │ ncclNetSocket  │   │ libnccl-net-<name>.so   │
 │ net_ib/      │   │ net_socket.cc  │   │ symbol ncclNetPlugin_vN │
 └──────┬───────┘   └────────────────┘   └─────────────────────────┘
        ▼
 wrap_ibv_post_send / wrap_ibv_poll_cq  ──► RDMA NIC
```

## 性能要点与常见坑

- proxy 线程需要 CPU 核：被绑到与 GPU 同 NUMA 的核上（init 中 `ncclTopoGetCpuAffinity`）；若训练进程把这些核占满或容器 CPU 配额太低，跨节点带宽和延迟会明显变差。
- GDR 是否开启看 `[send] via NET/IB/x/GDRDMA` 日志；没有 GDRDMA 时数据经 host 内存中转，大消息带宽通常明显下降。常见原因：`nvidia-peermem` 未加载或不支持 DMA-BUF、拓扑距离超过 `NCCL_NET_GDR_LEVEL`、虚拟化下拓扑不对。
- PXN（`NET/IB/x(proxyRank)`）让数据先经 NVLink 到与目标网卡"同 rail"的 GPU，再发出；可以减少跨 rail 流量，但会多一次 NVLink 拷贝；`NCCL_PXN_DISABLE=1` 可对比。
- 插件版本不匹配时会打印 `External network plugin %s is unsupported` 并回落到内置 IB/Socket，可能导致意外地走 Socket（带宽极低）。看 `Initialized NET plugin` 一行确认实际插件。
- CUDA graph 下每次 replay 都依赖 host 回调推 proxy op；host 线程被阻塞会导致 GPU 侧等待。

## 面试自测

1. 问：NCCL 跨节点发送一个 chunk 的完整路径？
   答：kernel 把数据写到 NET 连接的发送缓冲 slot（GDR 时是 GPU 显存，否则是 pinned host 内存），更新 tail；proxy progress 线程看到后调用 `ncclNet->isend`；IB 插件 `ibv_post_send` RDMA write 到对端 `irecv` 提供的地址；对端 proxy `test` 到完成、必要时 `iflush`，更新 tail；对端 kernel 在 `waitPeer` 中看到 tail 前进后读取数据。
2. 问：写一个 NCCL net 插件最少要实现哪些函数？如何让 NCCL 加载它？
   答：实现 `ncclNet_v12_t`（或兼容的旧版本）的 `init/devices/getProperties/listen/connect/accept/regMr/deregMr/isend/irecv/iflush/test/close*/finalize` 等，导出符号 `ncclNetPlugin_v12`；编译为 `libnccl-net-<name>.so`，设置 `NCCL_NET_PLUGIN=<name>` 或放在库路径中命名为 `libnccl-net.so`。参考 `plugins/net/example/plugin.c`。
3. 问：proxy service 线程和 progress 线程有什么区别？
   答：service 线程处理控制面 RPC（setup/connect/注册，来自本节点的 rank 或 PXN 对端），progress 线程处理数据面，循环推进所有 proxy op。
4. 问：为什么接收端需要 flush？
   答：网卡通过 PCIe 写 GPU 显存是 posted write，CQE 到达时数据不一定已对 GPU SM 可见；`iflush`（IB 中是一次 RDMA read 回自己）或 GDRCopy 的 PCIe 读，强制前面的写完成。某些平台（如 C2C）用 `ncclTopoNeedFlush` 判定是否需要。
5. 问：IB transport 和 NET transport 是什么关系？
   答：NET transport 是 NCCL 的通用网络传输层（缓冲、step、proxy），IB 只是它下面的一个 `ncclNet_t` 实现；外部插件替换的就是这一层。

## 待深入

- `NCCL_NET_SHARED_BUFFERS`/`NCCL_NET_SHARED_COMMS` 对 p2p 与集合通信内存占用的影响：`net.cc::sharedNetBuffersInit`。
- 待确认：多插件同时可用时，各 comm 选择插件的完整优先级（`netPluginLibs` 顺序、`NCCL_NET` 名字匹配）；确认方式：读 `src/plugin/net.cc` 中 `ncclNetInit` 附近（"Assigned NET plugin" 前后）的循环。
- GIN（GPU-initiated networking）与 RMA 插件：`src/gin/`、`src/include/plugin/nccl_gin.h`、`nccl_rma.h`，是 GPU 直接发起网络的路线，与 06-comm-compute-fusion 的 NVSHMEM 主题相关。
- IB 端口故障恢复：`src/transport/net_ib/p2p_resiliency.cc`、`p2p_resiliency_recovery.cc`（`NCCL_IB_RESILIENCY_PORT_FAILOVER` 等，默认 0）。
