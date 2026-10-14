# GPUDirect RDMA 与 NCCL 的 IB transport

> 层次：下层（与中层 NCCL 的接口）· 来源：third_party/nccl@3e46c08（`src/transport/net_ib/`、`src/transport/net.cc`、`src/graph/paths.cc`）、third_party/perftest@05200c2、third_party/rdma-core@bd3282a、third_party/ai-infra-book@3bdcb4fc 第 6.4.4、7.3.2 节

## 一句话

GPUDirect RDMA 让网卡直接 DMA 读写 GPU 显存，跨机数据不再经过主机内存。NCCL 的 `net_ib` 把这件事落成四步：判断能不能 GDR（拓扑距离 + 内核模块 / dmabuf），把 GPU buffer 注册成 MR，用 TCP 交换 QP 信息建多条 RC QP，然后用"接收端 RDMA WRITE 一个 CTS → 发送端 RDMA WRITE 数据 + WRITE_WITH_IMM 通知"的方式搬运。

## 在调用链中的位置

```
NCCL proxy 线程（src/transport/net.cc：sendProxy*/recvProxy*，调 proxyState->ncclNet->...）
   │  ncclNet_t 函数表
ncclNetIb（src/transport/net_ib/common.cc）
   ├─ init / devices / getProperties ── init.cc
   ├─ listen / connect / accept ────── connect.cc
   ├─ regMr / regMrDmaBuf / deregMr ── reg.cc
   ├─ isend / irecv / iflush / test ── p2p.cc
   └─ GDR / dmabuf 能力探测 ────────── gdr.cc
   │  wrap_ibv_*（src/misc/ibvwrap.cc，运行时 dlopen libibverbs，见 src/misc/ibvsymbols.cc）
libibverbs / mlx5dv → RNIC ──PCIe P2P──> GPU 显存
```

- 上层：NCCL 拓扑模块决定每条 net 连接是否用 GDR（`src/graph/paths.cc`：`ncclTopoCheckGdr`），`src/transport/net.cc` 根据 `useGdr` / `useDmaBuf` 决定 buffer 放显存还是主机内存，以及调 `regMrDmaBuf` 还是 `regMr`。
- 下层：verbs（见 [01-verbs-basics.md](01-verbs-basics.md)、[02-transport-and-ops.md](02-transport-and-ops.md)）。

## 核心概念

### GPUDirect RDMA 的两条内核路径

| 路径 | 机制 | NCCL 如何探测 |
|---|---|---|
| **peermem**（`nvidia-peermem`，旧名 `nv_peer_mem`） | 内核模块把 GPU 显存注册进 RDMA 子系统的 "peer memory" 接口，之后普通 `ibv_reg_mr(gpu_ptr)` 就能 pin 住显存 | `gdr.cc`：`ibGdrSupportInitOnce` 检查 `/sys/kernel/mm/memory_peers/nv_mem/version`、`.../nv_mem_nc/version`、`/sys/module/nvidia_peermem/version` 之一是否存在 → `ncclIbGdrSupport` |
| **dmabuf**（上游内核标准接口） | CUDA 用 `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)` 导出 dmabuf fd，再调 `ibv_reg_dmabuf_mr(pd, offset, len, iova, fd, access)`（`rdma-core/libibverbs/verbs.h`） | `gdr.cc`：`ibDmaBufSupportInitOnce` 用 fd=-1 做一次哑调用 `ibv_reg_dmabuf_mr`：errno 为 `EOPNOTSUPP/EPROTONOSUPPORT` 表示不支持，`EBADF` 表示支持 → `ncclIbDmaBufSupport` |

- `init.cc`：`ncclIbGetPhysProperties` 中，`ptrSupport` 默认是 `NCCL_PTR_HOST`；peermem 可用时加 `NCCL_PTR_CUDA`，dmabuf 可用时加 `NCCL_PTR_DMABUF`。`graph/topo.cc` 根据它给 net 节点打 `gdr` 属性，日志里打印 `"GPU Direct RDMA Enabled/Disabled for HCA ..."`。
- perftest 对应：`--use_cuda=<gpu>` 走 peermem 路径（`cuMemAlloc` 出的指针直接 reg_mr），加 `--use_cuda_dmabuf` 走 dmabuf 路径（`src/cuda_memory.c` 中调 `cuMemGetHandleForAddressRange`，`src/perftest_resources.c` 中调 `ibv_reg_dmabuf_mr`）。README 列出的 dmabuf 要求是 CUDA ≥ 11.7、NVIDIA 开源内核模块 ≥ 515。
- mlx5 还有一个 "Data Direct" 变体：NCCL 在 `reg.cc` 中，如果 `capsProvider.mlx5.dataDirect` 为真，就改调 `mlx5dv_reg_dmabuf_mr(..., MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT)`，对应 `NCCL_IB_DATA_DIRECT`（默认 1）。

### NCCL 何时启用 GDR（`src/graph/paths.cc`：`ncclTopoCheckGdr`）

1. NIC 和 GPU 都必须标记 `gdrSupport`。
2. 对 read 方向（发送端由网卡从显存读）：`NCCL_NET_GDR_READ`（默认 -2，即自动）。自动模式下，Ampere 之前（`cudaCompCap < 80`）且没有 NVLink 对端时关闭 GDR read。
3. 距离判定：默认阈值 `PATH_PXB`（GPU 与 NIC 之间只经过 PCIe switch、不经过 CPU）；`NCCL_NET_GDR_C2C`（默认 1）在 C2C 系统上可以放宽到 `PATH_P2C`；用户可以用 `NCCL_NET_GDR_LEVEL` 覆盖。PXN 场景下改用中转 GPU 的距离。
4. C2C 系统上，路径如果经过 PCI，就设为 `ncclTopoGdrModePci`（dmabuf 映射类型改为 PCIe，见 `net.cc`：`getHandleForAddressRangeFlags`）。
5. 是否需要 flush：`ncclTopoNeedFlush` 中，Hopper 及以后（`cudaCompCap >= 90`）默认不 flush，除非是 C2C 控制路径 + PCIe 数据路径（Data Direct）；Ampere 及以前总是 flush；`NCCL_NET_FORCE_FLUSH` 可以强制 flush。

**为什么要 flush**：网卡对显存的 PCIe 写可能还没对 GPU kernel 可见（posted write），NCCL 接收端在把数据交给 kernel 之前，发一个 1 字节的 RDMA READ 读这块 GPU buffer（`p2p.cc`：`ncclIbIflush`，读回到主机上的 `gpuFlushHostMem`）。READ 的完成保证前面的写都已落地。flush 用的是一个自环 QP（`connect.cc`：`ncclIbCreateFlushQp`，注释写明 "loopback QP (connected to itself)"）。只有 GDR/dmabuf 可用且 `NCCL_GDR_FLUSH_DISABLE=0` 时才启用（`ncclIbAcceptImpl` 中的 `rComm->flushEnabled`）。

### 建链（`connect.cc`）

- **listen**（`ncclIbListen`）：建 socket 监听，把地址放进 `ncclIbHandle`，经 NCCL bootstrap 交给对端。
- **connect（发送端，`ncclIbConnectImpl`，非阻塞状态机）**：`ncclIbCommStateStart → Connect → SendDevList → RecvDevList → Send → Connecting → Connected`，每一步只推进 socket，没完成就返回，由 proxy 反复调用。
  1. `ncclSocketConnect`，然后先交换 `ncclNetVDeviceProps_t`（虚拟设备由几块物理网卡组成）和 `ncclIbDevExtraProps`（是否支持 OOO RQ）。
  2. QP 数：`nqps = max(本端, 对端) = NCCL_IB_QPS_PER_CONNECTION × ndevs`；`nDataQps = max(ndevs)`。
  3. 每块物理网卡 `ncclIbInitCommDevBase`：共享的 PD（`ibDev->pdRefs` 引用计数）、每个 comm 一个 CQ（大小 `NET_IB_MAX_REQUESTS × nQpsPerDev`）、pkey index。
  4. `ncclIbSenderQpsCreate`：建 RC QP，`qpAccessFlags = IBV_ACCESS_REMOTE_WRITE`；`NCCL_IB_ECE_ENABLE` 时 `ibv_query_ece`。
  5. 注册本端 **CTS FIFO**（`ctsFifoMr`，`LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ`），把 rkey、`ctsFifo` 地址、`lid`、`mtu`、GID、`link_layer`、SL、TC 放进 `ncclIbConnectionMetadata` 发给对端。SL 和 TC 的取值顺序是：环境变量 → 网络插件设置的 traffic class → 默认值。
  6. 收到对端 meta 后：检查 link_layer 一致（IB 和 RoCE 混用会直接报错，提示用 `NCCL_IB_HCA` 选网卡）；记录对端 completion records 的地址和 rkey；`ncclIbSenderQpsToRts` 用对端 QPN / GID / LID 把 QP 推到 RTR、RTS，对端支持 ECE 时先 `wrap_ibv_set_ece`。
  7. 通过 socket 发 `ready`。
- **accept（接收端，`ncclIbAcceptImpl`）**：对称流程；`ncclIbReceiverQpsCreateToRts` 建 QP（`REMOTE_WRITE | REMOTE_ATOMIC | REMOTE_READ`），可选 `ibv_set_ece`；按需预投 RECV（`ncclIbReceiverPrePostReceiveWorkRequests`，每 QP `NET_IB_MAX_REQUESTS` 个，即 `NCCL_NET_MAX_REQUESTS(32) × NCCL_NET_IB_MAX_RECVS(8)` = 256，见 `net_ib/common.h` 与 `include/plugin/nccl_net.h`；供 WRITE_WITH_IMM 消耗）；注册 `remCtsFifo`、`cmplsRecords`；按需建 flush QP。
- **QP 属性细节**（`ncclIbQpRtr` / `ncclIbQpRts`）：`rq_psn = sq_psn = 0`，不像 rc_pingpong 那样用随机 PSN；`min_rnr_timer=12`、`rnr_retry=7`、`max_rd_atomic=1`；`timeout`/`retry_cnt` 来自 `NCCL_IB_TIMEOUT`（默认 20）和 `NCCL_IB_RETRY_CNT`（默认 7）。IB 同子网用 LID，跨子网用 FLID + GRH；RoCE 用 GRH，`hop_limit=255`，`traffic_class=tc`。
- **GID 选择**：`ncclIbGetGidIndex` / `ncclUpdateGidIndex`，按 `NCCL_IB_ROCE_VERSION_NUM`、`NCCL_IB_ADDR_FAMILY`、`NCCL_IB_ADDR_RANGE` 遍历 GID 表，读 sysfs `gid_attrs/types` 判断 RoCE 版本；`NCCL_IB_GID_INDEX` 可以强制指定。

### MR 注册（`reg.cc`）

- `ncclIbRegMr` 和 `ncclIbRegMrDmaBuf` 都调 `ncclIbRegMrDmaBufInternal`。它对虚拟设备里的每块物理网卡分别注册（返回的 `ncclIbMrHandle` 中有 `mrs[ndevs]`）。
- `ncclIbRegMrDmaBufInternal2`：地址按页对齐后，在每块设备的 **MR cache**（按地址排序的数组 + `refs` 引用计数，加锁）里查找，命中就直接复用；没命中时：
  - 权限固定为 `LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ | REMOTE_ATOMIC`，支持时加 `IBV_ACCESS_RELAXED_ORDERING`（`NCCL_IB_PCI_RELAXED_ORDERING`，默认 2 = 自动；要用 `ibv_reg_mr_iova2`，即 IBVERBS_1.8 API）；
  - 有 fd 时调 `ibv_reg_dmabuf_mr` 或 `mlx5dv_reg_dmabuf_mr`，没有 fd 时调 `ibv_reg_mr`。
- `ncclIbDeregMr`：引用计数减到 0 才真正 `ibv_dereg_mr`。
- 调用方（`src/transport/net.cc`）：buffer 在显存且 `useDmaBuf` 时，先用 `cuMemGetHandleForAddressRange` 拿 dmabuf fd，再调 `regMrDmaBuf`，注册完立即 `close(fd)`；否则调 `regMr`，类型为 `NCCL_PTR_CUDA` 或 `NCCL_PTR_HOST`。注释写着 "FALL-THROUGH to nv_peermem GDR path"。DMA-BUF 总开关是 `NCCL_DMABUF_ENABLE`（默认 1，`src/init.cc`：`dmaBufSupported`）。

### 数据面：CTS + WRITE + WRITE_WITH_IMM（`p2p.cc`）

1. 接收端 `ncclIbIrecv` 为 n 个接收 buffer 填好 `ncclIbSendFifo`（地址、各设备的 rkey、size 等），`ncclIbPostFifo` 用一次 **RDMA WRITE** 写进发送端的 `ctsFifo[slot]`。可以 inline（`NCCL_IB_USE_INLINE`）；只在 `slot == devIndex` 时 signal，避免 SQ 堆满（见 [02-transport-and-ops.md](02-transport-and-ops.md)）。
2. 发送端 `ncclIbIsend` 检查 `ctsFifo[slot]` 中各项的 `idx` 是否等于 `fifoHead + 1`，以此判断 CTS 是否已到达（没到就直接返回，等下次调用），然后调 `ncclIbMultiSend`：
   - 每个请求一个 `IBV_WR_RDMA_WRITE`（不 signal），`remote_addr/rkey` 取自 CTS；
   - 最后串一个 `IBV_WR_RDMA_WRITE_WITH_IMM` 并 signal，`imm_data` 在 BY_ID 匹配模式下是 request id，否则是 size。多接收（nreqs>1）或 AR 大消息（双方不都支持 OOO RQ 时）情况下，它会作为一个独立的 WR 追加在数据 WRITE 之后。其中 nreqs>1 时，这个 WR 携带 `nreqs × sizeof(int)` 的各请求 size，写进对端的 completion records；AR 单请求时它不带数据（`num_sge` 保持 0），只负责送 IMM；
   - **多 QP 切分**：一条请求铺到 `nqps` 个 QP 上（`NCCL_IB_SPLIT_DATA_ON_QPS=1` 时用全部 QP，否则每块设备一个，见 `common.h`：`ncclIbCommBaseGetNqpsPerRequest`）。QP 按 `(id * nQps + qpIndex) % nqps` 轮转（`ncclIbCommBaseGetQpForRequest`）。每个 QP 分到的字节数 = size × 设备权重 / 100 / 该设备 QP 数，按 `IB_WRITE_CHUNK_ALIGNMENT`（128 B，`p2p.cc`）对齐；权重来自设备速率（`ncclIbComputeLbWeights`）。
3. 接收端用 `ncclIbTest` 中的 `wrap_ibv_poll_cq` 拿到 `IBV_WC_RECV_RDMA_WITH_IMM`，所有 QP 的事件都到齐后请求完成。如果 buffer 在 GPU 上，再走 `ncclIbIflush`。

这就是"接收端驱动的 rendezvous"：接收端先告诉发送端写到哪里，发送端做纯单边 WRITE，不需要中间拷贝；CTS 机制本身也天然起到流控作用，接收端没准备好时发送端不会发。

### 相关环境变量（只列在源码中 grep 到的）

`NCCL_PARAM(name, "X", default)` 展开后读取的环境变量是 `NCCL_X`（`src/include/param.h`：`ncclLoadParam("NCCL_" env, ...)`）。

| 变量 | 默认 | 定义位置 | 作用 |
|---|---|---|---|
| `NCCL_IB_DISABLE` | 0 | net_ib/init.cc | 禁用 IB transport |
| `NCCL_IB_HCA` | – | net_ib/init.cc（`ncclGetEnv`） | 选网卡；`^` 前缀表示排除，`=` 前缀表示精确匹配；逗号分隔，每项格式为 `prefix[:port[:rail[:plane]]]`（`src/misc/utils.cc`：`parseStringList`） |
| `NCCL_IB_MERGE_NICS` / `NCCL_IB_MERGE_VFS` / `NCCL_IB_SORT_MERGE_NICS` / `NCCL_IB_DEVICE_PCI_ORDER` | 1 | net_ib/init.cc | 多端口 / VF 合并为虚拟设备，以及排序方式 |
| `NCCL_IB_RAIL_POLICY` | – | net_ib/init.cc | rail / plane 分配策略 |
| `NCCL_IB_ADAPTIVE_ROUTING` | -2（IB 开，RoCE 关） | net_ib/init.cc | 自适应路由 |
| `NCCL_IB_AR_THRESHOLD` | -2（内部默认 8192） | net_ib/p2p.cc | AR 下大于此值时数据与 IMM 分开发送 |
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2 | net_ib/init.cc | MR 是否加 RELAXED_ORDERING |
| `NCCL_IB_DATA_DIRECT` | 1 | net_ib/init.cc | mlx5 Data Direct dmabuf |
| `NCCL_IB_OOO_RQ` | 0 | net_ib/init.cc | 用 `MLX5DV_QP_CREATE_OOO_DP` 建 QP |
| `NCCL_IB_GID_INDEX` | -1 | net_ib/connect.cc | 强制 GID index |
| `NCCL_IB_ROUTABLE_FLID_GID_INDEX` | 1 | net_ib/connect.cc | IB 跨子网 FLID 用的 GID index |
| `NCCL_IB_ROCE_VERSION_NUM` | 2 | net_ib/connect.cc | 自动选 GID 时期望的 RoCE 版本 |
| `NCCL_IB_ADDR_FAMILY` / `NCCL_IB_ADDR_RANGE` | – | net_ib/connect.cc | 自动选 GID 时的地址族 / 网段 |
| `NCCL_IB_SUBNET_AWARE_ROUTING` / `NCCL_IB_SUBNET_PREFIX_LEN` | 0 / 24 | net_ib/connect.cc | 按对端子网挑本地网卡 |
| `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT` | 20 / 7 | net_ib/connect.cc | QP ACK 超时指数 / 重传次数 |
| `NCCL_IB_PKEY` / `NCCL_IB_PKEY_VALUE` | 0 / -1 | net_ib/connect.cc | 分区 |
| `NCCL_IB_SL` / `NCCL_IB_TC` / `NCCL_IB_FIFO_TC` | -1 | net_ib/connect.cc | 服务等级 / RoCE traffic class / CTS 用的 TC |
| `NCCL_IB_USE_INLINE` | 0 | net_ib/connect.cc | CTS inline |
| `NCCL_IB_ECE_ENABLE` | 1 | net_ib/connect.cc | ECE 协商 |
| `NCCL_IB_QPS_PER_CONNECTION` | 1 | net_ib/connect.cc | 每连接每设备的 QP 数 |
| `NCCL_IB_WARN_RAIL_LOCAL` | 0 | net_ib/connect.cc | rail-local 告警 |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | 0 | net_ib/common.cc | 一条请求是否铺到所有 QP |
| `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` | -2 | net_ib/common.cc | 预投 RECV 策略 |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | 1 | net_ib/common.cc | 异步事件（端口 down、GID 变化等）上报 |
| `NCCL_IB_EVENT_BASED_LB` / `NCCL_IB_EVENT_BASED_LB_REMOTE` | 0 / 1 | net_ib/common.cc | 基于事件（如速率变化）的负载均衡 |
| `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` | -2 | net_ib/p2p.cc | 接收端匹配方式（BY_INDEX / BY_ID） |
| `NCCL_IB_WQE_LATENCY_THRESHOLD_NS` / `NCCL_IB_WQE_LATENCY_REPORT` | 0 / 1 | net_ib/wqe_lat_mon.cc | WQE 延迟监控 |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER`（及 `_MAX_ATTEMPTS`、`_PROBE_DELAY`） | 0 | net_ib/p2p_resiliency.cc | 端口故障切换 |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY`（及 `_START_DELAY`、`_ALIVE_MSG_*`、`_ACK_TIMEOUT`、`_ATTEMPTS_MAX`） | 0 | net_ib/p2p_resiliency_recovery.cc | 端口恢复 |
| `NCCL_IB_MQP_RETRY_ALL` / `NCCL_IB_MQP_RETRY_CNT` / `NCCL_IB_MQP_RETRY_SLEEP_MSEC` | 0 / 34 / 100 | misc/ibvwrap.cc | modify_qp 失败重试 |
| `NCCL_IB_QUERY_PORT_SPEED` | 1 | misc/ibvwrap.cc | 查询端口速率 |
| `NCCL_GDR_FLUSH_DISABLE` | 0 | net_ib/connect.cc | 关闭 GDR flush |
| `NCCL_NET_GDR_LEVEL` | 自动（PXB） | graph/paths.cc（`ncclGetLevel`） | GDR 允许的最大 GPU-NIC 距离 |
| `NCCL_NET_GDR_READ` / `NCCL_NET_GDR_C2C` | -2 / 1 | graph/paths.cc | 发送端 GDR read 开关 / C2C 放宽 |
| `NCCL_NET_FORCE_FLUSH` | 0 | graph/paths.cc | 强制 flush |
| `NCCL_DMABUF_ENABLE` | 1 | init.cc | DMA-BUF 总开关 |

## 源码导读（路径 + 函数 + 作用）

| 文件 | 关键函数 | 作用 |
|---|---|---|
| `src/transport/net_ib/common.cc` | `ncclNetIb`（函数表）、`ncclIbAsyncThreadMain` | 插件入口；异步事件线程（GID 变化、速率变化） |
| `src/transport/net_ib/init.cc` | `ncclIbInitDevices`、`ncclIbMakeVDevice`、`ncclIbGetPhysProperties` | 枚举 / 过滤 / 合并网卡，上报 ptrSupport、速度、maxQp |
| `src/transport/net_ib/gdr.cc` | `ncclIbGdrSupport`、`ncclIbPeerMemSupport`、`ncclIbDmaBufSupport` | GDR 能力探测 |
| `src/transport/net_ib/connect.cc` | `ncclIbListen`、`ncclIbConnectImpl`、`ncclIbAcceptImpl`、`ncclIbQpCreate`、`ncclIbQpInit/Rtr/Rts`、`ncclIbCreateFlushQp`、`ncclIbGetGidIndex` | 建链状态机与 QP 状态迁移 |
| `src/transport/net_ib/reg.cc` | `ncclIbRegMrDmaBufInternal2`、`ncclIbDeregMrInternal` | MR cache、dmabuf / relaxed ordering |
| `src/transport/net_ib/p2p.cc` | `ncclIbIsend`、`ncclIbMultiSend`、`ncclIbIrecv`、`ncclIbPostFifo`、`ncclIbIflush`、`ncclIbTest` | 数据面 |
| `src/transport/net_ib/common.h` | `ncclIbCommBaseGetNqpsPerRequest`、`ncclIbCommBaseGetQpForRequest`、`ncclIbComputeLbWeights` | 多 QP 分配与权重 |
| `src/transport/net_ib/p2p_resiliency*.cc` | – | 端口故障切换 / 恢复（RDMA READ 探测，UC QP 发 alive 消息） |
| `src/transport/net.cc` | `ncclTopoCheckGdr` 的调用点、`regMrDmaBuf` / `regMr` 分支 | NCCL 核心如何使用 net 插件 |
| `src/graph/paths.cc` | `ncclTopoCheckGdr`、`ncclTopoNeedFlush` | GDR 与 flush 策略 |
| `src/misc/ibvwrap.cc`、`src/misc/ibvsymbols.cc` | `wrap_ibv_*` | 动态加载 verbs、统一错误处理 |

## 性能要点与常见坑

- **看日志确认 GDR**：`NCCL_DEBUG=INFO` 时，`net.cc` 会在连接日志里打印 `/GDRDMA`（C2C 时附带 `(PCI)`）和 `flush=` 类型；`topo.cc` 打印 "GPU Direct RDMA Enabled/Disabled for HCA"。没有 GDRDMA 就意味着数据要经过主机内存中转。
- **GDR 被关掉的常见原因**：没装或没加载 nvidia-peermem，且 dmabuf 不可用（内核或驱动太旧）；GPU 与 NIC 跨 CPU（距离大于 PXB）；ACS/IOMMU 让 P2P 走 root complex；容器里 sysfs 不可见（`ncclIbRoceGetVersionNum` 对 EINVAL 有专门的注释）。
- **混用 IB 和 RoCE 网卡**会在建链时直接报错，要用 `NCCL_IB_HCA` 选定一种。
- **RoCE 选错 GID** 是最常见的建链失败 / 性能问题，先用 `NCCL_IB_GID_INDEX` 显式指定排除。
- **单 QP 带宽不足**时加 `NCCL_IB_QPS_PER_CONNECTION`。它同时增加 ECMP 熵，但也会让 QP 总数（rank 对 × channel × QP）膨胀，带来网卡 QP context 缓存压力（见 [03-roce-ib-congestion.md](03-roce-ib-congestion.md)）。
- **timeout 太小**：大规模拥塞时出现 `IBV_WC_RETRY_EXC_ERR`（transport retry counter exceeded）。NCCL 的报错格式是 `"NET/IB: Got completion from peer %s with status=%s(%d) opcode=..."`（`p2p.cc`：`ncclIbLogCompletionWithError`）。看到 status 为 retry exceeded 时，先查链路和拥塞，再考虑调大 `NCCL_IB_TIMEOUT`。
- **flush 的开销**：Ampere 及以前每次 GPU 接收都多一次 RDMA READ；Hopper 默认省掉。

## 面试自测（问题 + 要点答案）

1. **GPUDirect RDMA 解决什么问题？有哪两种内核实现方式？**
   网卡直接读写显存，省掉 GPU↔主机内存的拷贝和这部分主机内存带宽占用。nvidia-peermem 通过 peer memory 接口，让 `ibv_reg_mr` 能直接注册 GPU 指针；dmabuf 是上游标准接口，CUDA 导出 fd 后调 `ibv_reg_dmabuf_mr`。NCCL 两种都会探测，优先用 dmabuf。
2. **NCCL 怎么决定某条连接用不用 GDR？**
   NIC 和 GPU 都支持；发送端 GDR read 在老架构且无 NVLink 时关闭；GPU-NIC 拓扑距离不超过 `NCCL_NET_GDR_LEVEL`（默认 PXB）；C2C 系统有特殊处理。代码见 `ncclTopoCheckGdr`。
3. **GDR flush 是什么？为什么 Hopper 不需要？**
   网卡写显存的 PCIe 写可能还没对 GPU 可见，用一个自环 QP 的 RDMA READ 读一下目标 buffer，以确保可见性。Hopper 及以后 NCCL 默认不 flush（`ncclTopoNeedFlush` 对 `cudaCompCap >= 90` 的判断），说明硬件或驱动已经保证了顺序；Data Direct（C2C 控制路径 + PCIe 数据路径）仍然需要 flush。
4. **描述 NCCL IB 一次 send/recv 的完整 verbs 序列。**
   接收端 irecv：填 SendFifo，RDMA WRITE(可 inline) 写到发送端的 ctsFifo。发送端 isend：读到 CTS 后，对每个 QP post 多个 RDMA WRITE + 一个 signaled WRITE_WITH_IMM（数据按速率权重切分到多 QP）。接收端：预投的 RECV 被 IMM 消耗，poll 到 `RECV_RDMA_WITH_IMM`；如果是 GPU buffer 再做 iflush。test 中 poll CQ 判断完成。
5. **NCCL 的 MR 为什么要做 cache？注册的权限是什么？**
   reg_mr 要 pin 页并写网卡翻译表，开销远大于一次 post（大 buffer 尤其明显），不能放在数据路径上。待确认：具体耗时；确认方式：在目标机器上对 1 MiB/1 GiB buffer 计时 `ibv_reg_mr`。按页对齐后做区间命中 + 引用计数。权限是 LOCAL_WRITE、REMOTE_WRITE、REMOTE_READ、REMOTE_ATOMIC，可选加 RELAXED_ORDERING。
6. **NCCL 建链交换了哪些信息？与 rc_pingpong 有何不同？**
   设备列表（虚拟设备的组成、OOO 能力），然后是 meta：每个 QP 的 QPN（以及 ECE）、每个设备的 lid/mtu/GID/link_layer/rkey、CTS FIFO 地址、SL/TC。PSN 固定为 0；用异步状态机配合 proxy 轮询推进；一个逻辑连接对应多块网卡、多条 QP。
7. **如果底层换成乱序多路径传输，NCCL net_ib 哪些地方要改？**
   WRITE→IMM 的顺序假设（已有 OOO RQ、completion records、BY_ID matching 可以利用）；flush 语义；多 QP 切分逻辑可以简化为单 QP 加网卡侧喷洒；QP 状态机和 resiliency 可以交给协议层处理。

## 待深入

- `src/transport/net_ib/gdaki/`（doca-gpunetio，GPU 发起的 verbs，即 IBGDA 路线），以及与 ai-infra-book 6.4.4 节"GPU 的 SM 直接构造请求、敲门铃"的对应关系（`third_party/ai-infra-book/manuscripts/06-超节点.md`）。
- `src/transport/net_ib/gin.cc`（GIN：`IBV_WR_RDMA_WRITE/READ` 加原子信号）与 NCCL 设备端 API 的关系。
