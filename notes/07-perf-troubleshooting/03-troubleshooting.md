# NCCL 故障排查清单

> 层次：实践 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3；`src/ras/`、`src/diagnostics*`、`src/transport/net_ib/`，以及 `docs/userguide/source/troubleshooting/{ras,diagnostics,networking_troubleshooting,gpu_troubleshooting,runtime_and_mpi_issues}.rst`）、ai-infra-book@3bdcb4fc 第 7 章 7.6.1 节

## 一句话

按"症状 → 只读检查 → 可能原因 → 处理"排查：先用只读手段（日志、RAS、诊断、链路计数器）把问题定位到 rank、层次（init/调用不匹配/scale-up/网络/硬件）和具体资源，再做最小变更验证；不要一上来就改环境变量或重启集群。

## 在调用链中的位置

- 机制背景：init 与 bootstrap（[../04-nccl-internals/01-init-bootstrap.md](../04-nccl-internals/01-init-bootstrap.md)）、kernel 自旋等待（[../04-nccl-internals/03-channels-protocols-kernels.md](../04-nccl-internals/03-channels-protocols-kernels.md)）、proxy 与 IB（[../04-nccl-internals/04-proxy-and-net-plugin.md](../04-nccl-internals/04-proxy-and-net-plugin.md)、[../01-rdma/README.md](../01-rdma/README.md)）、P2P/SHM/NVLS（[../02-scale-up/02-nccl-p2p-shm-nvls.md](../02-scale-up/02-nccl-p2p-shm-nvls.md)）。
- 环境变量定义：[01-env-vars.md](01-env-vars.md)；性能问题方法：[02-perf-methodology.md](02-perf-methodology.md)。

## 核心概念

### NCCL 的错误与"hang"语义

- 返回码（`runtime_and_mpi_issues.rst` "Errors" 节）：`ncclUnhandledCudaError`、`ncclSystemError` 表示外部库（CUDA、系统调用、verbs）失败；`ncclInvalidArgument`、`ncclInvalidUsage` 表示应用用法错误；`NCCL_DEBUG=WARN` 会在返回前打印具体原因。
- 异步错误：proxy 线程出错时写入 `proxyState->asyncResult` 并退出循环（`src/proxy.cc::ncclProxyProgress`）；应用需通过 `ncclCommGetAsyncError` 轮询，非阻塞 comm 的错误也经此返回（`src/enqueue/enqueue.cc::ncclEnqueueCheck` 末尾）。
- NCCL 不给集合操作设整体超时：GPU kernel 在 primitives 的 `waitPeer` 中自旋，每 `NCCL_SPINS_BEFORE_CHECK_ABORT`（10000）次检查一次 abort 标志（`src/device/primitives.h`）。"timeout" 通常来自三处：框架看门狗（如 PyTorch，不在本仓库）、IB 传输层重传耗尽（`NCCL_IB_TIMEOUT`/`NCCL_IB_RETRY_CNT`）、bootstrap/socket 连接重试（`NCCL_SOCKET_RETRY_CNT` 等）。
- 中止：`ncclCommAbort`（`src/init.cc:3489`）设置 abort 标志让 kernel 和 proxy 退出；`ncclCommRevoke`、`ncclCommShrink` 用于容错场景下剔除故障 rank。

把上面几条放在一起：NCCL 自己不会"超时"，能让一次等待结束或暴露出来的只有这几条路径。

```text
  GPU kernel (primitives.h)
    waitPeer spin ──► every NCCL_SPINS_BEFORE_CHECK_ABORT (10000) spins: check abort flag
                                                                          ▲
  framework watchdog (e.g. PyTorch, not in this repo)                     │
    collective not done in time ──► ncclCommAbort ──► set abort flag ─────┘

  IB transport layer
    retransmit exhausted (NCCL_IB_TIMEOUT, NCCL_IB_RETRY_CNT)
      ──► WARN "NET/IB: Got completion from peer ... status=..."
      ──► proxy progressOps returns error
      ──► ncclProxyProgress stores proxyState->asyncResult, leaves loop
      ──► app sees it only via ncclCommGetAsyncError

  bootstrap / socket (init)
    connect retries exhausted (NCCL_SOCKET_RETRY_CNT ...) ──► init returns error
```

### RAS 子系统（`src/ras/`）

- `ras.rst`：NCCL 2.24 起提供，init 时启动（`NCCL_RAS_ENABLE` 默认 1，`src/ras/ras.cc:91`），每进程一组线程，经 bootstrap 使用的 OOB 网卡建立 TCP 连接、互发 keep-alive；进程崩溃或卡死时，其他进程的 RAS 线程能通过连接断开或无响应感知。
- 查询：RAS 线程在 `localhost:28028` 监听（`NCCL_RAS_CLIENT_PORT`，`src/ras/ras_internal.h:13`；`NCCL_RAS_ADDR` 可改），用 `ncclras`（`-v` 详细、`-t` 超时、`-f json`、`-m` 监控、`-D` 诊断）或 `echo verbose status | nc localhost 28028`。输出包含作业摘要（节点、进程、GPU 数）与各 communicator 的状态和错误（`ras.rst` "Sample Output"）。
- 诊断（`diagnostics.rst`）：RAS 诊断被动比较各 rank 的 GPU、驱动与 NCCL 配置（`ncclras -D` 或 `NCCL_RUN_RAS_DIAGNOSTICS`）；主动诊断 `NCCL_RUN_DIAGNOSTICS=1` 在 comm init 时实际验证 P2P 访问、IB 带宽等路径，rank 0 输出 `NCCL DIAG [OK]/[INFO] ...`（实现：`src/diagnostics.cc`、`src/diagnostics/p2p.cc`、`src/diagnostics/ib_write_bw.cc`、`src/ras/diagnostics_{gpu,pci,paths,env,xid}.cc`）。
- 进度计数：`NCCL_PROGRESS_COUNTERS=1` 且 RAS 开启时（`src/init.cc:805`），`src/ras/progress_monitor.cc` 按 `NCCL_PROGRESS_COUNTER_MONITOR_POLL_MS`（1000）轮询设备侧进度计数镜像，供 RAS 判断谁在前进。

RAS 的结构：每个进程一组 RAS 线程，彼此经 OOB 网卡的 TCP 连接互发 keep-alive；运维在任一节点用客户端连本机端口即可拿到全局视图。

```text
  node A                                   node B
  ┌──────────────────────────┐             ┌──────────────────────────┐
  │ rank 0..7 processes      │             │ rank 8..15 processes     │
  │   NCCL comms             │             │   NCCL comms             │
  │   RAS threads ◄──────────┼─ TCP over ──┼─► RAS threads            │
  │     listen :28028        │  OOB NIC    │     listen :28028        │
  └────────────▲─────────────┘ keep-alive  └──────────────────────────┘
               │
   ncclras -v  /  echo verbose status | nc localhost 28028
   → job summary (nodes, processes, GPUs) + per-communicator state / errors
   peer crashed or stuck → connection drop / no response seen by other RAS threads
```

## 排查清单

先按症状分流到下面 A~E 五张表，再在表内按"只读检查"一列逐项排除：

```text
  symptom
  │
  ├─ never reaches "Init COMPLETE" / init returns error ───────────► A. init
  │
  ├─ running, GPU 100%, progress stopped
  │    ├─ all ranks in NCCL? (ncclras -v, host stacks, COLL opCount)
  │    │    ├─ no:  some rank late / crashed / different call ─────► B. hang (app side)
  │    │    └─ yes: only cross-node ranks stuck ───────────────────► B + D (proxy / IB)
  │    └─ (may also be readiness skew: slow compute / IO on one rank)
  │
  ├─ works but slower than before ─────────────────────────────────► C. perf drop
  │    └─ log shows NET/IB WARN or async event ────────────────────► D. IB / RoCE
  │
  └─ #wrong != 0 or loss abnormal ─────────────────────────────────► E. data error
```

### A. init 阶段卡住或失败

| 症状 | 只读检查 | 可能原因 | 处理 |
|---|---|---|---|
| 停在 `Init START` 之后，无 `Bootstrap: Using` | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV`；`ip addr`；对比各节点选中的接口 | 自动选到 docker0/lo/错误子网；`NCCL_SOCKET_IFNAME` 拼错 | 显式设置 `NCCL_SOCKET_IFNAME`（`^`/`=` 语法见 01-env-vars） |
| 部分 rank 长时间等待，无 `Init COMPLETE` | 各 rank 日志的 nranks/rank；unique id 是否同一个；`ncclras` 看哪些进程没加入 | 某 rank 没启动或崩溃；rank 编号重复；unique id 广播错误；防火墙阻断 TCP | 修正启动脚本；`net.ipv4.ip_local_port_range` 限定端口后开放防火墙（`networking_troubleshooting.rst` "IP ports"） |
| `Invalid NCCL_COMM_ID` | 环境变量值 | 格式不是 `<ipv4>:<port>`/`[<ipv6>]:<port>`/`<host>:<port>` | 修正格式（`src/bootstrap.cc:117`） |
| `No transport found for rank X -> rank Y` | `NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET` | 所有 transport 的 `canConnect` 都返回不可用：P2P 被禁、SHM 不同 `/dev/shm`、无可用网络插件/网卡 | 修正容器共享 `/dev/shm`、检查 `NCCL_P2P_DISABLE/SHM_DISABLE/IB_DISABLE` 组合（`src/transport.cc::selectTransport`） |
| `failed to extend /dev/shm/nccl-xxxx` | `df -h /dev/shm` | 共享内存不足（容器默认很小） | 增大 `/dev/shm`（`runtime_and_mpi_issues.rst` "Shared memory"） |
| `P2P is disabled between NVLINK connected GPUs` 并返回 `ncclUnhandledCudaError` | `nvidia-smi topo -m`、`nvidia-smi nvlink --status`、dmesg Xid | NVML 报告 NVLink 相连但 P2P 不可用，多为硬件/驱动问题 | 交给硬件排查；临时绕过 `NCCL_IGNORE_DISABLED_P2P=1`（`src/graph/paths.cc:420` 附近）需评估风险 |
| 网卡数不一致 | `Rank %d has %d local Net devices (max %d).`；默认（`NCCL_IGNORE_NET_MISMATCH=1`）打印 INFO `Detected mixed local Net device counts ... Ignoring due to NCCL_IGNORE_NET_MISMATCH.`，设为 0 时 WARN 并返回 `ncclSystemError`（`src/init.cc` AllGather3 之后） | 某节点网卡掉线或驱动未加载；默认忽略时可能表现为通道数变少、性能下降 | 修复节点；排查期可设 `NCCL_IGNORE_NET_MISMATCH=0` 让问题尽早暴露 |
| 外部网络插件未生效 | `Successfully loaded external network plugin` / `is unsupported` / `Initialized NET plugin` | 插件版本不兼容、路径不对，回落到 IB 或 Socket | 修正 `NCCL_NET_PLUGIN`/库路径；用 `NCCL_NET` 指定名字验证 |

### B. 运行中 hang（GPU 100%、进度停止）

| 症状 | 只读检查 | 可能原因 | 处理 |
|---|---|---|---|
| 所有 rank 卡在同一集合调用 | `ncclras -v`（或 `nc localhost 28028`）看 communicator 状态与缺失 rank；主机栈（py-spy/gdb，工具不在本仓库）；`NCCL_DEBUG_SUBSYS=COLL` 比对各 rank 最后几条 `opCount ... count ... datatype ... op ... root` | 某 rank 未调用（代码分支不一致、提前退出）、调用顺序不同、count/datatype/root 不一致 | 修正应用逻辑；所有 rank 必须以相同顺序调用同一 comm 的集合操作 |
| 在 group 内的 send/recv 卡住 | COLL 日志中 send/recv 对端与 count | send 与 recv 不配对、不同 rank 的 group 内容不一致 | 保证每个 send 有匹配 recv，并放在同一 group（`groups.rst`） |
| 只有跨节点的 rank 卡住 | `NCCL_PROXY_DUMP_SIGNAL=<signo>` 后 `kill -<signo> <pid>` 打印 proxy op 状态（`src/proxy.cc:922`）；IB 计数器 | proxy 线程出错退出或网络丢包导致 RDMA 卡在重传 | 看是否有 NET/IB WARN（见 D）；检查 `ncclCommGetAsyncError` 是否已报错 |
| 某 rank 进程已崩溃，其他 rank 一直等 | `ncclras` 中该进程无响应/断开 | 进程 OOM、段错误、被调度器杀掉 | 由框架看门狗 `ncclCommAbort`；容错场景用 `ncclCommShrink` 剔除 |
| 多 comm 交错使用后死锁 | 主机栈：多个线程/stream 上的 NCCL 调用顺序 | 不同 rank 对多个 comm 的调用顺序不一致；单线程多 GPU 未用 group | 统一调用顺序；单线程多 GPU 必须 `ncclGroupStart/End`（`groups.rst`） |
| CUDA graph replay 后卡住 | 是否所有 rank 都 replay；graph 内 NCCL 调用一致性 | 部分 rank 未 replay；host 回调（`hostStreamPlanCallback`）被阻塞 | 保证 replay 对称；检查 host 线程负载 |

ai-infra-book 7.6.1 提醒：一张卡早进入集合通信调用而别的卡还没就绪时，等待会算在通信耗时里。"看起来像 NCCL hang"的长等待也可能只是某 rank 计算/IO 很慢，先看各 rank 是否都已进入调用。

### C. 性能掉档（以前正常、现在慢）

| 症状 | 只读检查 | 可能原因 | 处理 |
|---|---|---|---|
| 节点内 busbw 骤降 | 日志 transport 是否从 `P2P/CUMEM` 变为 `SHM`；`nvidia-smi nvlink --status`；`nvidia-smi topo -m` | NVLink 链路降级/掉线；新版本/新容器禁用了 P2P；ACS 被打开 | 硬件排查；ACS 检查 `sudo lspci -vvv \| grep ACSCtl`（出现 `SrcValid+` 可能已开启，`gpu_troubleshooting.rst`），关闭需运维操作 |
| 跨节点 busbw 骤降 | 日志是否丢失 `/GDRDMA`；`lsmod \| grep nvidia-peermem`；`ibstat` Rate；`rdma statistic`；`perfquery -x <lid>` 或 RoCE 下 `ethtool -S <nic>` | GDR 失效（peermem 未加载且无 DMA-BUF）、端口降速、链路误码、拥塞/PFC 风暴、交换机故障 | 加载 `nvidia-peermem`（`gpu_troubleshooting.rst`，需 root）或确认 DMA-BUF 可用；更换线缆/端口；网络侧排查 |
| 只有个别节点慢 | nccl-tests `-a 2/3`、`-I 1`；逐对节点测试 | 单节点网卡/GPU/PCIe 问题、绑核错误、热降频 | 隔离节点；`mlxlink -d <mst_device>` 查链路宽度和速率 |
| 升级 NCCL 后变慢 | 对比新旧版本 `NCCL_DEBUG=INFO` 的通道数、算法/协议（TUNING） | 默认算法/协议/通道策略变化；新特性（NVLS、PAT、懒连接）行为不同 | A/B `NCCL_ALGO/PROTO/NVLS_ENABLE` 定位，再反馈或用 tuner 插件 |
| 与计算并发时变慢 | nsys 时间线 | SM/HBM 争用（6.4.3） | 调整 `NCCL_MAX_CTAS`/通道数，尝试 NVLS/CE 路径 |
| 随机慢、抖动大 | `-I 1` 看 `i_cv%`；IB 计数器随时间增长 | 网络拥塞、ECMP 冲突、PCIe 错误重试 | `NCCL_IB_QPS_PER_CONNECTION` 打散流；网络侧 QoS 检查 |

### D. IB / RoCE 链路问题

| 症状（日志） | 只读检查 | 可能原因 | 处理 |
|---|---|---|---|
| `NET/IB: Got completion from peer <ip> with status=<str>(<n>) opcode=... vendor_err=... qp_num=... hca ...`（`src/transport/net_ib/p2p.cc:785` 附近，`ncclIbLogCompletionWithError`） | 对端节点 `ibstat`、`rdma statistic` 中 `local_ack_timeout_err`、`packet_seq_err` 等；交换机端口 | 典型如重传耗尽（链路断、对端进程崩溃、严重丢包/拥塞） | 先排对端是否存活；物理链路/交换机；必要时调大 `NCCL_IB_TIMEOUT`（4.096us × 2^n）与 `NCCL_IB_RETRY_CNT` 作为缓解 |
| `NET/IB: Got CQE with error ...`（`p2p.cc` `ncclIbTest`） | 同上 | 同上，未开启 resiliency 时不可恢复 | 可评估 `NCCL_IB_RESILIENCY_PORT_FAILOVER=1`（默认 0，`p2p_resiliency.cc:13`） |
| `NET/IB : <dev>:<port> Got non-fatal async event: <str>`（`src/transport/net_ib/common.cc:230`） | `dmesg`、`ibstat`、SM 日志 | `PORT_ERR`、`LID_CHANGE`、`SM_CHANGE`、`PKEY_CHANGE` 等：端口抖动、子网管理器变化 | 检查 SM（`sudo sminfo`，`networking_troubleshooting.rst` "Subnet Manager"）与线缆 |
| modify QP 失败并出现 `HINT: ... NICs are not cross-rail connected`（`src/misc/ibvwrap.cc:343`，ETIMEDOUT） | 网络拓扑：同号网卡是否在同一 rail/交换机 | 多轨网络中跨 rail 不通 | 按提示设 `NCCL_CROSS_NIC=0` 验证 |
| modify QP 失败并出现 `HINT: ... incorrect GID index is forced by NCCL_IB_GID_INDEX, or ... GID changed mid-run`（`ibvwrap.cc` EINVAL 分支） | `show_gids`/`ibv_devinfo -v`（工具未在本地验证）；`dmesg \| grep -i gid` | RoCE GID 索引错误（IPv4/IPv6、RoCEv1/v2 混用），或运行中 IP 变化 | 按提示设 `NCCL_IB_GID_INDEX=-1` 自动检测；确认 `NCCL_IB_ROCE_VERSION_NUM` |
| `NET/Socket : peer ... message truncated ... mismatch in collective sizes or environment settings (e.g. NCCL_PROTO, NCCL_ALGO)`（`src/transport/net_socket.cc:522`） | 各 rank 的环境变量与调用参数 | 走了 Socket（IB 不可用）且各 rank 参数/环境不一致 | 统一环境变量；先查为何没用 IB |
| 带宽只有几 GB/s 且日志是 `NET/Socket` | `Initialized NET plugin`；`NCCL_IB_DISABLE`；`ibv_devinfo` | IB 插件初始化失败（驱动/权限/`NCCL_IB_HCA` 过滤掉全部设备）后回落 Socket | 修正 `NCCL_IB_HCA`（注意前缀匹配，精确用 `=`）、驱动与容器设备挂载 |

裸链路验证（`networking_troubleshooting.rst`）：`ibstatus`/`ibstat` 看端口状态和 Rate；`ib_write_bw -d <dev> -a`（服务端）与 `ib_write_bw -d <dev> <server> -a`（客户端）测带宽，`--use_cuda=<gpu>`（或 `--use_cuda_dmabuf`）测 GPU 显存路径；`ib_write_lat` 测延迟；`ibping` 测连通。perftest 源码在 `third_party/perftest`（本笔记未展开，见 [../01-rdma/README.md](../01-rdma/README.md)）。

### E. 数据错误（nccl-tests `#wrong` 非 0 或训练 loss 异常）

| 症状 | 只读检查 | 可能原因 | 处理 |
|---|---|---|---|
| `#wrong` 非 0 | 换 `NCCL_PROTO`/`NCCL_ALGO`、单机/跨机分别测；`NCCL_RUN_DIAGNOSTICS=1` | P2P 在 IOMMU/ACS 下行为异常、NVLink/PCIe/网卡硬件错误、GDR flush 问题 | IOMMU 检查 `dmesg \| grep -i -E "iommu\|dmar\|default domain"`（`gpu_troubleshooting.rst`）；隔离硬件；不要用强制变量掩盖 |
| 只在 LL128 下出错 | `NCCL_PROTO=^LL128` 对比 | 平台不保证 128B 写入顺序（LL128 依赖该性质，见 03-channels） | 上报；临时禁用 LL128 |
| 只在 GDR 下出错 | `NCCL_NET_GDR_LEVEL=LOC` 对比、`NCCL_NET_FORCE_FLUSH=1` 对比 | PCIe 写可见性/flush 问题（`ncclTopoNeedFlush`） | 硬件/固件排查 |

## 源码导读（排查时常翻的位置）

| 路径 | 内容 |
|---|---|
| `src/ras/ras.cc`、`client.cc`、`client_support.cc`、`peers.cc`、`rasnet.cc` | RAS 主线程、客户端协议、peer 管理、RAS 网络 |
| `src/ras/progress_monitor.cc` | 进度计数监控 |
| `src/diagnostics.cc`、`src/diagnostics/` | 主动诊断（P2P、IB 带宽） |
| `src/proxy.cc::ncclProxyProgress`、`dumpProxyState` | proxy 异步错误与状态 dump |
| `src/transport/net_ib/common.cc`（异步事件）、`p2p.cc`（CQE 错误）、`p2p_resiliency*.cc`（端口故障切换） | IB 错误路径 |
| `src/misc/ibvwrap.cc::printIbModifyQpHint` | QP 状态迁移失败的提示 |
| `src/init.cc::ncclCommAbort`、`ncclCommRevoke`、`ncclCommShrink` | 中止与容错 |

## 性能要点与常见坑

- 不要在 hang 时直接重启：先 `ncclras -v`、dump proxy、抓主机栈，信息只在现场。
- `NCCL_DEBUG=INFO` 产生大量日志，按子系统过滤并写文件（`NCCL_DEBUG_FILE=...%h.%p`）。
- 环境变量在进程内首次读取后缓存（`NCCL_PARAM` 实现），"改了没生效"多因设置太晚（NCCL 已读过并缓存）。配置文件不会覆盖已设置的环境变量（`src/os/linux.cc::ncclOsSetEnv` 用 `setenv(name, value, 0)`），但会为未设置的变量补值：先读用户配置（`NCCL_CONF_FILE`，未设则 `~/.nccl.conf`），再读 `/etc/nccl.conf`（`src/misc/param.cc::initEnvFunc`），所以"没设却生效了某个值"要查这两个文件。
- 强制变量（`NCCL_IGNORE_DISABLED_P2P`、`NCCL_PROTO` 等）只能用于定位，不能当修复。
- 改 ACS、加载内核模块、调整端口范围等属于系统变更，需要运维授权。

## 面试自测

1. 问：训练跑到一半 hang 住，所有 GPU 利用率 100%，你怎么排查？
   答：GPU 100% 说明 NCCL kernel 在 `waitPeer` 自旋等对端。先用 `ncclras -v` 看各 communicator 状态和缺失/无响应进程；抓所有 rank 主机栈，看是否有 rank 不在 NCCL 调用中或在不同调用中；开 `NCCL_DEBUG_SUBSYS=COLL` 比对 opCount 和参数；跨节点时 dump proxy 状态、查 IB WARN 和计数器；最后定位到代码分支不一致、某 rank 崩溃、或网络故障。
2. 问：日志出现 `NET/IB: Got completion from peer ... status=...` 应该怎么处理？
   答：这是 IB 工作完成带错误，通常是重传耗尽：先确认对端进程/节点是否存活（对端崩溃也会导致本端报错），再查物理链路、交换机、`rdma statistic` 中 `local_ack_timeout_err` 等计数；拥塞环境下可评估调大 `NCCL_IB_TIMEOUT/RETRY_CNT`，或开启 port failover；真正的故障需替换硬件。
3. 问：NCCL 有超时机制吗？PyTorch 的 NCCL timeout 是谁实现的？
   答：NCCL 本身不对集合操作计时，kernel 无限等待，只周期性检查 abort 标志；IB 层有传输重传超时，bootstrap 有 socket 重试。PyTorch 的超时是框架的看门狗线程检测 CUDA event 未完成后调用 `ncclCommAbort`（PyTorch 实现不在本仓库，待确认具体文件）。
4. 问：RAS 能告诉你什么？与 `NCCL_DEBUG` 的区别？
   答：RAS 是运行时常驻、低开销的全局健康视图：作业有多少节点/进程/GPU，各 communicator 是否 RUNNING、是否有 rank 缺失或无响应、错误汇总，可在不重启的情况下查询；`NCCL_DEBUG` 是各进程的本地日志，需要预先开启且要人工比对。
5. 问：RoCE 集群报 modify QP 失败，可能原因？
   答：本地源码给出的提示：ETIMEDOUT 多见于网卡不是 cross-rail 互通（试 `NCCL_CROSS_NIC=0`）；EINVAL 多见于 `NCCL_IB_GID_INDEX` 强制了错误的 GID 或运行中 GID 变化（试 `-1` 自动检测、查 dmesg）。

## 待深入

- 待确认：PyTorch ProcessGroupNCCL 看门狗与 `TORCH_NCCL_*` 变量的实现位置；确认方式：在 PyTorch 源码中 grep `ncclCommAbort`（本仓库 `third_party/` 未包含 PyTorch，可从 sglang/vllm 调用侧间接追踪，见 05-framework-integration）。
- 待确认：`ncclras` 文本输出中各状态字段（如 `RUNNING`、错误列）的完整取值；确认方式：读 `third_party/nccl/src/ras/client_support.cc` 中状态字符串与 `docs/userguide/source/troubleshooting/ras.rst` 全文。
- IB resiliency（`p2p_resiliency.cc`、`p2p_resiliency_recovery.cc`）的端口切换与恢复流程。
- `ncclCommRevoke`/`ncclCommShrink` 在弹性训练中的使用方式（`src/init.cc:3385` 起）。
