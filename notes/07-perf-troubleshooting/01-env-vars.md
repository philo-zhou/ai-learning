# NCCL 关键环境变量（按源码定义位置整理）

> 层次：实践 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）源码 grep；补充参考 `third_party/nccl/docs/userguide/source/env.rst`

## 一句话

NCCL 的环境变量几乎都在源码里通过 `NCCL_PARAM(Name, "KEY", default)`（自动加 `NCCL_` 前缀）、`DEFINE_NCCL_PARAM(...)` 或直接 `ncclGetEnv("NCCL_...")` 读取；知道定义位置就能看到默认值、取值含义和生效时机。

## 在调用链中的位置

- 读取入口：`src/include/param.h`（`NCCL_PARAM` 宏 → `ncclLoadParam`）、`src/include/param/param.h`（`DEFINE_NCCL_PARAM` 模板，带解析器与说明）、`src/misc/param.cc`（`ncclGetEnv`、配置文件）。
- 来源优先级与文件：`src/misc/param.cc::initEnvFunc` 读取 `NCCL_CONF_FILE`（未设则 `~/.nccl.conf`）和 `/etc/nccl.conf`，以 `KEY=VALUE` 写入进程环境；另有 env 插件（`src/plugin/env.cc`，`ncclEnvPluginInit`）。
- 查看全部参数：`ncclparam -l`（`src/param/ncclparam.cc`），`-a` 含私有参数（等价设置 `NCCL_PARAM_DUMP_ALL`）。
- 本笔记被 [02-perf-methodology.md](02-perf-methodology.md)、[03-troubleshooting.md](03-troubleshooting.md) 引用。

一个值从哪里来、何时定住（`src/misc/param.cc` 的 `initEnvFunc`/`setEnvFile`/`ncclLoadParam`，`src/os/linux.cc::ncclOsSetEnv`）。配置文件经 `setenv(name, value, 0)` 写入，**不覆盖已存在的变量**，所以优先级是：进程环境 > 用户配置文件 > `/etc/nccl.conf`。

```text
  1. process env (export / launcher)          ── always wins
  2. NCCL_CONF_FILE, else ~/.nccl.conf        ── read first, fills unset keys only
  3. /etc/nccl.conf                           ── read last,  fills unset keys only
                    │  (initEnvFunc runs once, std::call_once)
                    ▼
  ncclGetEnv(name) ──► env plugin: ncclEnvPluginGetEnv   (default plugin = std::getenv)
                    │
                    ▼
  ncclParamXxx()  ──► ncclLoadParam: first call parses and caches the value
                      later calls return the cache  (NCCL_NO_CACHE: per-key opt-out)
```

## 核心概念

- 缓存语义：`NCCL_PARAM` 生成的 `ncclParamXxx()` 第一次调用时读取并缓存（`static int64_t cache`），之后在进程内不再变化；`NCCL_NO_CACHE`（`src/param/param.cc:20`）可对指定 key 关闭缓存。结论：环境变量要在第一次调用 NCCL 之前设好。
- 默认值 `-2` 常表示"自动/由 NCCL 决定"（如 `NCCL_CUMEM_ENABLE`、`NCCL_P2P_READ_ENABLE`、`NCCL_BUFFSIZE`）；`NCCL_CONFIG_UNDEF_INT` 表示"未设置，由 `ncclConfig_t` 或默认值决定"，环境变量会覆盖 comm config（`init.cc::envConfigOverride` 中打印 `Comm config ... reset to NCCL_...`）。
- 生效检查：大部分变量在 `NCCL_DEBUG_SUBSYS=ENV`（默认包含）下会打印 `... set by environment to ...`。

## 源码导读：变量速查表

路径均相对 `third_party/nccl/src/`。

下表按主题分成 8 节；先用一张图看每节变量作用在 NCCL 的哪一层（sec 列为本文小节号）：

```text
  layer                         typical variables                                    sec
  ─────────────────────────     ─────────────────────────────────────────────────    ───
  comm config / misc            NCCL_CONF_FILE, COMM_BLOCKING, CHECK_POINTERS        7
        │
  algo / proto / channel        ALGO, PROTO, MIN/MAX_NCHANNELS, *_CTAS, BUFFSIZE     2
  selection (tuning)            TUNER_PLUGIN, NTHREADS, P2P_*_CHUNKSIZE
        │
  topology and graph            TOPO_FILE, TOPO_DUMP_FILE, GRAPH_FILE                6
        │
        ├── scale-up transport  P2P_LEVEL/DISABLE, SHM_*, NVLS_*, CUMEM_*, MNNVL_*   3
        │
        └── scale-out transport NET, NET_PLUGIN, NET_GDR_*, PXN_DISABLE, CROSS_NIC   4
              │                 bootstrap: SOCKET_IFNAME, COMM_ID, SOCKET_RETRY_*    4
              └── IB / RoCE     IB_HCA, IB_GID_INDEX, IB_TC/SL, IB_TIMEOUT, IB_QPS_* 5

  cross-cutting: debug / log (DEBUG, DEBUG_SUBSYS, DEBUG_FILE)                       1
                 RAS / diagnostics / profiler (RAS_*, RUN_DIAGNOSTICS, PROFILER_*)   7
  (names above omit the NCCL_ prefix)
```

### 1. 调试与日志

| 变量 | 定义位置 | 默认 | 说明 |
|---|---|---|---|
| `NCCL_DEBUG` | `debug.cc:43`（`DEFINE_NCCL_PARAM`） | NONE | `VERSION/WARN/ATTN/INFO/ABORT/TRACE`，逐级包含 |
| `NCCL_DEBUG_LEVELS` | `debug.cc:54` | 0 | 在 `NCCL_DEBUG` 之上追加级别（逗号分隔） |
| `NCCL_DEBUG_SUBSYS` | `debug.cc:68` | `INIT,BOOTSTRAP,ENV` | `INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC,ALLOC_HOST,CALL,PROXY,NVLS,BOOTSTRAP,REG,PROFILE,RAS,DESTROY,ALL` |
| `NCCL_DEBUG_FILE` | `debug.cc:117` | 无 | 日志写文件，`%h` 主机名、`%p` pid；不接受 `~` |
| `NCCL_DEBUG_TIMESTAMP_LEVELS` / `_FORMAT` | `debug.cc:97`、`:113` | WARN,ATTN / `"[%F %T] "` | 哪些级别带时间戳 |
| `NCCL_WARN_ENABLE_DEBUG_INFO` | `debug.cc:93` | false | 出现 WARN 后自动把级别提到 INFO |
| `NCCL_SET_THREAD_NAME` | `debug.cc:479` | false | 给 NCCL 线程命名，便于 top/gdb |
| `NCCL_NVTX_DISABLE` | `init_nvtx.cc:16` | 0 | 关闭 NCCL 的 NVTX 标注 |
| `NCCL_PROXY_DUMP_SIGNAL` | `proxy.cc:922` | -1 | 指定信号号，收到后 dump proxy 状态 |

### 2. 算法、协议、通道、线程

| 变量 | 定义位置 | 默认 | 说明 |
|---|---|---|---|
| `NCCL_ALGO` | `tuning/cost_model.cc:325`（`ncclGetEnv`）；名字表 `init.cc:57` | 自动 | `Tree,Ring,CollNetDirect,CollNetChain,NVLS,NVLSTree,PAT`；支持 `^` 排除与 `func:` 前缀（`parseList` 注释） |
| `NCCL_PROTO` | `tuning/cost_model.cc:324`；名字表 `init.cc:59` | 自动 | `LL,LL128,Simple`，语法同上 |
| `NCCL_SYM_KERNEL` | `tuning/cost_model.cc:326` | 自动 | 对称内存 kernel 选择 |
| `NCCL_TUNER_PLUGIN` | `plugin/tuner.cc:59` | 无 | 外部 tuner 插件 |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | `graph/connect.cc:331-332` | -2 | 集合通信通道数范围 |
| `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` | `init.cc:2001`、`:2000` | 未设 | 对应 `ncclConfig_t` 的 CTA 数 |
| `NCCL_CTA_POLICY` | `init.cc:96` | DEFAULT | `DEFAULT/EFFICIENCY/ZERO` |
| `NCCL_NCHANNELS_PER_NET_PEER` | `init.cc:2003` | 未设 | 每个网络 peer 的 p2p 通道数 |
| `NCCL_MIN_P2P_NCHANNELS` / `NCCL_MAX_P2P_NCHANNELS` | `graph/paths.cc:991-992` | 1 / 64 | send/recv 通道范围 |
| `NCCL_NTHREADS` / `NCCL_LL128_NTHREADS` | `tuning/tuning_general.cc:12-13` | -2 | 每 block 线程数 |
| `NCCL_BUFFSIZE` / `NCCL_LL_BUFFSIZE` / `NCCL_LL128_BUFFSIZE` | `init.cc:1007-1009` | -2（Simple 默认 4MiB，`DEFAULT_BUFFSIZE`） | 每连接各协议缓冲大小 |
| `NCCL_P2P_NET_CHUNKSIZE` / `_PCI_` / `_NVL_` | `init.cc:1011-1013` | 128K / 128K / 512K | send/recv 分块 |
| `NCCL_P2P_LL_THRESHOLD` | `enqueue/enqueue.cc:949` | 16384 | p2p 用 LL 的阈值 |
| `NCCL_LAUNCH_MODE` | `init.cc:1906` | PARALLEL | `GROUP` 时多 comm 同步发射 |
| `NCCL_RUNTIME_CONNECT` | `init.cc:65` | 1 | 懒连接（需 cuMem） |
| `NCCL_ENQUEUE_REARCH_ENABLE` | `enqueue/enqueue.cc:33` | 0 | 新入队架构 |
| `NCCL_GRAPH_REGISTER` | `enqueue/enqueue.cc:332` | 1 | CUDA graph 中的用户缓冲注册 |

### 3. scale-up：P2P / SHM / NVLS / cuMem / MNNVL

| 变量 | 定义位置 | 默认 | 说明 |
|---|---|---|---|
| `NCCL_P2P_DISABLE` / `NCCL_P2P_LEVEL` | `graph/paths.cc:310`（`ncclGetUserP2pLevel` → `ncclGetLevel`） | 自动（默认 PXB，AMD 少卡 SYS） | `P2P_DISABLE=1` 等价 LOC；LEVEL 取 `LOC/NVL/PIX/PXB/PHB/SYS`，旧整数也接受 |
| `NCCL_IGNORE_DISABLED_P2P` | `graph/paths.cc:303` | 0 | 忽略 NVML 报告的 P2P 不可用 |
| `NCCL_P2P_READ_ENABLE` | `transport/p2p.cc:331` | -2 | 强制 P2P read 模式 |
| `NCCL_P2P_DIRECT_DISABLE` | `transport/p2p.cc:332` | 0 | 禁用同进程 direct pointer |
| `NCCL_P2P_USE_CUDA_MEMCPY` | `transport/p2p.cc:122` | 0 | 用 copy engine 做 P2P |
| `NCCL_NVB_DISABLE` | `graph/paths.cc:36` | 0 | 禁止经中间 GPU 的 NVLink 路径 |
| `NCCL_SHM_DISABLE` / `NCCL_SHM_LOCALITY` | `transport/shm.cc:55-56` | 0 / 接收端 | SHM transport 开关与缓冲位置 |
| `NCCL_CUMEM_ENABLE` / `NCCL_CUMEM_HOST_ENABLE` | `misc/cudawrap.cc:16-17` | -2 / -1 | cuMem 显存/主机内存 |
| `NCCL_NVLS_ENABLE` | `transport/nvls.cc:57` | 2（自动） | NVLink SHARP |
| `NCCL_NVLS_CHUNKSIZE` / `NCCL_NVLS_NCHANNELS` | `transport/nvls.cc:58`、`init.cc:68` | 128K / 未设 | NVLS 分块与通道 |
| `NCCL_MNNVL_ENABLE` | `init.cc:1055` | 2 | 多节点 NVLink |
| `NCCL_MNNVL_UUID` / `_CLIQUE_ID` / `_CROSS_CLIQUE` | `init.cc:871-873` | -1 / -1 / 0 | MNNVL 域与 clique 覆盖 |
| `NCCL_DMABUF_ENABLE` | `init.cc:470` | 1 | DMA-BUF 注册 |

### 4. 网络通用、插件、GDR、PXN

| 变量 | 定义位置 | 默认 | 说明 |
|---|---|---|---|
| `NCCL_SOCKET_IFNAME` | `misc/socket.cc:217` | 自动（ib → 与 COMM_ID 同子网 → 非 docker/lo/virbr → docker → lo → virbr） | bootstrap 与 Socket 插件网卡；`^` 排除、`=` 精确 |
| `NCCL_SOCKET_FAMILY` | `misc/socket.cc:201` | 不限 | `AF_INET` / `AF_INET6` |
| `NCCL_SOCKET_RETRY_CNT` / `_RETRY_SLEEP_MSEC` / `_POLL_TIMEOUT_MSEC` | `misc/socket.cc:19-21` | 34 / 100 / 0 | socket 连接重试 |
| `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD` | `transport/net_socket.cc:86`、`:85` | -2 | Socket 插件并发 |
| `NCCL_COMM_ID` | `bootstrap.cc:112`、`:436`；`init.cc:2912` | 无 | 固定 bootstrap root 地址 |
| `NCCL_OOB_NET_ENABLE` | `bootstrap.cc:106` | 0 | bootstrap 走 net 插件 |
| `NCCL_UID_STAGGER_RATE` / `_THRESHOLD` | `bootstrap.cc:750-751` | 7000 / 256 | 大规模连 root 错峰 |
| `NCCL_NET` | `init.cc:2426` | 自动 | 指定网络插件名（如 `IB`、`Socket`） |
| `NCCL_NET_PLUGIN` | `plugin/net.cc:297` | `libnccl-net.so` | 逗号分隔列表，`none` 禁用外部插件 |
| `NCCL_NET_GDR_LEVEL` | `graph/paths.cc:542` | PXB（C2C 平台 P2C） | GPU-NIC 距离不超过该级别才用 GDR |
| `NCCL_NET_GDR_READ` | `graph/paths.cc:487` | -2 | 发送侧 GDR read |
| `NCCL_NET_GDR_C2C` | `graph/paths.cc:492` | 1 | C2C 平台 GDR |
| `NCCL_GDR_FLUSH_DISABLE` | `transport/net_ib/connect.cc:1415` | 0 | 关闭 GDR flush |
| `NCCL_NET_FORCE_FLUSH` | `graph/paths.cc:602` | 0 | 强制 flush |
| `NCCL_GDRCOPY_ENABLE` | `init.cc:152` | 0 | 使用 GDRCopy |
| `NCCL_NET_DISABLE_INTRA` | `graph/paths.cc:632` | 0 | 同机不走网络 |
| `NCCL_PXN_DISABLE` / `NCCL_P2P_PXN_LEVEL` | `graph/paths.cc:701`；`graph/search.cc:1478` | 0 / 2 | PXN 开关与级别 |
| `NCCL_CROSS_NIC` | `graph/search.cc:16` | 2 | 0 不跨 NIC，1 允许，2 先不跨找不到再允许 |
| `NCCL_NET_SHARED_BUFFERS` / `NCCL_NET_SHARED_COMMS` | `transport/net.cc:170-171` | -2 / 1 | 网络缓冲/连接共享 |
| `NCCL_IGNORE_NET_MISMATCH` | `init.cc:153` | 1 | 各 rank 网卡数不一致时只打 INFO 继续；设 0 则报错退出 |
| `NCCL_PROGRESS_APPENDOP_FREQ` / `NCCL_PROXY_APPEND_BATCH_SIZE` | `proxy.cc:923`、`:828` | 8 / 16 | proxy 取新 op 的频率与批量 |

### 5. InfiniBand / RoCE（`transport/net_ib/`）

| 变量 | 定义位置 | 默认 | 说明 |
|---|---|---|---|
| `NCCL_IB_DISABLE` | `init.cc:27` | 0 | 禁用 IB 插件（会回落 Socket） |
| `NCCL_IB_HCA` | `init.cc:459`（`ncclGetEnv`） | 全部 | 列表 `dev[:port]`，逗号分隔，`^` 排除、`=` 精确（`parseStringList`） |
| `NCCL_IB_GID_INDEX` | `connect.cc:13` | -1（自动） | RoCE GID 选择 |
| `NCCL_IB_ROCE_VERSION_NUM` | `connect.cc:15` | 2 | RoCE 版本 |
| `NCCL_IB_TIMEOUT` | `connect.cc:16` | 20 | 本地 ACK 超时 = 4.096us × 2^timeout（`env.rst`） |
| `NCCL_IB_RETRY_CNT` | `connect.cc:17` | 7 | 重传次数 |
| `NCCL_IB_SL` / `NCCL_IB_TC` / `NCCL_IB_FIFO_TC` | `connect.cc:21-23` | -1 | 服务级别、流量类别（RoCE DSCP/QoS） |
| `NCCL_IB_PKEY` / `NCCL_IB_PKEY_VALUE` | `connect.cc:18-19` | 0 / -1 | 分区 |
| `NCCL_IB_USE_INLINE` | `connect.cc:20` | 0 | inline 小消息 |
| `NCCL_IB_ECE_ENABLE` | `connect.cc:24` | 1 | Enhanced Connection Establishment |
| `NCCL_IB_QPS_PER_CONNECTION` | `connect.cc:62` | 1 | 每连接 QP 数（多路径/ECMP 打散） |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | `common.cc:23` | 0 | 数据是否在多个 QP 间切分 |
| `NCCL_IB_ADAPTIVE_ROUTING` / `NCCL_IB_AR_THRESHOLD` | `init.cc:12`；`p2p.cc:13` | -2 | 自适应路由 |
| `NCCL_IB_PCI_RELAXED_ORDERING` | `init.cc:11` | 2 | PCIe relaxed ordering |
| `NCCL_IB_MERGE_NICS` / `NCCL_IB_MERGE_VFS` | `init.cc:29`、`:28` | 1 | 多端口合并为虚拟设备 |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | `common.cc:25` | 1 | 上报 IB 异步事件 |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER` | `p2p_resiliency.cc:13` | 0 | 端口故障切换 |
| `NCCL_IB_WQE_LATENCY_THRESHOLD_NS` / `_REPORT` | `wqe_lat_mon.cc:17-18` | 0 / 1 | WQE 延迟监控 |
| `NCCL_IB_MQP_RETRY_ALL` / `_CNT` / `_SLEEP_MSEC` | `misc/ibvwrap.cc:98-100` | 0 / 34 / 100 | verbs 调用重试 |

（IB 机制本身见 [../01-rdma/README.md](../01-rdma/README.md)。）

### 6. 拓扑与图

| 变量 | 定义位置 | 说明 |
|---|---|---|
| `NCCL_TOPO_FILE` | `graph/topo.cc:2000` | 读入拓扑 XML |
| `NCCL_TOPO_DUMP_FILE` / `NCCL_TOPO_DUMP_FILE_RANK` | `init.cc:1385`；`graph/topo.cc:1091` | 导出拓扑 XML（默认 rank 0） |
| `NCCL_GRAPH_FILE` | `graph/search.cc:1198` | 读入通道图 |
| `NCCL_GRAPH_DUMP_FILE` / `NCCL_GRAPH_DUMP_FILE_RANK` | `graph/search.cc:1435`；`init.cc:1049` | 导出通道图 |
| `NCCL_TOPO_SPLIT_MLOPART` | `graph/topo.cc:667` | MLOPart 拓扑拆分 |

### 7. RAS、诊断、插件、其他

| 变量 | 定义位置 | 默认 | 说明 |
|---|---|---|---|
| `NCCL_RAS_ENABLE` | `ras/ras.cc:91` | 1 | RAS 子系统 |
| `NCCL_RAS_ADDR` | `ras/client_support.cc:172` | `localhost:28028` | RAS 客户端监听地址 |
| `NCCL_RAS_TIMEOUT_FACTOR` | `ras/ras_param.cc:28` | 1.0 | RAS 各类超时的缩放因子（正浮点数，非法值回落 1.0；`ncclras -t` 默认超时也按它缩放） |
| `NCCL_RUN_RAS_DIAGNOSTICS` | `ras/ras.cc:90` | 0 | RAS 被动诊断 |
| `NCCL_RUN_DIAGNOSTICS` | `diagnostics.cc:31` | 0 | init 时主动诊断 |
| `NCCL_DIAGNOSTICS_IB_BW_TIMEOUT` | `diagnostics/ib_write_bw.cc:36` | 5 | 诊断中 IB 带宽测试超时 |
| `NCCL_PROGRESS_COUNTER_MONITOR_*` | `ras/progress_monitor.cc:25-27` | 1000ms / 5000ms / 600s | 进度计数监控（判定"卡住"） |
| `NCCL_PROFILER_PLUGIN` | `plugin/profiler.cc:109` | 无 | profiler 插件 |
| `NCCL_IGNORE_CPU_AFFINITY` | `graph/topo.cc:2445` | 0 | 忽略继承的 CPU 亲和，只用 GPU 亲和 |
| `NCCL_CONF_FILE` | `misc/param.cc:54` | `~/.nccl.conf`、`/etc/nccl.conf` | 配置文件 |
| `NCCL_COMM_BLOCKING` | `init.cc:64` | 未设 | 阻塞/非阻塞 comm |
| `NCCL_CHECK_POINTERS` | `init.cc:63` | 0 | 检查用户指针 |
| `NCCL_PARAM_DUMP_ALL` / `NCCL_NO_CACHE` | `param/param.cc:16`、`:20` | false / 无 | 参数打印与缓存控制 |

### 8. 不属于 NCCL 的同名前缀变量

- `TORCH_NCCL_*`、`NCCL_ASYNC_ERROR_HANDLING`、`NCCL_BLOCKING_WAIT` 等由 PyTorch 读取，不在 NCCL 源码中（待确认：在 PyTorch 源码中的位置与当前名称；确认方式：在 PyTorch 仓库 grep `TORCH_NCCL_`，本仓库 `third_party/` 未包含 PyTorch）。
- `NCCL_TESTS_SPLIT`、`NCCL_TESTS_SPLIT_MASK`、`NCCL_TESTS_MIN_BW`、`NCCL_TESTS_DEVICE` 由 nccl-tests 读取（`third_party/nccl-tests/src/common.cu`）。

## 性能要点与常见坑

- 不要在生产中长期设置 `NCCL_ALGO/PROTO/MIN_NCHANNELS` 等"强制"变量：它们绕过了代价模型，NCCL 升级或拓扑变化后可能更差；只在定位时做 A/B。
- 多变量叠加时先确认生效：日志 `NCCL_xxx set by environment to ...`；用 `/etc/nccl.conf` 时尤其要查（集群镜像里常被预置）。
- `NCCL_IB_HCA` 写成 `mlx5` 会前缀匹配 `mlx5_10`；需要精确匹配时用 `=mlx5_1:1`。
- `NCCL_SOCKET_IFNAME` 只影响 bootstrap/Socket，不决定 IB 数据面用哪块网卡（那是 `NCCL_IB_HCA`）。
- 环境变量在第一次读取后缓存，进程内修改无效（`NCCL_PARAM` 实现）。

## 面试自测

1. 问：`NCCL_P2P_LEVEL=NVL` 和 `NCCL_P2P_DISABLE=1` 分别有什么效果？
   答：前者只允许经 NVLink 的 GPU 对使用 P2P（PCIe 相连的走 SHM/NET）；后者等价把级别设为 LOC，完全禁用 P2P transport（`ncclGetLevel` 中 disable=1 → `PATH_LOC`）。
2. 问：RoCE 集群上 NCCL 跨机不通或性能差，你会检查哪些 IB 相关变量？
   答：`NCCL_IB_HCA`（是否选对网卡/端口）、`NCCL_IB_GID_INDEX` 与 `NCCL_IB_ROCE_VERSION_NUM`（GID 是否是 RoCEv2/对应 IP）、`NCCL_IB_TC/SL`（是否落在无损队列/PFC 优先级）、`NCCL_IB_TIMEOUT/RETRY_CNT`（拥塞时是否过早报错）、`NCCL_IB_QPS_PER_CONNECTION`（ECMP 打散），以及 `NCCL_NET_GDR_LEVEL` 是否让 GDR 生效。
3. 问：如何确认 NCCL 实际使用的网络插件？
   答：日志中的 `NCCL_NET_PLUGIN set by environment`、`Successfully loaded external network plugin`、`Initialized NET plugin <name>`、`Assigned NET plugin <name> to comm`，以及每个 channel 的 `via NET/<name>/<dev>`。
4. 问：`NCCL_DEBUG_SUBSYS` 默认值是什么？排查网络问题时应加什么？
   答：默认 `INIT,BOOTSTRAP,ENV`（`debug.cc:68`）；网络问题加 `NET`（插件、网卡、GDR）、`GRAPH`（网卡选择与通道）、必要时 `PROXY`。
5. 问：为什么在 Python 里 `os.environ["NCCL_ALGO"]="Ring"` 有时不生效？
   答：若进程中已初始化过 NCCL（或 PyTorch 已创建过 communicator），参数已被读取缓存；必须在第一次调用 NCCL 之前设置，或通过启动脚本导出。

## 待深入

- `DEFINE_NCCL_PARAM` 新机制的迁移进度：目前用它定义的变量较少（`grep -rn "DEFINE_NCCL_PARAM(" third_party/nccl/src` 约 10 处），其余仍是 `NCCL_PARAM`（约 248 处）。
- env 插件（`src/plugin/env/`）如何在 `ncclGetEnv` 之前注入值。
