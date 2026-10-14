# Verbs 基础：资源对象、QP 状态机与建链

> 层次：下层 · 来源：third_party/rdma-core@bd3282a（libibverbs 头文件、man、examples）

## 一句话

Verbs 编程 = 先建好"保护域里的一组资源"（PD → MR、CQ、QP），再通过带外通道（通常是 TCP）交换 QPN/PSN/LID/GID 把 QP 推到 RTS，之后数据面只剩两件事：往 QP 里 post WR、从 CQ 里 poll WC。

## 在调用链中的位置

- 上层：NCCL `net_ib`、UCX、MPI、perftest、DeepEP/NVSHMEM 的 IBRC/IBGDA 后端，都调用这一层 API。NCCL 自己的 QP 建链流程见 [05-gpudirect-and-nccl-ib.md](05-gpudirect-and-nccl-ib.md)。
- 本层：`libibverbs` 定义 API（`third_party/rdma-core/libibverbs/verbs.h`），大部分数据面函数是 `static inline`，直接跳到 provider 的函数指针（例如 `ibv_post_send` 在 `verbs.h` 中调 `qp->context->ops.post_send`）。
- 下层：provider（`third_party/rdma-core/providers/mlx5/` 等）在用户态写 WQE、敲 doorbell；控制面命令经 `cmd_*.c`（如 `cmd_qp.c`、`cmd_mr.c`）走内核 uverbs。

## 核心概念

### 资源对象

| 对象 | 头文件中的类型 / 创建函数（`libibverbs/verbs.h`） | 作用 |
|---|---|---|
| Context | `struct ibv_context`；`ibv_open_device` | 打开一块 RNIC，后续资源都挂在它下面 |
| PD 保护域 | `struct ibv_pd`；`ibv_alloc_pd` | 隔离边界：QP 只能访问同一 PD 下注册的 MR |
| MR 内存区域 | `struct ibv_mr`；`ibv_reg_mr` / `ibv_reg_mr_iova2` / `ibv_reg_dmabuf_mr` | 把虚拟内存 pin 住并建好网卡侧地址翻译；返回 `lkey`（本地 SGE 用）和 `rkey`（给远端做 RDMA 用）。权限位见 `enum ibv_access_flags`：`IBV_ACCESS_LOCAL_WRITE / REMOTE_WRITE / REMOTE_READ / REMOTE_ATOMIC / ON_DEMAND / RELAXED_ORDERING` 等 |
| CQ 完成队列 | `struct ibv_cq`；`ibv_create_cq` / `ibv_create_cq_ex` | 网卡把完成事件 WC 写进来；`ibv_poll_cq` 轮询，或 `ibv_req_notify_cq` + `ibv_get_cq_event` 走事件 |
| QP 队列对 | `struct ibv_qp`；`ibv_create_qp` / `ibv_create_qp_ex` | 一个 SQ + 一个 RQ，是"连接/端点"的硬件上下文；类型 `enum ibv_qp_type`：`IBV_QPT_RC / UC / UD / RAW_PACKET / XRC_SEND / XRC_RECV / DRIVER` |
| SRQ 共享接收队列 | `struct ibv_srq`；`ibv_create_srq` | 多个 QP 共享一个 RQ，减少每连接预投 recv 的内存 |
| AH 地址句柄 | `struct ibv_ah`；`ibv_create_ah`（由 `struct ibv_ah_attr`） | UD 每个 WR 需要指定目标地址；RC/UC 的地址放在 QP 属性 `ah_attr` 里，只设一次 |

`union ibv_gid` 是 128 bit 全局地址；GID 类型枚举有 `IBV_GID_TYPE_IB`、`IBV_GID_TYPE_ROCE_V1`、`IBV_GID_TYPE_ROCE_V2`（`verbs.h` 开头），可以用 `ibv_query_gid_ex` 查到某个 GID index 的类型。

### QP 状态机

`enum ibv_qp_state`：`IBV_QPS_RESET → INIT → RTR → RTS`，另有 `SQD`（drain）、`SQE`（SQ error）、`ERR`。`ibv_modify_qp.3` man 页（`third_party/rdma-core/libibverbs/man/ibv_modify_qp.3`）列出每次迁移必须携带的属性掩码，按 RC 归纳：

| 迁移 | 必填 attr mask | 含义 |
|---|---|---|
| RESET→INIT | `IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS` | 绑定端口、分区、允许远端做哪些操作 |
| INIT→RTR | `STATE | AV | PATH_MTU | DEST_QPN | RQ_PSN | MAX_DEST_RD_ATOMIC | MIN_RNR_TIMER` | "能收"：知道对端是谁（AV：LID 或 GID）、对端 QPN、期望的首个接收 PSN |
| RTR→RTS | `STATE | TIMEOUT | RETRY_CNT | RNR_RETRY | SQ_PSN | MAX_QP_RD_ATOMIC` | "能发"：设定发送起始 PSN 和重传参数 |

UC 少了 RNR/重传/RD_ATOMIC 相关项；UD 的 RTR 只要 `STATE`，RTS 只要 `STATE | SQ_PSN`（man 页 137–138 行）。RECV 可以在 INIT 之后就 post，但只有进入 RTR 后才会接收报文。

### 建链需要交换什么

对一个 RC QP，双方需要拿到对方的：

- **QPN**：填到本端 `dest_qp_num`。
- **PSN**：对端的 `sq_psn` 就是本端的 `rq_psn`。随机起始 PSN 可以避免旧连接残留报文被误收。
- **LID**（IB 子网内路由）或 **GID**（跨子网 IB / 所有 RoCE；RoCE 没有 LID，必须 `is_global=1` 并填 GRH）。
- 其余：MTU（取两端 active_mtu 的较小值）、SL/traffic class、以及要做单边操作的 **buffer 地址 + rkey**。

这些信息的交换不在 verbs 范围内，可以用 TCP socket（rc_pingpong、perftest 默认、NCCL bootstrap socket）或 `librdmacm`（`third_party/rdma-core/librdmacm/`，perftest `-R`）。

## 源码导读（路径 + 函数 + 作用）

`third_party/rdma-core/libibverbs/examples/` 下实际有：`rc_pingpong.c`、`uc_pingpong.c`、`ud_pingpong.c`、`srq_pingpong.c`、`xsrq_pingpong.c`、`pingpong.c/.h`（公共工具）、`devinfo.c`、`device_list.c`、`asyncwatch.c`。

以 `rc_pingpong.c` 为主线：

1. `pp_init_ctx`：`ibv_open_device` → 可选 `ibv_create_comp_channel`（`-e` 事件模式）→ `ibv_alloc_pd` → `ibv_reg_mr`（默认 `IBV_ACCESS_LOCAL_WRITE`；`-o` 加 `IBV_ACCESS_ON_DEMAND`，隐式 ODP 用 `ibv_reg_mr(pd, NULL, SIZE_MAX, ...)`）→ `ibv_create_cq`（或 `_ex`）→ `ibv_create_qp`（`IBV_QPT_RC`，`max_send_wr=1`，`max_recv_wr=rx_depth`；`-N` 用 `ibv_create_qp_ex` 新 API）→ `ibv_query_qp` 看 `max_inline_data`，消息够小就置 `IBV_SEND_INLINE` → `ibv_modify_qp` 到 `IBV_QPS_INIT`（`qp_access_flags=0`，因为只做 SEND/RECV）。
2. `main`：`ibv_query_port` 取 LID；`-g` 时 `ibv_query_gid` 取 GID；`my_dest.psn = lrand48() & 0xffffff`（PSN 是 24 bit）。
3. `pp_client_exch_dest` / `pp_server_exch_dest`：TCP 上交换形如 `"%04x:%06x:%06x:%s"` 的字符串，即 `LID:QPN:PSN:GID`。注意 server 端是"先读对端 → 调 `pp_connect_ctx` → 再回写自己的信息"，保证 client 发数据时 server 已经在 RTR。
4. `pp_connect_ctx`：一次 `ibv_modify_qp` 到 RTR（`dest_qp_num`、`rq_psn`、`path_mtu`、`min_rnr_timer=12`、`ah_attr.dlid`；如果对端 GID 的 `interface_id` 非 0，就设 `is_global=1` 并填 `grh.dgid / sgid_index`），再一次到 RTS（`timeout=14`、`retry_cnt=7`、`rnr_retry=7`、`sq_psn=my_psn`）。
5. `pp_post_recv` / `pp_post_send`：构造 `ibv_sge{addr,length,lkey}` 和 `ibv_recv_wr` / `ibv_send_wr`（`IBV_WR_SEND`），分别调 `ibv_post_recv` / `ibv_post_send`；`-N` 走 `ibv_wr_start / ibv_wr_send / ibv_wr_set_sge / ibv_wr_complete`。
6. 主循环：轮询模式 `ibv_poll_cq(cq, 2, wc)`；事件模式 `ibv_get_cq_event` → `ibv_req_notify_cq` 重新 arm → poll，退出前 `ibv_ack_cq_events`。
7. `pp_close_ctx`：逆序销毁 QP → CQ → MR → PD → comp channel → device。

其他示例的看点：

- `ud_pingpong.c`：`IBV_QPT_UD`，INIT 时设 `qkey = 0x11111111`；`ibv_create_ah` 建目标地址；接收 buffer 多留 40 字节，因为 UD 接收会先写入 40 B 的 GRH。
- `srq_pingpong.c`：一个 `ibv_create_srq` 给 `num_qp` 个 QP 共用，CQ 大小 `rx_depth + num_qp`。
- `xsrq_pingpong.c`：XRC（`IBV_QPT_XRC_SEND / XRC_RECV` + `ibv_open_xrcd`）。
- `uc_pingpong.c`：与 rc 版几乎一样，只把类型换成 `IBV_QPT_UC`。

## 性能要点与常见坑

- **RoCE 必须用 GID**：rc_pingpong 不带 `-g` 时 GID 为 0，在 RoCE 上会因为没有 LID 而失败（代码里 `link_layer != IBV_LINK_LAYER_ETHERNET && !lid` 才报错，以太网上 LID 为 0 是正常的）。GID index 选错（RoCEv1 或 IPv6 link-local 那条）是 RoCE 最常见的"连不上 / 走错 VLAN"原因。
- **MR 注册很贵**：要 pin 页并写网卡翻译表，不能放在数据路径上。通信库普遍做 MR cache（NCCL `third_party/nccl/src/transport/net_ib/reg.cc` 按页对齐地址做有序 cache + 引用计数）。
- **RNR**：RC 上对端没有预投 RECV 时，会按 `min_rnr_timer` 回 RNR NAK，`rnr_retry=7` 表示无限重试；用 WRITE（不消耗对端 RECV）可以绕开这个问题，但 WRITE_WITH_IMM 仍然要消耗对端一个 RECV。
- **`timeout` 的含义**：本地 ACK 超时 = 4.096 µs × 2^timeout。perftest README 写作 `(4 uSec)*(2^timeout)`，默认 14，约 67 ms；NCCL 默认 `NCCL_IB_TIMEOUT=20`（`third_party/nccl/src/transport/net_ib/connect.cc`），约 4.3 s。大规模集群里 timeout 太小会误触发重传或报 `IBV_WC_RETRY_EXC_ERR`，太大则故障发现慢。
- **CQ 溢出**：CQ 深度要不小于未完成 WR 数（rc_pingpong 用 `rx_depth + 1`），否则会触发 `IBV_EVENT_CQ_ERR`，QP 进 ERR。
- **QP 进 ERR 之后**：所有未完成 WR 都以 flush error 完成，只能 modify 到 RESET 重建；NCCL 的 `ncclIbQpReset` / `ncclIbQpError`（`connect.cc`）就是做这个的。

## 面试自测（问题 + 要点答案）

1. **PD、MR、QP 的关系？lkey 和 rkey 有什么区别？**
   PD 是保护边界，MR 和 QP 都属于某个 PD，QP 只能用同一 PD 的 MR。lkey 写在本地 SGE 里，让网卡做本地地址翻译和权限检查；rkey 交给对端，对端在 RDMA WRITE/READ/ATOMIC 的 WR 里携带它来访问本端内存。rkey 一旦泄露，持有者就能访问这块 MR，所以权限位要按需开启。
2. **说出 RC QP 从 RESET 到 RTS 每一步要设的关键属性。**
   INIT：port、pkey_index、access_flags。RTR：dest_qpn、rq_psn、path_mtu、AV（dlid 或 GRH dgid + sgid_index）、max_dest_rd_atomic、min_rnr_timer。RTS：sq_psn、timeout、retry_cnt、rnr_retry、max_rd_atomic。
3. **为什么 rc_pingpong 的 server 要先把自己推到 RTR 再回复 client？**
   client 收到回复后马上会发数据；如果 server QP 还没到 RTR，收到的报文会被丢弃，或者因为没有 RECV 而触发 RNR。
4. **RoCE 和 IB 建链交换的信息有什么不同？**
   IB 子网内用 LID 即可（GRH 可选）；RoCE 没有 LID，必须 is_global=1，用 GID（RoCEv2 GID 里嵌着 IPv4/IPv6 地址）和 sgid_index，traffic_class 决定 DSCP。
5. **UD 和 RC 在接收 buffer 上有什么区别？**
   UD 每个接收 buffer 前面多 40 B GRH 空间；UD 不分片，单个消息不能超过 path MTU；每个发送 WR 都要带 AH、remote_qpn 和 qkey。
6. **SRQ 解决什么问题？**
   N 个连接各预投 rx_depth 个 RECV，内存会随 N 线性增长；SRQ 让它们共享一个池。代价是需要处理 SRQ limit 事件（`IBV_EVENT_SRQ_LIMIT_REACHED`），并补投 RECV。
7. **事件模式和轮询模式怎么选？**
   轮询延迟最低，但会占满一个核；事件模式省 CPU，但要多一次中断 + 唤醒，延迟增加若干 µs。通信库的 proxy 线程一般选轮询。

## 待深入

- `ibv_wr_*` 新 post API（`ibv_qp_ex`）与旧 `ibv_post_send` 的性能差异：perftest 提供 `--use_old_post_send` 开关（`third_party/perftest/src/perftest_parameters.c`），可以做 A/B 对比。待确认：实测差异；确认方式：在有 RNIC 的机器上分别加、不加 `--use_old_post_send` 跑 `ib_write_bw -s 64 -l 32`。
- 内核侧 uverbs 的命令通道（`cmd_ioctl.c`）与 provider `alloc_context` 的握手。
- ODP（on-demand paging）缺页时网卡侧的行为和性能代价。
