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

下图是这些对象的从属关系（谁挂在谁下面），以及数据面 WR 进、WC 出的方向：

```text
ibv_context            ← ibv_open_device，一块 RNIC
 ├── ibv_cq            ← ibv_create_cq(context, ...)，不属于 PD
 │     ▲   ▲
 │     │   └──────────────────────────┐  WC：网卡写入，ibv_poll_cq 取走
 │     └──────────────────────┐       │
 └── ibv_pd                   │       │  ← ibv_alloc_pd，保护边界
      ├── ibv_mr              │       │  ← lkey 填进本地 SGE；rkey 交给对端
      ├── ibv_qp              │       │  ← ibv_create_qp(pd, {send_cq, recv_cq, ...})
      │    ├── SQ ── send_cq ─┘       │     ibv_post_send 往 SQ 放 WR
      │    └── RQ ── recv_cq ─────────┘     ibv_post_recv 往 RQ 放 WR（或改挂 SRQ）
      ├── ibv_srq                         ← 多个 QP 共享的 RQ
      └── ibv_ah                          ← 只有 UD 用：每个发送 WR 指定目标
```

rc_pingpong 里 send_cq 和 recv_cq 是同一个 CQ（CQ 大小 `rx_depth + 1`），图中分开画只是为了看清两条完成路径。

`union ibv_gid` 是 128 bit 全局地址；GID 类型枚举有 `IBV_GID_TYPE_IB`、`IBV_GID_TYPE_ROCE_V1`、`IBV_GID_TYPE_ROCE_V2`（`verbs.h` 开头），可以用 `ibv_query_gid_ex` 查到某个 GID index 的类型。

### QP 状态机

`enum ibv_qp_state`：`IBV_QPS_RESET → INIT → RTR → RTS`，另有 `SQD`（drain）、`SQE`（SQ error）、`ERR`。`ibv_modify_qp.3` man 页（`third_party/rdma-core/libibverbs/man/ibv_modify_qp.3`）列出每次迁移必须携带的属性掩码，按 RC 归纳：

| 迁移 | 必填 attr mask | 含义 |
|---|---|---|
| RESET→INIT | `IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS` | 绑定端口、分区、允许远端做哪些操作 |
| INIT→RTR | `STATE | AV | PATH_MTU | DEST_QPN | RQ_PSN | MAX_DEST_RD_ATOMIC | MIN_RNR_TIMER` | "能收"：知道对端是谁（AV：LID 或 GID）、对端 QPN、期望的首个接收 PSN |
| RTR→RTS | `STATE | TIMEOUT | RETRY_CNT | RNR_RETRY | SQ_PSN | MAX_QP_RD_ATOMIC` | "能发"：设定发送起始 PSN 和重传参数 |

UC 少了 RNR/重传/RD_ATOMIC 相关项；UD 的 RTR 只要 `STATE`，RTS 只要 `STATE | SQ_PSN`（man 页 137–138 行）。RECV 可以在 INIT 之后就 post，但只有进入 RTR 后才会接收报文。

把上表和"出错后怎么办"（见下文常见坑）连起来，RC QP 的主路径如下（箭头旁是该次 `ibv_modify_qp` 的必填 mask；SQD/SQE 未画）：

```text
  ibv_create_qp
       │
       ▼
  ┌─────────┐
  │  RESET  │
  └────┬────┘
       │ STATE | PKEY_INDEX | PORT | ACCESS_FLAGS
       ▼
  ┌─────────┐
  │  INIT   │   ← 可以 post RECV，但还不会收包
  └────┬────┘
       │ STATE | AV | PATH_MTU | DEST_QPN | RQ_PSN | MAX_DEST_RD_ATOMIC | MIN_RNR_TIMER
       ▼
  ┌─────────┐
  │   RTR   │   ← "能收"：知道对端 LID/GID、QPN、期望的首个 PSN
  └────┬────┘
       │ STATE | TIMEOUT | RETRY_CNT | RNR_RETRY | SQ_PSN | MAX_QP_RD_ATOMIC
       ▼
  ┌─────────┐
  │   RTS   │   ← "能发"：数据面只剩 post WR / poll WC
  └────┬────┘
       │ 出错（如 IBV_WC_RETRY_EXC_ERR、CQ 溢出）
       ▼
  ┌─────────┐
  │   ERR   │   ← 未完成的 WR 全部以 flush error 完成
  └────┬────┘
       │ modify 到 RESET，重新走一遍（NCCL：ncclIbQpReset）
       └──────► RESET
```

### 建链需要交换什么

对一个 RC QP，双方需要拿到对方的：

- **QPN**：填到本端 `dest_qp_num`。
- **PSN**：对端的 `sq_psn` 就是本端的 `rq_psn`。随机起始 PSN 可以避免旧连接残留报文被误收。
- **LID**（IB 子网内路由）或 **GID**（跨子网 IB / 所有 RoCE；RoCE 没有 LID，必须 `is_global=1` 并填 GRH）。
- 其余：MTU（取两端 active_mtu 的较小值）、SL/traffic class、以及要做单边操作的 **buffer 地址 + rkey**。

这些信息的交换不在 verbs 范围内，可以用 TCP socket（rc_pingpong、perftest 默认、NCCL bootstrap socket）或 `librdmacm`（`third_party/rdma-core/librdmacm/`，perftest `-R`）。

### 用 RDMA CM 建链：REQ / REP / RTU 与 QP 状态

RDMA CM（librdmacm）替应用做三件事：把 IP 地址解析成 RDMA 设备和路径；在两端之间交换 QPN、PSN 等连接参数；按顺序把 QP 推到 RTS。所以它同时牵涉**三套状态**：应用看到的 CM 事件、线上的 CM 消息、QP 状态。

```text
   Client（主动方）                                    Server（被动方）
   ────────────────                                    ────────────────
                                                       rdma_create_id / rdma_bind_addr / rdma_listen
   rdma_create_id
   rdma_resolve_addr(dst IP)
     ── ADDR_RESOLVED        IP → 本地 RDMA 设备/端口、对端 GID
   rdma_resolve_route
     ── ROUTE_RESOLVED       路径（IB 查 SA；RoCE 走 IP 路由）
   rdma_create_qp            QP: RESET → INIT
   rdma_connect ─────────── REQ ──────────────────────►
                                                       ── CONNECT_REQUEST（得到新的 cm_id）
                                                       rdma_create_qp       QP: RESET → INIT
                                                       rdma_accept          QP: INIT → RTR → RTS
                ◄─────────── REP ──────────────────────
   库内部处理 REP:           QP: INIT → RTR → RTS
                 ─────────── RTU ──────────────────────►
     ── ESTABLISHED                                    ── ESTABLISHED
   ═══════════════════ 开始 post_send / post_recv ════════════════════
   rdma_disconnect ──────── DREQ ─────────────────────►
                ◄────────── DREP ──────────────────────  ── DISCONNECTED
     ── DISCONNECTED
     ── TIMEWAIT_EXIT        之后才能安全复用或销毁 QP
```

**第一套：CM 事件**（`rdma_get_cm_event` 返回，枚举在 `third_party/rdma-core/librdmacm/rdma_cma.h`）

| 一方 | 正常顺序 | 失败分支 |
|---|---|---|
| Client | `ADDR_RESOLVED` → `ROUTE_RESOLVED` → `ESTABLISHED` → `DISCONNECTED` → `TIMEWAIT_EXIT` | `ADDR_ERROR`、`ROUTE_ERROR`、`UNREACHABLE`、`REJECTED`、`CONNECT_ERROR` |
| Server | `CONNECT_REQUEST` → `ESTABLISHED` → `DISCONNECTED` → `TIMEWAIT_EXIT` | 同上 |

**第二套：线上的 CM 消息**（IB 规范第 12 章 Communication Management；本地源码只在用户态通过 `write()` 给内核 CM 下命令，消息本身由内核 `ib_cm` 收发，本地未克隆内核）

| 消息 | 方向 | 内容 |
|---|---|---|
| REQ | C → S | client 的 QPN、起始 PSN、`responder_resources` / `initiator_depth`（READ/ATOMIC 并发深度）、私有数据 |
| REP | S → C | server 的 QPN、起始 PSN、协商后的深度、私有数据（常用来带 buffer 地址和 rkey） |
| **RTU**（Ready To Use） | C → S | client 已就绪，连接可用 |
| REJ | 任意方向 | 拒绝（如 `rdma_reject`） |
| DREQ / DREP | 任意方向 | 断开请求 / 应答 |

**第三套：QP 状态**。用 CM 时应用不用自己调 `ibv_modify_qp`，库在下面这些时机自动推进（`third_party/rdma-core/librdmacm/cma.c`）：

| 迁移 | Client | Server | 函数 |
|---|---|---|---|
| RESET → INIT | `rdma_create_qp` 内 | `rdma_create_qp` 内 | `ucma_init_conn_qp` |
| INIT → RTR → RTS | 收到 REP 后，发 RTU 前 | `rdma_accept` 内，发 REP 前 | `ucma_modify_qp_rtr`、`ucma_modify_qp_rts` |
| → ERR | 被拒、出错、断开 | 同左 | `ucma_modify_qp_err` |

源码里值得注意的三点：

- **server 先就绪**：`rdma_accept` 先把 QP 推到 RTR、RTS，再发 REP。这和 rc_pingpong 的 server "先 RTR 再回信"是同一个道理。
- **client 通常看不到 `CONNECT_RESPONSE`**：`rdma_get_cm_event` 收到它时，如果 client 已经建了 QP，就调 `ucma_process_conn_resp`（RTR → RTS → 发 RTU），再把事件改写成 `ESTABLISHED` 交给应用；只有没建 QP 时才把 `CONNECT_RESPONSE` 原样交出去。
- **RTR 前会再设一次 INIT**：`ucma_modify_qp_rtr` 先用 `rdma_init_qp_attr` 取 INIT 属性再 modify 一次，然后取 RTR 属性。RTR 需要的对端 QPN、PSN、路径、MTU 都由内核根据 CM 握手结果填好。

**RTU 不是 QP 状态**：`enum ibv_qp_state`（`third_party/rdma-core/libibverbs/verbs.h`）只有 `RESET / INIT / RTR / RTS / SQD / SQE / ERR`。RTR 是 Ready To **Receive**、RTS 是 Ready To **Send**，都是 QP 状态；RTU 是 Ready To **Use**，是 CM 握手的第三条**消息**。client 发 RTU 时，它的 QP 已经在 RTS 了。资料里把 RTU 画在状态图上时，画的通常是 CM 连接状态机，收到 RTU 后 CM 连接进入"已建立"，而不是 QP 的状态。

```text
CM 消息:         REQ  ──►        REP  ◄──        RTU  ──►
                                  ▲               ▲
QP 状态: Server  INIT → RTR → RTS ┘               │
         Client                   INIT → RTR → RTS┘
```

和手工建链（rc_pingpong、NCCL）的对比：

| | 手工建链 | RDMA CM |
|---|---|---|
| 交换 QPN / PSN / GID | 自己开 TCP socket | REQ / REP 里带 |
| 地址解析 | 自己指定设备、端口、GID index | `rdma_resolve_addr` 从 IP 算出 |
| QP 状态推进 | 自己调三次 `ibv_modify_qp` | 库自动完成 |
| ToS / DSCP | 自己设 `grh.traffic_class`（NCCL 用 `NCCL_IB_TC`） | `cma_roce_tos` 或 `rdma_set_option`；`cma_roce_tos` 只影响 RDMA_CM 建立的 QP（见 [RoCE QoS 配置调研](../../docs/references/2026-10-17-roce-qos-config.md) Q4） |

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

把第 1～5 步按时间排开，可以看清 server "先 RTR 再回信"的意义（RECV 在 `main` 里交换地址之前就已 `pp_post_recv(ctx, ctx->rx_depth)` 预投）：

```text
  client (rc_pingpong <server>)                  server (rc_pingpong)
  ─────────────────────────────                  ────────────────────
  pp_init_ctx: QP -> INIT                        pp_init_ctx: QP -> INIT
  pp_post_recv x rx_depth                        pp_post_recv x rx_depth
        │                                              │
        │ ── TCP "LID:QPN:PSN:GID" (client) ─────────► │ read
        │                                              │ pp_connect_ctx: QP -> RTR -> RTS
        │ ◄───────── TCP "LID:QPN:PSN:GID" (server) ── │ write
        │ ── TCP "done" ─────────────────────────────► │
  pp_connect_ctx: QP -> RTR -> RTS                     │
  pp_post_send ═══ IBV_WR_SEND ══════════════════════► │ 消耗一个预投 RECV，CQ 出 WC
        │ ◄═════════════════════════ IBV_WR_SEND ═════ │
        │          ... ping-pong loop until iters ...  │
```

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
8. **RDMA CM 建链的过程？RTU 是 QP 状态吗？**
   client：resolve_addr → resolve_route → create_qp（INIT）→ connect 发 REQ；server：收到 CONNECT_REQUEST → create_qp → accept（RTR → RTS）后回 REP；client 收到 REP 后 RTR → RTS 再发 RTU，双方 ESTABLISHED。RTU（Ready To Use）是 CM 消息，不是 QP 状态；QP 状态只有 RESET/INIT/RTR/RTS/SQD/SQE/ERR。

## 待深入

- `ibv_wr_*` 新 post API（`ibv_qp_ex`）与旧 `ibv_post_send` 的性能差异：perftest 提供 `--use_old_post_send` 开关（`third_party/perftest/src/perftest_parameters.c`），可以做 A/B 对比。待确认：实测差异；确认方式：在有 RNIC 的机器上分别加、不加 `--use_old_post_send` 跑 `ib_write_bw -s 64 -l 32`。
- 内核侧 uverbs 的命令通道（`cmd_ioctl.c`）与 provider `alloc_context` 的握手。
- ODP（on-demand paging）缺页时网卡侧的行为和性能代价。
- 待确认：内核 `ib_cm` 的 CM 连接状态机（各状态名及 RTU、MRA、超时重发的处理）；确认方式：克隆 Linux 内核后读 `drivers/infiniband/core/cm.c`，或读 IB 规范第 12 章。
