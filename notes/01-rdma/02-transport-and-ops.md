# 传输类型与操作语义

> 层次：下层 · 来源：third_party/rdma-core@bd3282a（`libibverbs/verbs.h`、`providers/mlx5/`）、third_party/nccl@3e46c08（`src/transport/net_ib/`）、third_party/perftest@05200c2

## 一句话

"用哪种 QP"决定可靠性和扩展性，"用哪种 opcode"决定是否需要对端 CPU 参与、怎样通知完成；通信库性能的大部分细节（inline、选择性 signal、post_list、doorbell 合并）都是在为"每条消息的软件开销"做减法。

## 在调用链中的位置

- 上层：NCCL net_ib 的数据 QP 全是 RC（`third_party/nccl/src/transport/net_ib/connect.cc` 中 `qpCreateAttrs.type = IBV_QPT_RC`），数据用 `IBV_WR_RDMA_WRITE` + 末尾一个 `IBV_WR_RDMA_WRITE_WITH_IMM`，接收端的"可以发了"通知（CTS FIFO）也是一次 RDMA WRITE（`p2p.cc`：`ncclIbMultiSend`、`ncclIbPostFifo`）。
- 下层：provider 把 WR 翻译成 WQE 写进 SQ ring，更新 doorbell record，再 MMIO 写 doorbell / BlueFlame（`third_party/rdma-core/providers/mlx5/qp.c`：`post_send_db`）。

## 核心概念

### 传输类型（`enum ibv_qp_type` 与厂商扩展）

| 类型 | 连接 | 可靠 | 支持的操作 | 消息大小 | 扩展性 |
|---|---|---|---|---|---|
| RC（`IBV_QPT_RC`） | 一对一 | 是：ACK/NAK、PSN、重传、按序 | SEND/RECV、WRITE(_IMM)、READ、ATOMIC | 最大 2^31，网卡分片 | 每对端一个 QP，N² 状态 |
| UC（`IBV_QPT_UC`） | 一对一 | 否：丢包时整条消息丢弃 | SEND、WRITE（无 READ/ATOMIC） | 可分片 | 同 RC |
| UD（`IBV_QPT_UD`） | 无连接，一对多 | 否 | 只有 SEND/RECV（含多播） | ≤ path MTU | 一个 QP 对所有端，每 WR 带 AH |
| XRC（`IBV_QPT_XRC_SEND/RECV`） | 进程间共享接收端 | 是 | 同 RC | 同 RC | 降低多进程 × 多节点的 QP 数 |
| DC（mlx5 扩展） | 动态连接：DCI 发起，DCT 接收 | 是 | 同 RC 子集 | 同 RC | 一个 DCI 可以按 WR 切换目标 |

DC 不在通用 `ibv_qp_type` 里，而是 mlx5 direct verbs 扩展：`third_party/rdma-core/providers/mlx5/mlx5dv.h` 中有 `MLX5DV_QP_INIT_ATTR_MASK_DC`、`MLX5DV_DCTYPE_DCT / MLX5DV_DCTYPE_DCI`，发送时用 `mlx5dv_wr_set_dc_addr(mqp, ah, remote_dctn, remote_dc_key)` 指定目标（man：`providers/mlx5/man/mlx5dv_create_qp.3.md` 的 `dc_init_attr` 一节）。perftest `-c` 支持 `RC/UC/UD/XRC/DC/SRD`。SRD 是 AWS EFA 的可靠无序数据报（`third_party/rdma-core/providers/efa/efadv.h`：`EFADV_QP_DRIVER_TYPE_SRD`），也是"自研传输协议"的一个公开样例。

规模问题：RC 每对端一个 QP，网卡片上 QP context 缓存放不下时会出现 cache miss，性能断崖下降。ai-infra-book 6.5.5 节给了一个算例：RoCE 可靠连接每条 512 B 上下文，与 UB 的端点 + 通道模型对比（`third_party/ai-infra-book/manuscripts/06-超节点.md`）。DC、XRC、SRD 都是在这个问题上做的取舍。

### 操作语义（`enum ibv_wr_opcode`）

| opcode | 对端 CPU / RECV | 本端完成含义 | 对端完成 |
|---|---|---|---|
| `IBV_WR_SEND` / `SEND_WITH_IMM` | 消耗对端一个 RECV WQE | RC：收到 ACK | 对端 CQ 出一个 RECV WC |
| `IBV_WR_RDMA_WRITE` | 不需要，也不通知 | RC：收到 ACK（数据已到对端网卡；是否已对 GPU/CPU 可见是另一个问题） | 无 |
| `IBV_WR_RDMA_WRITE_WITH_IMM` | 消耗对端一个 RECV WQE（buffer 可以为空） | 同 WRITE | 对端 CQ 出 `IBV_WC_RECV_RDMA_WITH_IMM`，`wc_flags & IBV_WC_WITH_IMM`，带 32 bit imm |
| `IBV_WR_RDMA_READ` | 不需要 | 数据已读回本地 | 无 |
| `IBV_WR_ATOMIC_CMP_AND_SWP` / `FETCH_AND_ADD` | 不需要 | 旧值已写回本地 | 无；8 字节，对齐 |
| 其他：`LOCAL_INV`、`BIND_MW`、`SEND_WITH_INV`、`TSO`、`FLUSH`、`ATOMIC_WRITE` | | | |

`struct ibv_send_wr` 中，`wr.rdma{remote_addr, rkey}`、`wr.atomic{remote_addr, compare_add, swap, rkey}`、`wr.ud{ah, remote_qpn, remote_qkey}` 是一个 union，`imm_data` 用网络字节序。

下图把上表的四种主要 opcode 在 RC 上画成报文往返，重点看右侧"对端是否消耗 RECV、是否出 WC"：

```text
              A (requester)                          B (responder)
              ─────────────                          ─────────────
SEND          post ──────── payload ───────────────► 消耗 1 个 RECV WQE，B 的 CQ 出 RECV WC
              CQE  ◄─────────────────────── ACK ────

RDMA_WRITE    post ──────── payload -> B's MR ─────► 不消耗 RECV、不出 WC，B 的 CPU 不知道
              CQE  ◄─────────────────────── ACK ────

WRITE_IMM     post ──────── payload + imm32 ───────► 消耗 1 个 RECV WQE，CQ 出 RECV_RDMA_WITH_IMM
              CQE  ◄─────────────────────── ACK ────

RDMA_READ     post ──────── read request ──────────► 网卡直接读 B 的 MR，不出 WC
              CQE  ◄────────────── response data ───  ← 需要完整往返，延迟约为 WRITE 的 2 倍
```

**WRITE_WITH_IMM 为什么是通信库的主力**：一个 WR 同时完成"数据落地 + 通知对端"，并且 RC 保证同一 QP 上前面的 WRITE 先于这条 IMM 生效。NCCL 正是这样用的：前面 `nreqs` 个 `IBV_WR_RDMA_WRITE`（不 signal），最后一个 `IBV_WR_RDMA_WRITE_WITH_IMM` 打 `IBV_SEND_SIGNALED`，imm 里放 size 或 request id（`ncclIbMultiSend`）。接收端需要预投 RECV，见 `connect.cc` 的 `ncclIbPostReceiveWorkRequestsOnQp`。

**READ 的特点**：受 `max_rd_atomic / max_dest_rd_atomic`（同时在途的 READ/ATOMIC 数）限制，perftest 用 `-o/--outs` 调整；READ 需要一个往返，延迟约为 WRITE 的 2 倍（perftest 测 READ 延迟时 `rtt_factor=1`，不除以 2，见 [04-perftest.md](04-perftest.md)）。NCCL 用一个 1 字节 RDMA READ 做 GDR flush（`p2p.cc`：`ncclIbIflush`），利用的是"READ 完成意味着之前写入 GPU 的 PCIe 写已经可见"这一顺序语义。

### Inline

`IBV_SEND_INLINE`：把 payload 直接拷进 WQE，网卡不需要再 DMA 读一次 buffer，省一次 PCIe 往返；而且调用返回后 buffer 立刻可以复用（`verbs.h` 中 `ibv_post_send` 的注释），也不需要 lkey。上限是创建 QP 时的 `cap.max_inline_data`（rc_pingpong 用 `ibv_query_qp` 查实际值）。只适合小消息（几十到几百字节）。NCCL 只对 CTS FIFO 这种小结构开 inline：`max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0`，并设置 `remCtsFifo.flags = IBV_SEND_INLINE`（`connect.cc`），默认关闭（`NCCL_IB_USE_INLINE=0`）。

### Signaled 与 CQ moderation（两个不同的概念）

- **Selective signaling（发送端）**：不给每个 WR 都设 `IBV_SEND_SIGNALED`，每 N 个设一次，减少 CQE 写回和 poll 次数。坑：未 signal 的 WR 只有在之后某个 signaled WR 的 CQE 被 poll 到时才算完成、才释放 SQ 槽位；如果一直不 signal，SQ 会被填满并卡死。NCCL `ncclIbPostFifo` 的长注释（引用 rdmamojo）讲的就是这一点，它让 "slot == devIndex" 时 signal，保证每个 CTS QP 都会被 drain。perftest 的 `-Q/--cq-mod` 指的就是这种"每 cq_mod 个 WR signal 一次"（`perftest_resources.c`：`ctx_set_send_wqes` 中 `(j + 1) % user_param->cq_mod == 0` 时置 `IBV_SEND_SIGNALED`）。
- **CQ moderation（中断/事件合并）**：`ibv_modify_cq` + `struct ibv_moderate_cq{cq_count, cq_period}`（`verbs.h`、`man/ibv_modify_cq.3`）：攒够 `cq_count` 个 CQE，或者等满 `cq_period` µs，才产生一次完成事件。它只影响事件通知，不改变 CQE 的数量；轮询模式下基本没有作用。

以 perftest `-Q 4` 为例（第 j 个 WR 在 `(j + 1) % 4 == 0` 时 signal），SQ 槽位的回收方式如下：

```text
  SQ slot   0     1     2     3     4     5     6     7
          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
          │ WR  │ WR  │ WR  │ WR* │ WR  │ WR  │ WR  │ WR* │   * = IBV_SEND_SIGNALED
          └─────┴─────┴─────┴──┬──┴─────┴─────┴─────┴──┬──┘
                               ▼                       ▼
                             CQE #1                  CQE #2
          poll 到 CQE #1 → slot 0..3 一起视为完成并释放
          poll 到 CQE #2 → slot 4..7 释放
          一个 WR* 都没有 → 永远没有 CQE，slot 不释放 → SQ 填满，post 卡死
```

### Doorbell

- post_send 时，provider 依次：(1) 在 SQ ring 中写 WQE；(2) `udma_to_device_barrier()` 后更新 doorbell record（`qp->db[MLX5_SND_DBR]`，网卡可以 DMA 读到的内存）；(3) MMIO 写 UAR 上的 doorbell 寄存器。见 `third_party/rdma-core/providers/mlx5/qp.c`：`post_send_db`。
- **BlueFlame**：当 `nreq == 1` 且是 inline（或 `prefer_bf`）、WQE 不大时，用 `mlx5_bf_copy` 把整个 WQE 通过 write-combining MMIO 推给网卡，网卡不需要再 DMA 读 WQE，降低小消息延迟；否则只 `mmio_write64_be` 写 8 字节控制段。代码注释还解释了为什么 `mmio_flush_writes()` 必须放在自旋锁内：两个 CPU 的 WC buffer 刷出顺序可能交错。
- **Doorbell batching / post_list**：一次 `ibv_post_send` 传一条 WR 链表（`wr->next`），只敲一次 doorbell。perftest `-l/--post_list` 就是测这个，README 说单次 post 大约是 10 Mpps 的软件上限，用 post_list 才能看到硬件的真实消息率。NCCL 的 `ncclIbMultiSend` 也是把多个 WRITE 和最后的 WRITE_IMM 串成链表后一次 post。

把上面三条串起来，一次 `ibv_post_send` 从用户态到网卡的路径是（mlx5，`providers/mlx5/qp.c`）：

```text
ibv_post_send(qp, wr_list)            wr_list：WR -> WR -> ... -> WR(SIGNALED)，wr->next 串起
   │  static inline：qp->context->ops.post_send，不进内核
   ▼
_mlx5_post_send
   │  (1) 逐条把 WR 写成 WQE 放进 SQ ring（inline 时 payload 也拷进 WQE）
   ▼
post_send_db(qp, bf, nreq, inl, size, ctrl)       ← 整条链表只调一次
   │  (2) udma_to_device_barrier()；qp->db[MLX5_SND_DBR] = sq.cur_post
   │  (3) 写 UAR doorbell（WC 映射）：
   │        ├─ nreq == 1 且 (inline 或 prefer_bf) 且 WQE 够小
   │        │     → mlx5_bf_copy：整个 WQE 推给网卡（BlueFlame）
   │        └─ 否则 → mmio_write64_be：只写 8 B 控制段，网卡再 DMA 读 WQE
   │      mmio_flush_writes()（bf->need_lock 时在自旋锁内）
   ▼
RNIC：DMA 读 payload（inline 时省掉）→ 发包 → 收 ACK → signaled WR 写 CQE → ibv_poll_cq
```

### 端到端：一次 RDMA WRITE_WITH_IMM 从提交到完成

把 PD、MR、lkey/rkey、CM、QP、WQE、doorbell、PSN/ACK、CQ/CQE 串成一条线：主机 A 用 `IBV_WR_RDMA_WRITE_WITH_IMM` 把 4 KB 写进主机 B 的内存。选它是因为两端都会用上：A 用 SQ、lkey、doorbell 和 send CQ；B 的数据按 rkey 直接落进内存，imm 消耗 B 的一个 RECV WQE 并在 B 的 recv CQ 上出一个 CQE。

驱动层对照本地 mlx5 provider（`third_party/rdma-core/providers/mlx5/`）；网卡内部的 DMA、查表、组包、ACK 本地没有源码，按 IB 规范和通用原理描述。

```text
        Host A（发送方）                                        Host B（接收方）
┌──────────────────────────────┐                       ┌──────────────────────────────┐
│ buf_a (4 KB)    MR: lkey La  │                       │ buf_b           MR: rkey Rb  │
│ PD - QP(SQ/RQ) - send_cq     │                       │ PD - QP(SQ/RQ) - recv_cq     │
│                              │                       │ RQ: 1 RECV WQE pre-posted    │
│ (1) post_send -> WQE in SQ   │                       │                              │
│ (2) doorbell (MMIO to RNIC)  │                       │ (7) poll_cq -> CQE(imm)      │
│ (6) poll_cq -> CQE           │                       │                              │
└──────────────┬───────────────┘                       └──────────────▲───────────────┘
               │ PCIe                                                 │ PCIe
┌──────────────▼───────────────┐ (4) BTH+RETH+payload  ┌──────────────┴───────────────┐
│ RNIC A                       │ ════════════════════► │ RNIC B                       │
│ (3) fetch WQE, check lkey,   │                       │ (5) check rkey, DMA write,   │
│     DMA read data, packetize │ ◄════════════════════ │     consume RECV, write CQE  │
│ (6) ACK -> write CQE         │         ACK           │                              │
└──────────────────────────────┘                       └──────────────────────────────┘
```

**阶段 0：准备资源（两端）。** `ibv_alloc_pd` → `ibv_reg_mr`（B 必须带 `IBV_ACCESS_REMOTE_WRITE`）→ `ibv_create_cq` → `ibv_create_qp`。注册 MR 会 pin 住内存、把虚拟地址到物理地址的映射写进网卡翻译表，并返回 lkey（本端网卡访问本机内存用）和 rkey（交给对端，授权对端访问）。B 还要预投一个 RECV WQE 来接 imm；WRITE 的数据不进这个 RECV 的 buffer，所以它可以不带 SGE。

**阶段 1：建链并交换 rkey。** 用 CM（见 [01-verbs-basics](01-verbs-basics.md) 的"用 RDMA CM 建链"）或手工 TCP 交换，双方 QP 到 RTS，互相知道 QPN、PSN。B 的 `buf_b` 地址和 rkey 必须告诉 A，常见做法是放进 REP 的私有数据，或连上后先用一次 SEND 发过去。网卡不会替应用交换它们。

**阶段 2：A 构造 WQE、按门铃。** 应用填 `ibv_sge{addr=buf_a, length=4096, lkey=La}` 和 `ibv_send_wr{opcode=RDMA_WRITE_WITH_IMM, send_flags=SIGNALED, imm_data, wr.rdma={remote_addr=buf_b, rkey=Rb}}`，调 `ibv_post_send`。mlx5 的 `_mlx5_post_send` 在用户态把它写成一个 WQE，段的定义在 `third_party/rdma-core/providers/mlx5/mlx5dv.h`：

```text
SQ ring（主机内存，网卡可 DMA）
 ┌───────────────────────────────────────────────────────────────┐
 │ WQE #n                                                        │
 │ ┌─ ctrl seg  (struct mlx5_wqe_ctrl_seg) ───────────────────┐  │
 │ │ opmod_idx_opcode: opcode=WRITE_IMM, wqe index n          │  │
 │ │ qpn_ds: QPN + number of 16B segments                     │  │
 │ │ fm_ce_se: CQ update bit (set when SIGNALED) | imm        │  │
 │ ├─ raddr seg (struct mlx5_wqe_raddr_seg) ──────────────────┤  │
 │ │ raddr = buf_b | rkey = Rb                                │  │  ← 远端：地址 + rkey
 │ ├─ data seg  (struct mlx5_wqe_data_seg) ───────────────────┤  │
 │ │ byte_count = 4096 | lkey = La | addr = buf_a             │  │  ← 本地：地址 + lkey
 │ └──────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────┘
```

然后 `post_send_db` 按门铃：屏障 → 更新 doorbell record → MMIO 写 UAR（小 WQE 走 BlueFlame 整个推给网卡）→ flush WC buffer，细节见上面的 Doorbell 小节。`ibv_post_send` 返回时一个字节都还没发出去，CPU 的活到此结束。

**阶段 3：A 的网卡取数据、组包。** 收到门铃后（非 BlueFlame 时）DMA 读回 WQE；用 lkey 查翻译表，核对 PD 和地址范围，把 `buf_a` 翻译成物理地址；DMA 读出 4 KB（开 GPUDirect RDMA 时读的是显存）；按 path MTU 切包。以 MTU 1024 为例：

```text
 Eth | IP (DSCP, ECN) | UDP (dst 4791) | BTH ... | ICRC
 pkt 1  BTH(WRITE_FIRST, dst QPN, PSN=p)   + RETH(VA=buf_b, rkey=Rb, len=4096) + 1 KB
 pkt 2  BTH(WRITE_MIDDLE, PSN=p+1)                                             + 1 KB
 pkt 3  BTH(WRITE_MIDDLE, PSN=p+2)                                             + 1 KB
 pkt 4  BTH(WRITE_LAST_WITH_IMM, PSN=p+3)  + ImmDt                             + 1 KB
```

rkey 和远端地址只在第一个包的 RETH 里，后面的包按 PSN 顺序接着往后写。（UDP 4791 本地源码未出现，见 [03-roce-ib-congestion](03-roce-ib-congestion.md) 的待确认。）

**阶段 4：网络。** 交换机按 DSCP 把包放进 RoCE 无损队列，拥塞时打 ECN、触发 CNP 和 DCQCN 降速，PFC 兜底（见 [03-roce-ib-congestion](03-roce-ib-congestion.md) 的"优先级与队列"）。

**阶段 5：B 的网卡写内存，B 的 CPU 不参与。** 按 BTH 的目的 QPN 找到 QP，检查 PSN 连续；对第一个包用 rkey 查表，核对 MR 与 QP 同属一个 PD、有 REMOTE_WRITE 权限、`buf_b + 4096` 不越界；DMA 写进 `buf_b`。最后一个包带 IMM：从 RQ 取出预投的 RECV WQE，在 recv CQ 写一个 CQE（`IBV_WC_RECV_RDMA_WITH_IMM`，带 imm 和 byte_len）。回 ACK，一个 ACK 可以确认到某个 PSN 为止的所有包。

**阶段 6：A 收到 ACK、产生完成。** 网卡确认 WQE #n 完成；因为 SIGNALED，往 send CQ 写一个 CQE。应用 `ibv_poll_cq` → `mlx5_poll_cq`（`third_party/rdma-core/providers/mlx5/cq.c`），同样不进内核：

```text
get_sw_cqe       看 CQE 的 owner 位（op_own & MLX5_CQE_OWNER_MASK）
                 → 与当前这一圈的期望值一致，说明是网卡新写的
mlx5_parse_cqe   解析成 ibv_wc：wr_id, status=IBV_WC_SUCCESS, opcode=IBV_WC_RDMA_WRITE
update_cons_index 更新 CQ 的 doorbell record，告诉网卡这些槽位可以复用
```

CQ 是环，网卡每写满一圈，owner 位的期望值翻转一次。软件不加锁、不读网卡寄存器，只看一个 bit 就能分清新 CQE 和上一圈留下的旧数据。

**阶段 7：B 得知数据已到。** B poll recv CQ 拿到 imm 和 byte_len，就知道 4 KB 已经在 `buf_b` 里了，然后补投一个 RECV。如果用的是不带 IMM 的 RDMA WRITE，B 既不消耗 RECV 也不出 CQE，B 的 CPU 完全不知道数据写进来了，只能靠应用约定（轮询尾字节、再发一次 SEND），见下面"性能要点与常见坑"。

各概念在这条线上的位置：

| 概念 | 作用 | 阶段 |
|---|---|---|
| PD | QP 和 MR 必须同属一个 PD，收发两端网卡都会校验 | 0、3、5 |
| MR | 注册过的内存：pin 住，并写进网卡翻译表 | 0 |
| lkey | 本端访问自己 MR 的钥匙，写在 WQE 的 data seg | 2、3 |
| rkey | 授权对端访问本端 MR，写在 raddr seg 和包的 RETH | 1（交换）、2、5（校验） |
| CM | 交换 QPN/PSN、推进 QP 状态、可顺带传 rkey | 1 |
| QP（SQ + RQ） | 通信端点，到 RTS 才能发 | 0、1 |
| WQE | SQ/RQ 环里的一条工作描述 | 0（RECV）、2（WRITE） |
| doorbell / BlueFlame | 通知网卡有新 WQE；小 WQE 随门铃直接推给网卡 | 2 |
| PSN / ACK | 包序号与确认，可靠传输的基础 | 3、5、6 |
| CQ / CQE | 完成队列与条目，靠 owner 位分新旧 | 5（B）、6（A） |

出错时：

| 情况 | 现象 |
|---|---|
| rkey 错、越界、没有 REMOTE_WRITE | B 回 NAK；A 拿到 `IBV_WC_REM_ACCESS_ERR`，QP 进 ERR |
| B 没有预投 RECV（WITH_IMM 或 SEND） | B 回 RNR NAK，A 按 `rnr_retry` 重试，用完后报 `IBV_WC_RNR_RETRY_EXC_ERR` |
| 丢包，PSN 不连续 | B 回 NAK（sequence error），A 从丢失的 PSN 重发（Go-back-N） |
| 收不到 ACK | 按 `timeout`、`retry_cnt` 重传，用完后报 `IBV_WC_RETRY_EXC_ERR`（NCCL 拥塞时常见） |
| lkey 错 | A 本地报 `IBV_WC_LOC_PROT_ERR`，包根本发不出去 |

## 源码导读（路径 + 函数 + 作用）

| 位置 | 看什么 |
|---|---|
| `third_party/rdma-core/libibverbs/verbs.h`：`enum ibv_wr_opcode`、`enum ibv_send_flags`、`struct ibv_send_wr` | opcode 全集；`IBV_SEND_FENCE/SIGNALED/SOLICITED/INLINE/IP_CSUM` |
| 同上：`IBV_QP_EX_WITH_*`、`ibv_wr_rdma_write`、`ibv_wr_rdma_write_imm` | 新 post API 需要在创建 QP 时声明要用的 opcode 集合 |
| 同上：`struct ibv_moderate_cq`、`ibv_modify_cq` | CQ moderation |
| `third_party/rdma-core/providers/mlx5/qp.c`：`_mlx5_post_send`、`post_send_db`、`mlx5_bf_copy` | WQE 构造与 doorbell / BlueFlame |
| `third_party/rdma-core/providers/mlx5/mlx5dv.h`：`struct mlx5_wqe_ctrl_seg`、`mlx5_wqe_raddr_seg`、`mlx5_wqe_data_seg`、`mlx5_cqe64` | WQE 各段（控制段、远端地址 + rkey、本地地址 + lkey）与 CQE 的布局 |
| `third_party/rdma-core/providers/mlx5/cq.c`：`mlx5_poll_cq`、`get_sw_cqe`、`mlx5_parse_cqe`、`update_cons_index` | poll CQ：owner 位判新旧、解析成 `ibv_wc`、更新消费者索引 |
| `third_party/rdma-core/providers/mlx5/mlx5dv.h`：`MLX5DV_DCTYPE_*`、`mlx5dv_wr_set_dc_addr`、`MLX5DV_QP_CREATE_OOO_DP` | DC；OOO_DP 允许接收端乱序消费 RECV WR、发送端不保证到达顺序（man `mlx5dv_create_qp.3.md`）。NCCL 的 `NCCL_IB_OOO_RQ` 用的就是它（`connect.cc`：`ncclIbCreateQpMlx5`） |
| `third_party/rdma-core/libibverbs/examples/ud_pingpong.c` | UD + AH + qkey + 40 B GRH |
| `third_party/nccl/src/transport/net_ib/p2p.cc`：`ncclIbMultiSend`、`ncclIbPostFifo`、`ncclIbIflush` | 生产级用法：WRITE 链 + WRITE_IMM、CTS 用 inline WRITE、READ 做 flush |

## 性能要点与常见坑

- 小消息瓶颈在消息率（Mpps）而不是带宽：每个 WR 的软件开销加上 PCIe 往返决定上限。可用的手段：inline、BlueFlame、post_list、selective signaling、多 QP 多线程。ai-infra-book 7.3.4 节用"启动间隔 δ"建模这个上限（`third_party/ai-infra-book/manuscripts/07-数据中心网络.md`）。
- WRITE 完成 ≠ 对端应用看到：对端需要另一种机制来"知道"数据到了，比如 IMM、轮询 buffer 尾字节（perftest `ib_write_lat` 就是轮询最后一个字节），或者再发一个 SEND。如果目标是 GPU 显存，还涉及 PCIe 写顺序 / flush 问题（见 [05-gpudirect-and-nccl-ib.md](05-gpudirect-and-nccl-ib.md)）。
- 轮询 buffer 尾字节依赖"按地址递增写入"这个实现行为。在乱序 / 多路径传输下（OOO、包喷洒）这个假设不再成立，必须改用 IMM 或完成记录。这正是新协议会打破通信库旧假设的典型例子。
- `IBV_SEND_FENCE`：让本 WR 等前面的 READ/ATOMIC 完成后再执行，用于"先 READ 再 WRITE"一类依赖。
- UC 没有 READ/ATOMIC；UD 消息不能超过 MTU，大消息要在软件里分片和重组。

## 面试自测（问题 + 要点答案）

1. **RC、UC、UD 各适合什么场景？为什么集合通信库基本都用 RC？**
   RC 可靠、按序、支持单边操作，大消息由网卡分片，正好满足集合通信的大块、可靠搬运。UD 扩展性好，但只能 SEND、单包不超过 MTU、不可靠，适合控制面和小消息。UC 很少使用。大规模下 RC 的 QP 数问题用 DC/XRC/SRD 或者"少 QP + 软件复用"来缓解。
2. **WRITE_WITH_IMM 和 "WRITE + SEND" 相比有什么优势？**
   少一个 WR、少一次网卡处理，同 QP 上保证数据先于通知可见；imm 能带 32 bit 元数据（NCCL 放 size 或 request id）。代价是仍要对端预投 RECV。
3. **inline 的收益和限制？**
   省掉网卡 DMA 读 payload 的一次 PCIe 往返、buffer 可以立即复用、不需要 lkey；受 max_inline_data 限制，而且是 CPU memcpy，大消息反而更慢。
4. **selective signaling 的坑？和 CQ moderation 的区别？**
   未 signal 的 WR 要靠后续 signaled WR 的 CQE 才能回收 SQ 槽位，长期不 signal 会导致 SQ 满、死锁。CQ moderation 是合并"完成事件 / 中断"，CQE 数量不变。
5. **讲讲 doorbell 和 BlueFlame。**
   先写 WQE，然后屏障 + 更新 DBR，最后 MMIO 写 UAR。BlueFlame 把整个小 WQE 通过 WC 映射直接写进网卡，省掉网卡回读 WQE；多 CPU 共享 BF 寄存器时要加锁并 flush WC buffer。
6. **RDMA READ 在通信库里有什么用处？**
   接收方拉数据（rendezvous 中的 GET 模式）、GDR flush（NCCL `ncclIbIflush`）、链路探测（NCCL resiliency 用 READ 做 probe，见 `p2p_resiliency.cc`）。
7. **DC 解决了什么问题？代价是什么？**
   一个 DCI 可以按 WR 切换目标，QP 数从 O(N) 降到 O(1)；代价是切换目标时有隐式的连接建立 / 拆除开销，同一 DCI 上要串行化，并且是厂商私有扩展。
8. **完整讲一次 RDMA WRITE_WITH_IMM：从 post_send 到两端拿到 CQE。**
   要点：注册 MR 得到 lkey/rkey，建链时把对端地址和 rkey 交给发送方；post_send 在用户态写 WQE（ctrl + raddr/rkey + data/lkey）、更新 doorbell record、MMIO 敲门铃；发送网卡用 lkey 查表、DMA 读数据、按 MTU 切包，首包 RETH 带 rkey；接收网卡校验 rkey/PD/权限/范围后 DMA 写内存，末包带 IMM 时消耗一个 RECV WQE 并写 CQE，回 ACK；发送网卡收 ACK 后为 signaled WR 写 CQE，poll_cq 靠 owner 位识别新 CQE。

## 待深入

- 待确认：mlx5 的 WQE 合并（"enhanced MPW"）在哪里实现、是否对 RC 生效。在 `third_party/rdma-core/providers/mlx5/qp.c` 中 grep `mpw|MPW` 没有结果；确认方式：`grep -rni "mpw" third_party/rdma-core/providers/mlx5/`。
- `ibv_query_qp_data_in_order`（`verbs.h`）的语义，以及它和 GPU 侧可见性的关系。
- ATOMIC 的 PCIe 原子支持与 GPU 显存上的原子（NVSHMEM 信号）。
