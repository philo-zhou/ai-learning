# IB vs RoCEv2、拥塞控制与大规模 RDMA 的问题

> 层次：下层 · 来源：third_party/ai-infra-book@3bdcb4fc 第 7 章（7.5 节）、第 6 章（6.5.5 节）；third_party/rdma-core@bd3282a；third_party/nccl@3e46c08；third_party/perftest@05200c2；外部资料 [RoCE QoS 配置调研](../../docs/references/2026-10-17-roce-qos-config.md)

## 一句话

RDMA 的可靠传输假设网络"几乎不丢包"。IB 靠链路级 credit 流控做到这一点，RoCEv2 靠 PFC + ECN/DCQCN 在以太网上模拟出来。AI 集群把同步的、突发的、多对一的集合通信流量压到这套机制上，于是 incast、PFC 风暴/死锁、ECMP 哈希冲突就成了主要瓶颈。"自研 RDMA 协议 / 拥塞算法"就是在重写这一层的取舍。

## 在调用链中的位置

- 上层：通信库只能间接影响这一层，手段包括 traffic class / SL（选优先级队列）、QP 数量（影响 ECMP 熵）、发送节奏（错峰、限速）、消息切分粒度。在 NCCL 中对应 `NCCL_IB_TC`、`NCCL_IB_SL`、`NCCL_IB_QPS_PER_CONNECTION`、`NCCL_IB_ADAPTIVE_ROUTING` 等（`third_party/nccl/src/transport/net_ib/connect.cc`、`init.cc`）。
- 本层：RNIC 内的传输协议（PSN、ACK/NAK、重传）和拥塞控制（反应点 RP / 通知点 NP），交换机的队列管理（ECN 标记、PFC 暂停、路由选择）。
- 下层：物理链路、缓冲区大小、线缆长度（决定 PFC headroom）。

## 核心概念

### IB vs RoCEv2

| 维度 | InfiniBand | RoCEv2 |
|---|---|---|
| 链路层 | IB 链路，基于 credit 的逐跳流控，天然无损 | 以太网，需要 PFC（IEEE 802.1Qbb）实现无损 |
| 网络层 / 寻址 | 子网内 LID（由 SM 分配），跨子网用 GRH/GID | IP（GID 嵌入 IPv4/IPv6 地址），可以三层路由 |
| 封装 | IB LRH + BTH + payload | Eth + IP + UDP + BTH + payload + ICRC |
| 管理 | 集中式 Subnet Manager 算路由（perftest README 提醒 IB 上必须有 SM 运行） | 分布式路由（BGP/ECMP），运维模型与以太网一致 |
| 拥塞控制 | IB CC（FECN/BECN）；交换机自适应路由（AR） | ECN 标记 + CNP 回传 + DCQCN（或厂商算法） |
| 多路径 | AR 可以逐包选路，网卡处理乱序 | ECMP 按五元组哈希，同一 QP 走同一条路径 |

表中"封装"一行画成报文头对比：两者共用 IB 传输头 BTH，区别在 BTH 外面包的是 IB 链路/网络头还是以太网 + IP + UDP。

```text
InfiniBand   ┌─────┬───────┬─────┬───────────┐
             │ LRH │ [GRH] │ BTH │  payload  │     ← LRH 里是 SM 分配的 LID；GRH 只在跨子网时需要
             └─────┴───────┴─────┴───────────┘

RoCEv2       ┌─────┬────────┬─────┬─────┬───────────┬──────┐
             │ Eth │ IPv4/6 │ UDP │ BTH │  payload  │ ICRC │
             └─────┴───┬────┴──┬──┴─────┴───────────┴──────┘
                       │       └── dst port 4791（待确认，见下）；src port 作 ECMP 熵
                       └── DSCP/ECN 字段：QP 的 traffic_class 映射到这里（映射待确认）；交换机在这里打 ECN CE
```

- **UDP 4791**：RoCEv2 的目的 UDP 端口固定为 4791；源端口通常由网卡按 QP 生成，作为 ECMP 熵。待确认：本地源码中没有出现 4791（对 rdma-core、nccl、perftest 都 grep 过）；确认方式：查 IBTA RoCEv2 Annex（A17），或在 RoCE 网卡上 `tcpdump -i <if> udp port 4791`。待确认：源端口的具体生成方式（按 QPN 哈希还是可配置）；确认方式：查厂商文档或抓包对比不同 QP。
- **GID 类型**：`verbs.h` 的 `IBV_GID_TYPE_IB / ROCE_V1 / ROCE_V2`。sysfs 中每个 GID index 的类型可以在 `/sys/class/infiniband/<dev>/ports/<port>/gid_attrs/types/<idx>` 读到，内容是 `"IB/RoCE v1"` 或 `"RoCE v2"`。NCCL 的 `ncclIbRoceGetVersionNum` 就是读这个文件（`connect.cc`），再结合 `NCCL_IB_ROCE_VERSION_NUM`（默认 2）、`NCCL_IB_ADDR_FAMILY`、`NCCL_IB_ADDR_RANGE` 在 `ncclUpdateGidIndex` 中自动挑选 GID index；`NCCL_IB_GID_INDEX` 可以强制指定。
- RoCE 在 RTR 时，NCCL 设置 `ah_attr.is_global=1`、`grh.traffic_class = tc`、`hop_limit = 255`、`flow_label = 0`（`ncclIbQpRtr`）。traffic_class 映射到 IP DSCP/ECN 字段，交换机据此把流量放进无损优先级队列。TC 值到 DSCP 的映射：NVIDIA 文档写明 ToS = DSCP × 4（高 6 位是 DSCP，低 2 位是 ECN），见 [RoCE QoS 配置调研](../../docs/references/2026-10-17-roce-qos-config.md) Q3。

### PFC（链路流控，一跳）

接收端口某优先级队列超过阈值（XOFF）时，向上游发 PAUSE 帧暂停该优先级，降到阈值以下（XON）再恢复。必须预留 **headroom**，用来容纳"暂停帧在路上 + 已在线上的数据"。ai-infra-book 7.5.2 节的算例：30 m 线缆、50 GB/s 链路，一跳反馈约 0.33 µs，incast N=64 时 headroom 约 1.02 MB；它同时引用 DCQCN 论文的一个例子，每端口每优先级 22.4 KB（`third_party/ai-infra-book/manuscripts/07-数据中心网络.md`）。

PFC 的副作用：
- **Head-of-line blocking / 拥塞扩散**：暂停是按端口 + 优先级的，不区分流，一条拥塞流会把共享该优先级的无辜流也停掉，并且逐跳向上游扩散（PFC storm）。
- **PFC 死锁**：多台交换机之间的缓冲依赖成环（例如路由异常导致绕环、链路故障后的重路由），环上所有队列都在等下游释放，于是永久停滞。ai-infra-book 7.5.4 节从"资源依赖图成环"讲死锁的一般原理，解法是规定资源申请顺序、为响应预留独立缓冲 / 虚拟通道，打破环。对应到 PFC，就是 up-down 路由约束、PFC watchdog、限定无损优先级的使用范围。

### ECN + DCQCN（端到端拥塞控制）

- 交换机出口队列超过阈值时，按概率给报文打 ECN CE 标记（类似 RED 曲线：Kmin/Kmax/Pmax）。
- 接收网卡（NP）看到 CE 后，向发送方回 **CNP**（拥塞通知包），有频率限制。
- 发送网卡（RP）收到 CNP 后，按 `Rc ← Rc·(1 − α/2)` 降速；α 是对拥塞程度的 EWMA 估计；在没有 CNP 的周期里，经过 fast recovery → additive increase → hyper increase 几个阶段逐步恢复速率。
- 待确认：上面公式和阶段的具体参数名（如 rate_reduce_monitor_period、rpg_* 等）是按 DCQCN 论文（Zhu et al., SIGCOMM 2015）的记忆写的，本地源码和书中没有原文；确认方式：读 DCQCN 论文第 3 节，或在 mlx5 网卡上查看 `/sys/class/net/<if>/ecn/` 下的参数（该路径已核实为 MLNX_OFED 的 ECN 配置入口，上游内核驱动则用 debugfs `cc_params/`，见 [RoCE QoS 配置调研](../../docs/references/2026-10-17-roce-qos-config.md) Q6）。论文在数据中心使用的参数：g = 1/256、定时器 55 µs、字节计数器 10 MB、CNP 间隔 50 µs（Q8）。

ai-infra-book 7.5.2 给出的分工很关键：**PFC 管一跳（约 0.33 µs），DCQCN 管一个 RTT（约 20 µs）**。incast 时队列涨得比端到端反馈快得多（N=8 时 1 MiB 缓冲 3.5 µs 就满了），所以只靠端到端反馈必然丢包，需要 PFC 兜底；而 PFC 只能挡，真正降速还得靠 DCQCN，否则暂停解除不了。书中给出的只靠端到端反馈不丢包的条件为 (N−2)×50 GB/s×20 µs ≤ Q_free，1 MiB 缓冲时只有 N≤3。

下图把两个反馈环分开画：① 是 DCQCN（跨整条路径的端到端环），② 是 PFC（只在相邻两跳之间）。

```text
① DCQCN：端到端环，约一个 RTT（≈ 20 µs）

  ┌──────────┐  data  ┌──────────────┐  data + CE  ┌──────────┐
  │ RP       │═══════►│ CP (switch)  │════════════►│ NP       │
  │ sender   │        │ egress queue │             │ receiver │
  │ RNIC     │        └──────────────┘             │ RNIC     │
  └──────────┘                                     └────┬─────┘
       ▲                                                │
       └─────────────────── CNP ◄───────────────────────┘

  CP：队列超过阈值时按概率打 ECN CE（Kmin/Kmax/Pmax，类似 RED）
  NP：看到 CE → 回 CNP（有频率限制）
  RP：收到 CNP → Rc ← Rc·(1 − α/2)；无 CNP 的周期里 fast recovery → additive → hyper increase

② PFC：逐跳环，约 0.33 µs，只挡不降速

  ┌──────────┐  data  ┌──────────────┐  data  ┌──────────────┐
  │ upstream │═══════►│ switch port  │═══════►│ downstream   │
  └──────────┘        │ prio queue   │        └──────────────┘
       ▲              └──────┬───────┘
       │                     │ queue > XOFF
       └── PAUSE(prio) ◄─────┘

  降到 XON 以下再恢复；headroom 吸收"PAUSE 在路上 + 线上已有"的数据
  暂停按"端口 + 优先级"生效、不分流 → 无辜流也被停，并逐跳向上游扩散（PFC storm）
```

厂商还提供了"按 QP 选择拥塞算法"的接口，可以作为自研算法如何接入 verbs 的参考：华为 HNS 的 `hnsdv_create_qp` 有 `congest_type`，可选 `HNSDV_QP_CREATE_ENABLE_DCQCN / LDCP / HC3 / DIP`（`third_party/rdma-core/providers/hns/hnsdv.h`、`providers/hns/man/hnsdv_create_qp.3.md`）；perftest 对应参数 `--congest_type=<DCQCN, LDCP, HC3, DIP>`（`third_party/perftest/src/perftest_parameters.c`：`set_congest_type`）。

### 优先级与队列：PFC 帧、CNP、RoCEv2 数据、TCP 怎么排

以下数值和命令的出处、置信度见 [RoCE QoS 配置调研](../../docs/references/2026-10-17-roce-qos-config.md)（下文用 Q 编号引用）。

先分清两件事：**优先级**决定一个包进交换机或网卡的哪个队列，依据是 IP 头的 DSCP（三层网络）或 VLAN 头的 PCP（二层网络），一共 8 个优先级；**ECN** 是队列内部的拥塞标记，**PFC** 是作用在某个优先级上的"暂停"开关。它们不是同一维度的东西。

从高到低，四类流量的顺序是：

```text
优先级高
  ▲
  │  PFC 帧（PAUSE）     不进任何优先级队列，MAC 层直接插队发出
  │ ───────────────────────────────────────────────────────────
  │  CNP                 prio 6，严格优先（strict）：有包就先发
  │ ───────────────────────────────────────────────────────────
  │  RoCEv2 数据         prio 3，无损（开 PFC + ECN），ETS 权重大
  │  TCP / 其他          prio 0，有损（不开 PFC），ETS 权重小
  ▼                      ↑ 这两个之间按权重分带宽，不是严格先后
优先级低
```

| 流量 | 在哪 | 怎么优先 | 为什么 |
|---|---|---|---|
| PFC 帧 | MAC 控制帧（EtherType 0x8808），不带 VLAN tag，不属于 8 个优先级（Q9） | 当前帧发完就插入，不排队；由此推出它自己不会被暂停 | 它是"让上游停下"的信号，晚到一点，下游缓冲区就溢出 |
| CNP | prio 6 / TC 6（Cumulus 默认：DSCP 48，Q1） | strict 调度，先于所有 ETS 队列发出 | 它是"降速"信号，到得晚，拥塞会继续累积，最后只能靠 PFC 兜底；CNP 包小、数量少，strict 也不会饿死别人 |
| RoCEv2 数据 | prio 3 / TC 3（Cumulus 默认：DSCP 26，在 prio 3 上开 PFC，Q1） | ETS/DWRR 加权，权重大 | 要保证带宽和不丢包，又不能把 TCP 完全饿死 |
| TCP、管理流量 | prio 0 | ETS/DWRR 加权，权重小 | 能丢包重传；SSH、监控、存储都靠它，要保底带宽 |

两个容易混的地方：

1. **"PFC 的优先级"有两层意思**：PFC 帧本身，它最先发；以及开了 PFC 的那个优先级（prio 3），它的特点是不丢包，而不是"发得更早"。
2. **RoCE 和 TCP 之间不是严格优先**。RoCE 如果设成 strict，训练和推理的大流量期间 TCP 一个包也发不出去。

网卡和交换机两边都有这 8 个队列，三处配置必须对齐：DSCP→prio 映射、开 PFC 的优先级、ETS 权重。一条 RoCE 流经过的队列：

```text
发送端主机                     交换机（每个出端口 8 个队列）              接收端主机
┌──────────────────┐          ┌──────────────────────────────┐          ┌──────────────────┐
│ QP  -> TC3 queue │  DSCP 26 │ ingress: DSCP -> prio        │          │ RNIC rx buffer   │
│ TCP -> TC0 queue │ ───────► │   count buffer per prio      │ ───────► │   PFC per prio   │
│ CNP -> TC6 queue │          │   > XOFF: PAUSE upstream     │          │   CE -> send CNP │
│ egress sched:    │          │ egress: 8 queues             │          └──────────────────┘
│   strict + ETS   │          │   Q3 > Kmin: mark ECN        │
└──────────────────┘          │   Q6 strict, Q3/Q0 ETS       │
                              └──────────────────────────────┘
```

网卡也会主动发 PFC：主机侧接不住（PCIe 或内存慢）时，网卡收包缓冲区满，就向交换机发 PAUSE。所以排查 PFC 风暴时，源头有时在主机网卡。

ECN 阈值和 PFC 阈值的先后关系是配置的核心：ECN 先触发、端到端降速，PFC 只兜底。反过来 PFC 会频繁触发，带来队头阻塞、拥塞扩散甚至死锁。

```text
队列长度 ──────────────────────────────────────────────────────►
         0      Kmin           Kmax                XOFF
         │ 不标记 │  按概率标 ECN   │   全部标记     │ 发 PFC PAUSE
                   └── DCQCN 端到端降速 ──┘            └ 逐跳暂停（兜底）
```

### 网卡侧配置：命令与常见取值

以 NVIDIA/Mellanox ConnectX 网卡、Linux、root 权限为前提。必须和交换机配置一起改，只改一边会让 RoCE 掉进有损队列。

**1. 先看现状（只读）**

```bash
ibdev2netdev                 # RDMA 设备 ↔ 网口名
mlnx_qos -i eth0             # trust、PFC、ETS、dscp→prio、prio→tc、buffer 一次看全
show_gids                    # 每个 GID index 对应 RoCE v1/v2、哪个 IP（选 NCCL_IB_GID_INDEX）
ethtool -a eth0              # 802.3x 全局 PAUSE 状态
```

**2. 分类与 PFC（`mlnx_qos`，选项名见 Q5）**

```bash
mlnx_qos -i eth0 --trust dscp                    # 按 DSCP 判定优先级
mlnx_qos -i eth0 --dscp2prio set,26,3            # RoCE 数据 → prio 3
mlnx_qos -i eth0 --dscp2prio set,48,6            # CNP → prio 6
mlnx_qos -i eth0 --pfc 0,0,0,1,0,0,0,0           # 只在 prio 3 开 PFC（8 位对应 prio 0..7）
mlnx_qos -i eth0 --prio2buffer 0,0,0,1,0,0,0,0   # prio 3 单独一个 buffer
mlnx_qos -i eth0 --cable_len 7                   # 线缆长度（米），驱动据此算 headroom 和 XOFF/XON
# --buffer_size 可以手动指定每个 buffer 的大小，单位是字节（help 文本写成百分比是错的），一般让驱动自动算
ethtool -A eth0 rx off tx off                    # 关全局 PAUSE（依据 IEEE PFC 提案，Q10）
```

**3. 调度（ETS）**

```bash
mlnx_qos -i eth0 --prio_tc 0,1,2,3,4,5,6,7 \
                 --tsa ets,ets,ets,ets,ets,ets,strict,strict \
                 --tcbw 40,0,0,60,0,0,0,0        # 权重是示例：TC0(TCP) 40%、TC3(RoCE) 60%
```

**4. RoCE 流量打哪个 DSCP**

ToS = DSCP × 4（左移 2 位，Q3），所以 DSCP 26 → ToS 104；常见的 106 是 104 再加上 ECN 位 `10`（ECT，表示这个包支持 ECN）。

```bash
echo 106 > /sys/class/infiniband/mlx5_0/tc/1/traffic_class  # 设备级强制，影响所有 QP（Q4）
cma_roce_tos -d mlx5_0 -t 106                               # 只影响 RDMA_CM 建立的 QP（Q4）
export NCCL_IB_TC=106                                       # NCCL 在 QP 上设置
ib_write_bw -d mlx5_0 --tclass=106 -x 3 <server_ip>         # perftest 验证
```

**5. ECN / DCQCN 开关（MLNX_OFED sysfs，Q6）**

```bash
echo 1 > /sys/class/net/eth0/ecn/roce_np/enable/3   # 接收端：看到 CE 就回 CNP
echo 1 > /sys/class/net/eth0/ecn/roce_rp/enable/3   # 发送端：收到 CNP 就降速
ls /sys/class/net/eth0/ecn/roce_np/params/           # CNP 的 DSCP/优先级、最小间隔等参数
# 上游内核驱动不走这套 sysfs，用 debugfs 的 cc_params/（np_cnp_dscp、np_cnp_prio、np_min_time_between_cnps）
```

**6. PFC 风暴保护（Q7）**

```bash
ethtool --get-tunable eth0 pfc-prevention-tout
ethtool --set-tunable eth0 pfc-prevention-tout 100   # 毫秒；0 = 关闭，65535 = auto（mlx5 为 100 ms），范围 80~8000
```

**7. 不用厂商工具：iproute2 `dcb`（Q11）**

```bash
dcb app add dev eth0 dscp-prio 26:3 48:6
dcb pfc set dev eth0 prio-pfc all:off 3:on
dcb ets set dev eth0 tc-tsa all:ets 6:strict tc-bw all:0 0:40 3:60
```

如果交换机通过 LLDP/DCBX 下发 DCB 配置、网卡自动接受，手工改的会被覆盖。先确认 DCBX 由谁管理。

**8. 验证（计数器名见 Q12）**

```bash
ethtool -S eth0 | grep -E "prio3_pause|pause_storm"
grep . /sys/class/infiniband/mlx5_0/ports/1/hw_counters/{np_ecn_marked_roce_packets,np_cnp_sent,rp_cnp_handled,out_of_sequence,out_of_buffer}
```

| 现象 | 说明 |
|---|---|
| 压测时 `np_cnp_sent`、`rp_cnp_handled` 在涨，PFC pause 很少 | ECN 先生效、PFC 兜底，理想状态 |
| PFC pause 涨得快，CNP 几乎不涨 | ECN 没开，或 ECN 阈值高于 XOFF |
| `out_of_sequence` 在涨 | 有丢包，RoCE 很可能没进无损队列，先查 trust 和 TC |

**常见取值汇总**

| 配置项 | 常见值 | 依据 |
|---|---|---|
| trust | `dscp` | 三层 RoCEv2 网络的通行做法 |
| RoCE 数据 | DSCP 26 → prio 3，ToS 106 | Q1、Q3 |
| CNP | DSCP 48 → prio 6，strict | Q1（交换机侧）；网卡侧默认值未核实（Q2） |
| 开 PFC 的优先级 | 只开 prio 3 | Q1；开得越多，死锁和扩散风险越大，每个端口要多预留一份 headroom |
| headroom、XOFF/XON | 驱动或交换机按线缆长度、速率、MTU 自动算 | 估算见上文 PFC 小节的书中算例 |
| 网卡风暴保护 | auto = 100 ms | Q7 |
| ECN 阈值 | DCQCN 论文（40G）：Kmin 5 KB、Kmax 200 KB、Pmax 1% | Q8；100G/400G 上的推荐值以厂商文档为准 |
| 阈值关系 | Kmin < Kmax < XOFF | ECN 先于 PFC |
| DCQCN | 论文：g = 1/256，定时器 55 µs，字节计数器 10 MB，CNP 间隔 50 µs | Q8；网卡上一般保持默认 |
| ETS 权重 | RoCE 50%~80%，TCP 剩余 | 部署经验，示例值 |
| `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT` | 20 / 7（默认） | [05-gpudirect-and-nccl-ib](05-gpudirect-and-nccl-ib.md) |

### Go-back-N vs 选择重传

- 经典 RoCE RC 网卡实现是 **Go-back-N**：接收端看到 PSN 跳号就回 NAK(sequence error)，并丢弃后续乱序包；发送端从丢失的 PSN 开始重发整个窗口。实现简单，接收端不需要乱序缓存，但丢包代价是重发整个 BDP。这也是 RoCE 必须依赖无损网络的根本原因。
- **选择重传（SACK）**：接收端缓存乱序包，只补发缺口。ai-infra-book 7.5.3 节的算例：8 个 4 KiB 包丢首包时，选择重传只补 4 KiB，回退 N 步要再发 32 KiB；书中还引用 OpenURMA 仿真，指出选择确认下吞吐随丢包率平缓下降。Ultra Ethernet 把"包喷洒 + 选择重传"写进了传输层（同节）。
- 乱序容忍是多路径的前提：逐包喷洒必然乱序，Go-back-N 会把乱序误判为丢包。mlx5 的 `MLX5DV_QP_CREATE_OOO_DP`（允许乱序到达和乱序消费 RECV，见 `mlx5dv_create_qp.3.md`）和 NCCL 的 `NCCL_IB_OOO_RQ` 就是在往这个方向走。
- 待确认："经典 RoCE 网卡 = Go-back-N"指的是 IBTA 规范语义加上主流实现；新一代网卡已有选择重传（厂商资料）。本地源码中没有读到网卡固件的重传实现；确认方式：查厂商白皮书或 IBTA 规范 9.7 节。

用上面 ai-infra-book 7.5.3 的算例（8 个 4 KiB 包，首包丢失）对比两种重传：

```text
  PSN          0     1     2     3     4     5     6     7      （每包 4 KiB）
  arrive       X     ok    ok    ok    ok    ok    ok    ok     ← PSN 0 丢失

  Go-back-N：接收端收到 PSN 1 发现跳号 → 回 NAK(sequence error)，丢弃 1..7
  resend       0     1     2     3     4     5     6     7      = 32 KiB

  选择重传：接收端缓存 1..7，只报告缺口 0
  resend       0                                              = 4 KiB
```

### AI 集群中的大规模 RDMA 问题

| 问题 | 机理 | ai-infra-book 中的位置 | 通信库 / 协议侧手段 |
|---|---|---|---|
| **Incast** | ReduceScatter/AllToAll 若所有 rank 按同一顺序发，N−1 条流同时打向一个出口，到达速率 (N−1)B | 7.5.2 "多对一汇聚与链路流控"，带 N=8/16/64 的表 | 错开发送顺序（按 rank 偏移），限制每个目标的在途量，接收端驱动（CTS/credit） |
| **周期性突发叠加** | 多个作业的通信高峰对齐，平均利用率 40% 也会排队 1 GB | 7.5.1 例题 7.5、CASSINI | 作业级错峰调度、拥塞控制 |
| **PFC 风暴 / 死锁** | 逐跳暂停扩散；缓冲依赖成环 | 7.5.2 末尾、7.5.4 | 限制无损域、PFC watchdog、拓扑 / 路由约束；新协议转向"有损 + 快速选择重传"，摆脱 PFC |
| **ECMP 哈希冲突** | 流级哈希下，n 条流落到 m 条上联，最忙链路拖慢整组；n≈m 时最差（32 流 / 32 上联只剩 28.3% 速度） | 7.5.3 "多条路径上的哈希冲突"，表与图 7-48 | rail 对齐拓扑（7.2.5）、多 QP 增加熵（`NCCL_IB_QPS_PER_CONNECTION`）、自适应路由（`NCCL_IB_ADAPTIVE_ROUTING`）、逐包喷洒 + 乱序容忍 |
| **多路径时延差** | 喷洒到时延不同的路径，按序交付要等慢路径 | 7.5.3 图 7-45/46、"逐包喷洒"8 MiB 算例 | 按路径质量加权分配，接收端乱序缓存 |
| **连接状态规模** | RC 每对端一个 QP，片上 context 缓存溢出 | 6.5.5（512 B/连接 vs UB 端点 + 通道）、7.4.2 | DC/SRD/共享传输通道、少 QP 设计 |
| **小消息消息率** | 每请求固定启动开销 δ | 7.3.4 | 批量 post、设备发起（IBGDA） |

NCCL 中与多路径相关的实际代码：
- `ncclIbDevs[...].ar` 默认对 IB 链路层为 1、对 RoCE 为 0，可用 `NCCL_IB_ADAPTIVE_ROUTING` 覆盖（`init.cc`）。
- 开启 AR 且消息大于 `ncclIbArThreshold`（默认 8192，`NCCL_IB_AR_THRESHOLD`）时，`ncclIbMultiSend` 会把数据作为单独的 RDMA WRITE 发送，再追加一个 WRITE_WITH_IMM 做通知（`p2p.cc`）。待确认：这样拆分的原因（推测是 AR 下报文可能乱序到达，把通知和大块数据分开以保证通知最后生效）；确认方式：读 NCCL release notes，或在 `git -C third_party/nccl log -S "ArThreshold"` 中找引入它的提交说明。
- 每个连接的 QP 数取 `NCCL_IB_QPS_PER_CONNECTION × ndevs`，数据按设备速率权重切分到各 QP（`ncclIbMultiSend` 中的 `weights` 与 `IB_WRITE_CHUNK_ALIGNMENT`），`NCCL_IB_SPLIT_DATA_ON_QPS` 控制一条请求是否铺到所有 QP（`common.h`：`ncclIbCommBaseGetNqpsPerRequest`）。

### 与 JD "自研 RDMA 协议、拥塞算法" 的关系

1. **为什么要自研**：标准 RoCE（Go-back-N + PFC + DCQCN）在万卡规模的问题有 PFC 死锁/风暴、ECMP 冲突导致带宽利用率低、incast 下 DCQCN 反应慢、RC 连接数膨胀。公开的方向有 SRD（AWS，可靠无序 + 多路径）、Ultra Ethernet（包喷洒 + 选择重传）、UB（华为，事务层与传输层分离，见 ai-infra-book 7.4.2）、HNS 的 HC3/LDCP（按 QP 可选的拥塞算法）。
2. **协议改动会打破通信库的哪些假设**（面试重点）：
   - 按序：通信库常用"轮询 buffer 尾字节"或"同 QP 上 WRITE 先于 IMM"判断数据到达。乱序传输下需要改成完成记录 / 计数器（NCCL 已经有 `cmplsRecords`、BY_ID matching scheme）。
   - 每连接状态：NCCL 每对 rank、每个 channel 都建 QP；共享传输通道后，QP 复用策略要重新设计。
   - 拥塞信号的上送：如果网卡能把拥塞 / 路径质量暴露给软件，通信库可以动态调权重（NCCL 已有 `NCCL_IB_EVENT_BASED_LB`（`common.cc`）和按设备速率算权重的 `ncclIbComputeLbWeights`（`common.h`））。
   - 故障恢复：选择重传和多路径让单链路故障不再等于 QP 进 ERR；NCCL 的 `p2p_resiliency*.cc`（`NCCL_IB_RESILIENCY_PORT_FAILOVER`）是软件层的对应物。
3. **集合通信可以反过来减轻拥塞**：算法层控制谁、在什么时候、向谁发多少，比如 incast 时按 rank 错开发送顺序、分层算法减少跨叶流量、rail 对齐、限制在途 chunk 数。拥塞控制和集合通信调度是一个联合设计问题。

## 源码导读（路径 + 函数 + 作用）

| 位置 | 作用 |
|---|---|
| `third_party/rdma-core/libibverbs/verbs.h`：`IBV_GID_TYPE_*`、`ibv_query_gid_ex` | GID 类型 |
| `third_party/nccl/src/transport/net_ib/connect.cc`：`ncclIbRoceGetVersionNum`、`ncclUpdateGidIndex`、`ncclIbGetGidIndex` | 从 sysfs 读 GID 类型，按 RoCE 版本 / 地址族 / 子网自动选 GID |
| 同上：`ncclIbQpRtr` | IB 走 LID（跨子网时用 FLID + GRH）；RoCE 用 GRH + traffic_class |
| `third_party/nccl/src/transport/net_ib/init.cc`：`ar` 初始化 | IB 默认开 AR，RoCE 默认关 |
| `third_party/rdma-core/providers/hns/hnsdv.h`：`enum hnsdv_qp_congest_ctrl_type` | 按 QP 选择拥塞算法的厂商接口 |
| `third_party/rdma-core/providers/mlx5/mlx5dv.h`：`MLX5DV_QP_CREATE_OOO_DP` | 乱序数据放置 |
| `third_party/rdma-core/providers/efa/efadv.h`：`EFADV_QP_DRIVER_TYPE_SRD` | SRD |
| `third_party/rdma-core/Documentation/rxe.md` | Soft-RoCE（rxe），没有 RNIC 时也能跑 RoCEv2 协议栈做实验 |
| `third_party/ai-infra-book/manuscripts/07-数据中心网络.md` 7.5.1–7.5.4 | 拥塞、incast、PFC/DCQCN 分工、多路径、选择重传、ECMP 冲突、死锁 |
| `third_party/ai-infra-book/manuscripts/06-超节点.md` 6.5.5 | 连接状态规模，RoCE RC 与 UB 对比 |

## 性能要点与常见坑

- RoCE 网络的 PFC/ECN 必须端到端一致：网卡 trust 模式（DSCP/PCP）、交换机无损队列、NCCL `NCCL_IB_TC` 三者要对齐，否则 RDMA 流量会落进有损队列，丢包后 Go-back-N 导致吞吐崩溃。
- `NCCL_IB_TIMEOUT` / `NCCL_IB_RETRY_CNT` 太小时，在拥塞下会被误判为链路故障（`IBV_WC_RETRY_EXC_ERR`）。
- ECMP 冲突用 perftest 单流测不出来，要多对多同时跑，看 p99 和最慢流。
- 只测平均带宽会掩盖周期性突发和 incast 的问题，要看队列深度、ECN 标记计数、PFC pause 计数（交换机侧和网卡计数器侧都要看）。

## 面试自测（问题 + 要点答案）

1. **IB 和 RoCEv2 的根本区别？为什么 RoCE 需要 PFC？**
   IB 链路层有 credit 流控，天然无损，靠 SM 集中管理；RoCEv2 跑在 UDP/IP 上，以太网本身会丢包，而经典 RC 网卡用 Go-back-N，丢包代价很高，所以要用 PFC 把以太网变成无损网络。
2. **PFC 和 DCQCN 的分工？只用其中一个行不行？**
   PFC 一跳反馈（亚微秒级），防止溢出；DCQCN 端到端反馈（一个 RTT，十几微秒），从源头降速。只用 PFC 会导致拥塞扩散 / 死锁；只用 DCQCN 在 incast 下来不及反应，会丢包（书中给出 1 MiB 缓冲时 N≤3 的条件）。
3. **讲一下 DCQCN 的 RP/NP/CP。**
   CP（交换机）按队列长度概率打 ECN；NP（接收网卡）收到 CE 后限频回 CNP；RP（发送网卡）乘性降速、α 做 EWMA，没有 CNP 时分阶段恢复。
4. **PFC 死锁是怎么形成的？怎么避免？**
   缓冲依赖成环（路由环、故障后重路由、上下行混用），环上队列互相等待。避免办法：拓扑 / 路由约束（up-down）、对请求和响应做优先级隔离、PFC watchdog，或者干脆走有损 + 选择重传。
5. **Go-back-N 和选择重传各自的代价？为什么多路径需要选择重传？**
   GBN 接收端无状态，但丢一个包要重发整个窗口；SR 要维护乱序缓存和位图。多路径 / 喷洒必然乱序，GBN 会把乱序当丢包。
6. **ECMP 哈希冲突对集合通信的影响？怎么缓解？**
   集合通信需要等最慢的流，少量冲突就会让整组带宽打折（书中 32 流 / 32 上联只剩 28.3%）。缓解：rail 对齐、多 QP 增加熵、AR、包喷洒、通信库按链路负载调权重。
7. **PFC 帧、CNP、RoCEv2 数据、TCP 的优先级怎么排？RoCE 为什么不设成严格优先？**
   PFC 帧是 MAC 控制帧，不进优先级队列，直接插队；CNP 放高优先级（如 prio 6）、严格调度，保证降速信号最快到；RoCE 数据放无损优先级（如 prio 3），开 PFC 和 ECN；TCP 放有损队列。RoCE 和 TCP 之间用 ETS 按权重分带宽，严格优先会在大流量时把 TCP 饿死。ECN 阈值要低于 PFC 的 XOFF，让 ECN 先降速，PFC 只兜底。
8. **如果你在设计一个新的 RDMA 传输协议，通信库侧需要配合改什么？**
   完成语义从"按序 + 尾字节轮询"改为完成记录 / 计数；QP / 连接复用模型；拥塞和路径信息上送后做动态负载均衡；故障时做重传或切换路径而不是让 QP 进 ERR；发送调度要感知 incast。

## 待深入

- DCQCN 在网卡上的默认参数（网卡 CNP 的默认 DSCP/优先级、`min_time_between_cnps` 的默认值和单位）待确认；确认方式：在 ConnectX 网卡上读 `/sys/class/net/<if>/ecn/roce_np/params/` 或 debugfs `cc_params/np_*`。
- 待确认：mlx5 RDMA 计数器所在的 sysfs 子目录名（笔记中写作 `hw_counters/`）；确认方式：在真机上 `ls /sys/class/infiniband/mlx5_0/ports/1/`。
- HPCC / Swift / Timely 等基于 INT 或时延的拥塞控制与 DCQCN 的对比（ai-infra-book 7.5.2 提到"基于时延的方法"，没有展开）。
- 待确认：NCCL ECE（`NCCL_IB_ECE_ENABLE`，`connect.cc` 中调 `wrap_ibv_query_ece` 和 `wrap_ibv_set_ece`）具体协商了哪些选项（是否包括拥塞控制算法）；确认方式：`third_party/rdma-core/libibverbs/man/ibv_query_ece.3.md`、`ibv_set_ece.3.md`，以及厂商对 `options` 位的定义。
