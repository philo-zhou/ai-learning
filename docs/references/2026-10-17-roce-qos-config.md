# RoCEv2 QoS 配置：优先级、PFC、ECN/CNP 的取值与命令

- 访问日期：2026-10-17
- 问题：RoCE 网络中 PFC 帧、CNP、RoCEv2 数据、TCP 的优先级如何划分；网卡侧用什么命令配置，常见取值是多少。
- 方法：逐条在一手资料（NVIDIA 文档、Linux 内核源码与 man page、DCQCN 论文、IEEE 802.1 文档）中核对。NVIDIA enterprise-support 的 "Recommended Network Configuration Examples for RoCE Deployment" 页面无法加载，未能核对。

## 结论

| 编号 | 结论 | 置信度 | 来源 |
|---|---|---|---|
| Q1 | 交换机侧（Cumulus Linux）RoCE 默认：DSCP 26 → switch priority 3 → traffic class 3（RoCE），在 priority 3 上开 PFC；DSCP 48 → priority 6 → traffic class 6（CNP） | high | [S2] |
| Q2 | 网卡侧 CNP 的 DSCP/优先级可配（上游驱动 debugfs `cc_params/np_cnp_dscp`、`np_cnp_prio`），但默认值是否为 48/6 未找到出处 | low（默认值未核实） | [S1] |
| Q3 | ToS 与 DSCP：NVIDIA 文档写明"DSCP 值乘以 4（左移 2 位）得到 ToS"，例如 DSCP 24 → ToS 96。所以 DSCP 26 → 104；常见的 106 = 104 + ECN 位 `10`（ECT），属本文推算 | medium（换算 high，106 的组成为推算） | [S3] |
| Q4 | 设备级强制 DSCP：`/sys/class/infiniband/<dev>/tc/<port>/traffic_class`；`cma_roce_tos -t <tos>` 写 configfs `rdma_cm/<dev>/ports/<port>/default_roce_tos`，只影响 RDMA_CM 建立的 QP | high | [S3][S4] |
| Q5 | `mlnx_qos` 选项：`--trust {pcp,dscp}`、`--dscp2prio set,<dscp>,<prio>`、`-f/--pfc`、`-p/--prio_tc`、`-s/--tsa`、`-t/--tcbw`、`--cable_len`、`--prio2buffer`、`--buffer_size`（实际单位是字节，help 文本写的 "percents" 有误） | high | [S5][S6] |
| Q6 | MLNX_OFED 的 ECN 开关：`/sys/class/net/<if>/ecn/{roce_rp,roce_np}/enable/<prio>`，参数在 `.../ecn/<protocol>/params/`；上游驱动改用 debugfs `cc_params/`（`np_min_time_between_cnps`、`np_cnp_dscp` 等） | high（路径）；`min_time_between_cnps` 默认值与单位未找到 | [S7][S1] |
| Q7 | PFC 风暴保护：`ethtool --set-tunable <if> pfc-prevention-tout <ms>`；0 = 关闭，65535 = 设备默认（auto）；mlx5 中 auto = 100 ms，可设范围 80~8000 ms | high | [S8][S9] |
| Q8 | DCQCN 论文在数据中心的参数（Fig.14）：Kmin 5 KB、Kmax 200 KB、Pmax 1%、g = 1/256、速率增加定时器 55 µs、字节计数器 10 MB；CNP 生成间隔固定为 50 µs（硬件限制） | high | [S10] |
| Q9 | PFC 帧是 MAC 控制帧：EtherType 0x8808、opcode 0x0101、目的地址 01-80-C2-00-00-01，"MAC control frames are never tagged"，暂停作用于对应优先级的数据帧队列。"PFC 帧本身不会被暂停"是由此推出，未找到原文直述 | medium | [S11][S12] |
| Q10 | 协商使用 PFC 后不应再用 802.3x PAUSE（IEEE PFC 提案草案原文）；NVIDIA 文档未见"必须关闭全局 PAUSE"的明确要求 | medium（出自 2008 年草案） | [S11] |
| Q11 | iproute2 `dcb` 语法：`dcb app add dev X dscp-prio 26:3`、`dcb pfc set dev X prio-pfc 3:on`、`dcb ets set dev X tc-tsa ... tc-bw ...` | high | [S13] |
| Q12 | 计数器：RDMA 侧 `out_of_buffer`、`out_of_sequence`、`rp_cnp_handled`、`rp_cnp_ignored`、`np_ecn_marked_roce_packets`、`np_cnp_sent`（mlx5 源码）；以太网侧 `rx_prio<p>_pause`、`tx_prio<p>_pause`、`tx_pause_storm_warning_events`、`tx_pause_storm_error_events`。sysfs 子目录名 `hw_counters/` 未核实 | high（名称）；目录名 low | [S14][S15][S3] |

## 来源

| 编号 | 来源 | 类型 |
|---|---|---|
| S1 | Linux master `drivers/infiniband/hw/mlx5/cong.c`：https://raw.githubusercontent.com/torvalds/linux/master/drivers/infiniband/hw/mlx5/cong.c | 一手（内核源码） |
| S2 | NVIDIA Cumulus Linux 5.18, RoCE：https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-518/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/ | 一手（厂商） |
| S3 | NVIDIA MLNX_OFED 24.01, RoCE：https://networking-docs.nvidia.com/mlnxofedswum/24010331/rdma-over-converged-ethernet-roce | 一手（厂商） |
| S4 | Mellanox/mlnx-tools `sbin/cma_roce_tos`：https://github.com/Mellanox/mlnx-tools/blob/master/sbin/cma_roce_tos | 一手（工具源码） |
| S5 | Mellanox/mlnx-tools `python/mlnx_qos`：https://github.com/Mellanox/mlnx-tools/blob/master/python/mlnx_qos | 一手（工具源码） |
| S6 | NVIDIA MLNX_OFED 24.01, QoS：https://networking-docs.nvidia.com/mlnxofedswum/24010331/quality-of-service-qos | 一手（厂商） |
| S7 | NVIDIA MLNX_OFED 24.01, ECN：https://networking-docs.nvidia.com/mlnxofedswum/24010331/explicit-congestion-notification-ecn | 一手（厂商） |
| S8 | ethtool(8)：https://man7.org/linux/man-pages/man8/ethtool.8.html | 一手 |
| S9 | Linux master `drivers/net/ethernet/mellanox/mlx5/core/en_ethtool.c`、`include/uapi/linux/ethtool.h` | 一手（内核源码） |
| S10 | Zhu et al., Congestion Control for Large-Scale RDMA Deployments, SIGCOMM 2015：https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf | 一手（论文） |
| S11 | Pelissier, PFC proposal, IEEE 802.1, 2008-05：https://www.ieee802.org/1/files/public/docs2008/bb-pelissier-pfc-proposal-0508.pdf | 一手（标准草案） |
| S12 | IEEE 802.1Qbb 项目页：https://1.ieee802.org/dcb/802-1qbb/ | 一手（仅范围说明） |
| S13 | dcb-app(8)、dcb-pfc(8)、dcb-ets(8)：https://man7.org/linux/man-pages/man8/dcb-app.8.html 等 | 一手 |
| S14 | Linux master `drivers/infiniband/hw/mlx5/counters.c` | 一手（内核源码） |
| S15 | Kernel mlx5 ethtool counters：https://docs.kernel.org/networking/device_drivers/ethernet/mellanox/mlx5/counters.html | 一手 |

## 局限

- 内核源码引用的是 2026-10-17 的 master，未固定 commit。
- ETS 权重（RoCE 与 TCP 如何分）、交换机 ECN 阈值在 100G/400G 上的推荐值、交换机 PFC watchdog 的检测时间都没有核实到一手出处，属于部署经验，笔记中按"示例"处理。

## 未决问题

- 网卡 CNP 默认 DSCP/优先级、`min_time_between_cnps` 的默认值与单位：在 ConnectX 网卡上读 debugfs `cc_params/np_*` 或 OFED sysfs。
- `hw_counters/` 目录名：在真机上列出 `/sys/class/infiniband/mlx5_0/ports/1/`。
