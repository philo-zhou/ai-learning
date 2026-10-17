# perftest：RDMA 微基准的用法、解读与源码结构

> 层次：下层 · 来源：third_party/perftest@05200c2（`README`、`Makefile.am`、`src/`）

## 一句话

perftest 是"不经过通信库、直接用 verbs 打满网卡"的标尺：先用 `ib_write_bw` / `ib_write_lat` 确认 RNIC 和网络本身能跑到多少，再拿 NCCL/nccl-tests 的结果去比，差距才能归到通信库或者拓扑头上。

## 在调用链中的位置

- 与 NCCL 平级，直接调 libibverbs（可选 librdmacm），不经过任何通信库。
- 在排障链里的位置：`ibv_devinfo`（端口是否 ACTIVE、MTU、link_layer）→ perftest 点对点（单链路能力）→ nccl-tests（集合通信）→ 业务。NCCL 自身也带一个 ib_write_bw 风格的诊断：`third_party/nccl/src/diagnostics/ib_write_bw.cc`（`ncclDiagRunIbWriteBw`，头文件注释说明它"纯观察、不会让 communicator 初始化失败"）。

下图左边是软件栈上的位置（perftest 与 NCCL 平级，都直接调 verbs），右边是排障时逐层往上加变量的顺序：

```text
  software stack                         troubleshooting order
  ──────────────                         ─────────────────────
  ┌──────────────┐ ┌──────────────┐      ┌──────────────┐
  │ perftest     │ │ NCCL net_ib  │      │ ibv_devinfo  │  ← 端口 ACTIVE、MTU、link_layer
  └──────┬───────┘ └──────┬───────┘      └──────┬───────┘
         └───────┬────────┘                     ▼
          ┌──────┴───────┐               ┌──────────────┐
          │ libibverbs   │               │ perftest     │  ← 单对、单流：RNIC + 链路本身
          │ (+librdmacm) │               └──────┬───────┘
          └──────┬───────┘                      ▼
          ┌──────┴───────┐               ┌──────────────┐
          │ RNIC         │               │ nccl-tests   │  ← 加上算法、拓扑、多流并发
          └──────────────┘               └──────┬───────┘
                                                ▼
                                         ┌──────────────┐
                                         │ workload     │
                                         └──────────────┘
```

## 核心概念

### 工具清单（`Makefile.am` 中的 `bin_PROGRAMS`）

| 二进制 | 源文件 | 测什么 |
|---|---|---|
| `ib_write_bw` / `ib_write_lat` | `src/write_bw.c` / `src/write_lat.c` | RDMA WRITE 带宽 / 延迟 |
| `ib_read_bw` / `ib_read_lat` | `src/read_bw.c` / `src/read_lat.c` | RDMA READ |
| `ib_send_bw` / `ib_send_lat` | `src/send_bw.c` / `src/send_lat.c`（加 `multicast_resources.c`） | SEND/RECV（支持 UD 多播） |
| `ib_atomic_bw` / `ib_atomic_lat` | `src/atomic_bw.c` / `src/atomic_lat.c` | CAS / FAA |
| `raw_ethernet_*` | `src/raw_ethernet_*.c` | Raw Packet QP |

用法：server 端执行 `./ib_write_bw <opts>`，client 端执行 `./ib_write_bw <opts> <server_ip>`，**两端选项必须一致**（README 中的 IMPORTANT NOTE）。带外通道默认是 TCP 18515 端口。

### 关键参数（README "Common Options" 与 `perftest_parameters.c`）

| 参数 | 作用 | 面试 / 排障中的意义 |
|---|---|---|
| `-d mlx5_0`、`-i 1` | 设备、端口 | 多网卡机器一定要显式指定，并和 GPU 的 NUMA / PCIe 亲和对齐 |
| `-x <gid_idx>` | GID index | RoCE 必填，选 RoCEv2 + IPv4 那一条 |
| `-c RC/UC/UD/XRC/DC/SRD` | 传输类型 | 默认 RC |
| `-s <size>` / `-a` | 消息大小 / 扫 2B 到 2^23 B | 用 `-a` 看带宽曲线的拐点 |
| `-n <iters>` / `-D <sec>` | 迭代次数 / 时长 | 长时间跑要用 `-D`，避免 `-n` 很大时记录数组过大（README 说明内存随 iters 增长） |
| `-q <n>` | QP 数 | 单 QP 往往打不满 400G 网卡，需要多 QP |
| `-t <tx_depth>` | 发送队列深度 / 在途 WR 数，默认 128 | 带宽 = 在途字节 / RTT，深度不够打不满 |
| `-l <post_list>` | 一次 post 一串 WR | 测硬件消息率上限（README：单次 post 软件上限约 10 Mpps） |
| `-Q <cq_mod>` | 每 cq_mod 个 WR signal 一次 | selective signaling，见 [02-transport-and-ops.md](02-transport-and-ops.md) |
| `-I <size>` | inline 上限 | 小消息延迟 |
| `-m <mtu>` | path MTU | IB 4096；RoCE 受以太网 MTU 限制 |
| `-b` | 双向 | 双向时各端测自己发出的流量，最后汇总（README "Testing Methodology"） |
| `-o <outs>` | READ/ATOMIC 在途数 | 受 `max_rd_atomic` 约束 |
| `-R` / `-z` | 用 rdma_cm 建链 / 用 rdma_cm 交换数据 | 没有可用 TCP 时 |
| `-F` | CPU 降频（ondemand）时不报错 | 结果用 CPU cycle 计时，降频会让结果失真 |
| `--report_gbits` | 用 Gb/s 输出 | 默认 MiB/s |
| `--use_cuda=<gpu>`、`--use_cuda_dmabuf` | buffer 放 GPU 显存，测 GPUDirect RDMA | 见 [05-gpudirect-and-nccl-ib.md](05-gpudirect-and-nccl-ib.md) |
| `--tclass`、`-S <sl>` | 流量类别 / 服务等级 | RoCE 无损队列对齐 |
| `--write_with_imm` | ib_write_* 改用 WRITE_WITH_IMM | 对比通信库的真实用法 |
| `-W/--report-counters=<list>` | 打印网卡计数器变化，例如 `counters/port_xmit_data,hw_counters/out_of_buffer` | 排查丢包 / 缓冲不足 |
| `--congest_type=<DCQCN,LDCP,HC3,DIP>` | 仅 HNS（`HAVE_HNSDV`） | 按 QP 选择拥塞算法 |
| `--rate_limit`、`--burst_size` | 发送限速 | 模拟背景流量、观察拥塞 |
| `--out_json`、`--cpu_util`、`--use_hugepages`、`--odp`、`--mr_per_qp`、`--use_old_post_send`、`--use_ooo` | 输出格式与各种特性开关 | |

典型命令（RoCE，双端 GPU）：

```
server: ib_write_bw -d mlx5_0 -x 3 -q 4 -s 1048576 -D 10 --report_gbits --use_cuda=0
client: ib_write_bw -d mlx5_0 -x 3 -q 4 -s 1048576 -D 10 --report_gbits --use_cuda=0 <server_ip>
```

### 结果怎么读

- **带宽测试**，列头是 `#bytes #iterations BW peak BW average MsgRate[Mpps]`（`perftest_parameters.h`：`RESULT_FMT` / `RESULT_FMT_G`）。
  - `BW average` = 总字节 × CPU 频率 / 总 cycles / format_factor。`--report_gbits` 时 format_factor 是 125000000（十进制 Gb/s），默认是 0x100000（MiB/s）（`print_report_bw`）。
  - `BW peak` 取单次 cq_mod 区间内的最快样本，`-N` 关闭；看 average 即可，peak 意义不大。
  - 单向测试由 client 端统计（README）。
  - 合理预期：大消息的 average 接近线速扣除协议头后的值，比如 400 Gb/s 网卡，在 4 KB MTU 下大约能到 9x% 的线速。待确认：具体百分比；确认方式：在目标网卡上跑 `-a`，用最大消息的 average 除以 `ibv_devinfo -v` 中的 active_speed。
  - 小消息看 `MsgRate`：它受每 WR 软件开销、doorbell、PCIe 往返限制；用 `-l` / `-q` 可以看到硬件上限。
- **延迟测试**，列头是 `t_min t_max t_typical t_avg t_stdev 99% 99.9%`（`RESULT_FMT_LAT`）。
  - `t_typical` 是中位数（`print_report_lat` 里的 `get_median`），比 avg 更稳；README 指出首个样本通常最大（warm-up）。
  - SEND/WRITE 是 ping-pong，报告的是 RTT/2（`rtt_factor = 2`）；READ/ATOMIC 是本端发起、读回，报告完整 RTT（`rtt_factor = 1`）。所以 `ib_read_lat` 天然比 `ib_write_lat` 大一倍左右，这不说明 READ 更慢。
  - `ib_write_lat` 的实现是两端互相写对方 buffer 的最后一个字节，然后忙等轮询本地 buffer 尾字节（`run_iter_lat_write`：`post_buf = buf + size - 1`，`while (*poll_buf != (char)rcnt)`）。它依赖"WRITE 按地址顺序落地、最后一个字节最后到"。README 提到 MUSA 显存上的 `ib_write_lat` 必须用 `--write_with_imm`，原因正是 CPU 无法轮询设备内存。

把两种延迟测试的一次迭代并排画出，就能看出 `rtt_factor` 为什么不同：

```text
ib_write_lat：ping-pong，一次迭代 = 一个来回，报告 RTT / 2（rtt_factor = 2）
（run_iter_lat_write 中 server 第一轮不先轮询，直接先写）

  server                                   client
    │── RDMA WRITE (last byte = n) ────────►│  busy-poll: *poll_buf == n
    │◄──────── RDMA WRITE (last byte = n) ──│
    │  busy-poll: *poll_buf == n            │

ib_read_lat：一次迭代 = 一个 READ 往返，报告完整 RTT（rtt_factor = 1）

  requester                                responder
    │── READ request ──────────────────────►│  RNIC 直接读 MR，对端 CPU 不参与
    │◄─────────────────────── READ response │
    │  poll CQ                              │
```

## 源码导读（路径 + 函数 + 作用）

以 `ib_write_bw` 为主线（`third_party/perftest/src/write_bw.c` 的 `main`）：

1. `parser`（`perftest_parameters.c`）：解析参数，填 `struct perftest_parameters`。
2. `ctx_find_dev`、`ctx_open_device`、`check_link`：找设备、打开、检查链路层 / MTU / GID。
3. `create_comm_struct` + `establish_connection`（`perftest_communication.c`）：建立带外通道（TCP 或 rdma_cm）。
4. `alloc_ctx` + `ctx_init`（`perftest_resources.c`）：建 PD/MR/CQ/QP。内存来源通过 `struct memory_ctx` 函数表抽象（`src/memory.h`：`init / allocate_buffer / free_buffer / copy_*`），实现分别在 `host_memory.c`、`cuda_memory.c`、`rocm_memory.c`、`neuron_memory.c`、`hl_memory.c`、`mlu_memory.c`、`musa_memory.c`、`opencl_memory.c`、`dm_memory.c`、`mmap_memory.c`、`system_dmabuf_memory.c`。`create_mr` / `create_single_mr` 在有 dmabuf fd 时调 `ibv_reg_dmabuf_mr`，否则调 `ibv_reg_mr`。`create_qp_main` / `create_reg_qp_main`、`modify_qp_to_init` 负责 QP。
5. `set_up_connection`：填本端 `struct pingpong_dest`（`perftest_resources.h`：`lid, out_reads, qpn, psn, rkey, vaddr, gid, srqn, gid_index` 等）。
6. `ctx_hand_shake`：交换 dest，线上格式是 `KEY_PRINT_FMT "%04x:%04x:%06x:%06x:%08x:%016llx:%08x"`（`perftest_communication.h`，GID 版本是 `KEY_PRINT_FMT_GID`），比 rc_pingpong 多了 rkey 和 vaddr，因为 WRITE 需要知道远端地址。`ctx_check_gid_compatibility` 检查两端 GID 是否同为 IPv4-mapped 或同为 IPv6（用 `ipv6_addr_v4mapped` 判断；该实现位于 `#ifdef HAVE_MLX5DV` 分支内）。参数协商在 `perftest_negotiation.c`（`negotiate_params`；例如 `compare_max_qp_rd_atom`：本端 `out_reads` 大于对端 `max_qp_rd_atom` 时，降到对端的值）。
7. `ctx_connect`：对每个 QP 调 `ctx_modify_qp_to_rtr` / `ctx_modify_qp_to_rts`。
8. `ctx_set_send_wqes`：预先构造好 WR 数组。按 `post_list` 串链表；`(j + 1) % cq_mod == 0` 时才置 `IBV_SEND_SIGNALED`；小于 inline 上限时置 `IBV_SEND_INLINE`；按 verb 填 `opcode_verbs_array[...]` 和 `wr.rdma.rkey / remote_addr`。
9. `run_iter_bw`：主循环。发送窗口条件是 `scnt + post_list <= tx_depth + ccnt`（在途数不超过 tx_depth），用 `get_cycles()` 记录 `tposted`，调 `post_send_method`，然后 `ibv_poll_cq` 回收；`run_iter_bw_server` 是被动端；`run_iter_bi` 是双向；`run_iter_bw_infinitely` 对应 `--run_infinitely`。
10. `print_report_bw`（`perftest_parameters.c`）：按上面的公式输出。

第 9 步的发送窗口画出来是这样（单个 QP；`ccnt` 在每拿到一个 CQE 时加 `cq_mod`，因为一个 signaled WR 代表 `cq_mod` 个 WR 完成）：

```text
  WR index   0 ─────────────────────────────────────────────────────────────►
             ├────── completed ───────┼──── in flight (posted) ────┼── next ──►
             0                       ccnt                         scnt
                                      │◄─────── scnt - ccnt ──────►│

  post 条件：scnt + post_list <= tx_depth + ccnt      → 在途 WR 不超过 tx_depth
  post 后：  scnt += post_list                         （一次 doorbell 一串 WR）
  poll 后：  每个 CQE 让 ccnt += cq_mod                 （selective signaling）
  所以：     在途字节上限 ≈ tx_depth × size；带宽 ≈ 在途字节 / RTT，深度不够就打不满
```

延迟主线：`write_lat.c` → `run_iter_lat_write`（尾字节轮询）/ `run_iter_lat_write_imm`；`send_lat.c` → `run_iter_lat_send`；`read_lat.c` → `run_iter_lat`；`print_report_lat` 计算 median、百分位数和 `rtt_factor`。

其他：`get_clock.c`（CPU cycle 计时，`get_cpu_mhz`）、`perftest_counters.c`（`-W` 网卡计数器）、`multicast_resources.c`（UD 多播）、`raw_ethernet_resources.c`。

## 性能要点与常见坑

- **两端参数不一致**（尤其是 `-s`、`-q`、`-c`、`--use_cuda`）会导致协商失败或结果异常。
- **NUMA / PCIe 亲和**：用 `numactl --cpunodebind` 把进程绑到网卡所在的 NUMA 节点；测 GDR 时 GPU 和网卡要在同一个 PCIe switch 下，否则 P2P 走 CPU root complex，带宽会明显下降。
- **CPU 频率**：cycle 计时假设频率恒定，要关掉 turbo/ondemand，或者至少知道结果有偏差（`-F` 只是关掉检查）。
- **单 QP 打不满**：大带宽网卡要 `-q 2..8`；这也解释了 NCCL 为什么需要 `NCCL_IB_QPS_PER_CONNECTION`、多 channel。
- **RoCE 上带宽偏低或抖动**：先查 PFC/ECN 配置和 `--tclass`，再用 `-W` 看 `out_of_buffer`、丢包、CNP 等计数器。
- **点对点结果好 ≠ 集合通信好**：perftest 是单对、单流，测不出 ECMP 冲突和 incast（见 [03-roce-ib-congestion.md](03-roce-ib-congestion.md)），需要多对并发或 nccl-tests 才能看到。

## 面试自测（问题 + 要点答案）

1. **`ib_write_bw` 跑不满线速，你按什么顺序排查？**
   先看 `ibv_devinfo` 中端口速率 / 宽度和 MTU → 调消息大小（`-a` 看拐点）→ 加 `-q` 和 `-t` 增加在途量 → 检查 NUMA 绑核和 PCIe 拓扑（`lspci -tv`，PCIe Gen/宽度）→ RoCE 检查 tclass/PFC/ECN、计数器 → GDR 场景检查 GPU 与 NIC 的 PCIe 距离以及 ACS/IOMMU 设置。
2. **`ib_read_lat` 为什么约等于 `ib_write_lat` 的 2 倍？**
   write/send 延迟是 ping-pong RTT/2；read 是一次完整的请求-响应 RTT，perftest 对 READ/ATOMIC 的 `rtt_factor=1`，不再除以 2。
3. **`-Q`（cq_mod）和 `-l`（post_list）分别优化了什么？**
   `-Q` 减少 CQE 写回和 poll 次数（selective signaling）；`-l` 减少 doorbell 次数和每次 post 的软件开销，从而测出硬件消息率。
4. **`ib_write_lat` 是怎么知道数据到了的？这个方法在什么情况下会失效？**
   轮询 buffer 最后一个字节变为期望值。失效情况：网卡乱序放置数据（OOO、多路径喷洒），buffer 在 GPU 显存而 CPU 无法轮询，或者 PCIe relaxed ordering 导致尾字节先于前面的数据可见。
5. **perftest 和 nccl-tests 结果差很多，可能有哪些原因？**
   NCCL 选错网卡 / 没开 GDR（看 `NCCL_DEBUG=INFO` 日志中的 `GDRDMA` 字样）、channel/QP 数不够、拓扑识别错误、集合通信算法的带宽系数（busbw 与 algbw 的换算）、多流并发时的拥塞和 ECMP 冲突。

## 待深入

- `--use_ooo`、`--use_data_direct` 对应的 mlx5 特性，以及它们和 NCCL `NCCL_IB_OOO_RQ` / `NCCL_IB_DATA_DIRECT` 的对应关系。
- 在 labs/ 中做一组实验：在 Soft-RoCE（`third_party/rdma-core/Documentation/rxe.md`）上跑 `ib_write_bw -a`，记录曲线。
