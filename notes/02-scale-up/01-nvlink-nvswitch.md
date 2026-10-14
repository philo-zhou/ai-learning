# NVLink、NVSwitch、NVL72 与 scale-up / scale-out

> 层次：下层 scale-up · 来源：ai-infra-book@3bdcb4fc 第 1 章、第 6 章 6.5.1~6.5.3 与 6.5.5 节、第 7 章 7.1 与 7.3.3 节及 `references/files/specs/nvidia-nvlink-spec.md`；third_party/nccl@3e46c08（`src/graph/topo.h`、`src/mnnvl.cc`）

## 一句话

NVLink 是 GPU 之间（以及 Grace-GPU 之间）的高带宽点对点链路，NVSwitch 把多条 NVLink 汇成一个交换网络，使域内任意两块 GPU 等距、可以直接 load/store 对端显存；NVL72 把这个域从 8 卡服务器扩大到 72 卡机柜。与 PCIe 和网卡相比，它带宽高一个数量级、路径短、是内存语义，但规模受交换芯片端口数、线缆距离、功率和一致性开销限制。

## 在调用链中的位置

- 上层：NCCL 拓扑把 GPU-GPU 的 NVLink 路径判成 `PATH_NVL`、经中间 GPU 的判成 `PATH_NVB`（[../04-nccl-internals/02-topology-graph.md](../04-nccl-internals/02-topology-graph.md)），据此选择 P2P transport、LL128 协议和 NVLS 算法。
- 本篇：硬件与系统层面的概念。
- 软件映射：[02-nccl-p2p-shm-nvls.md](02-nccl-p2p-shm-nvls.md)。

## 核心概念

### scale-up 与 scale-out

- ai-infra-book 第 1 章：超节点内部互联称 **scale-up**（纵向扩展），重点是让一组加速器以较低通信开销共同计算；多个超节点通过网卡与交换网络相连称 **scale-out**（横向扩展），用于扩大集群范围。以 GB200 NVL72 为例，36 个 Grace CPU 和 72 个 Blackwell GPU 在一个机柜中通过 NVLink 形成同一个互联域，集群网络负责对外连接。
- 第 7 章 7.1.3 的说法：紧耦合互联把设备接到处理器和内存系统上，适合频繁、细粒度协作；集群网络连接大量独立节点，还要处理节点和链路故障。同一份数据在机内和跨机传输往往要用不同接口（load/store vs 异步 read/write 请求）。
- 第 7 章 7.3.3 区分 Load/Store（处理器指令，硬件跟踪依赖）与异步 Read/Write（提交请求到队列，运行时等完成事件），并提醒不能凭接口名称推断缓存一致性。

| 维度 | scale-up（NVLink/NVSwitch） | scale-out（RDMA 网卡 + 交换网络） |
|---|---|---|
| 访问语义 | 内存映射，SM 直接 load/store 对端显存；也可用 copy engine | 消息/RDMA，需注册内存、QP、提交 WQE、轮询 CQ |
| 典型带宽（书中 H100 例） | 每卡 NVLink 总 900 GB/s 双向，即每方向 450 GB/s | 每卡 400 Gbit/s 网卡，每方向 50 GB/s |
| 发起者 | GPU kernel 自己 | 通常 CPU proxy 线程（或 GPU 发起的 GIN/IBGDA） |
| 规模限制 | 交换芯片端口、线缆距离、功率散热、一致性目录 | 交换机层数、拥塞控制、故障 |
| NCCL transport | P2P、NVLS、（同机无 P2P 时）SHM | NET（IB/Socket/插件）、CollNet |

### NVLink 与 NVSwitch（书中数据）

- H100：每张卡 NVLink 总带宽 900 GB/s（收发合计），每方向 450 GB/s（第 7 章开头的算例）。
- 第三代 NVSwitch（HGX H100）：一颗芯片 64 个 NVLink 4 端口，每端口每方向 25 GB/s；书中用它说明交换机的"基数"与上联/下联端口分配取舍：32 下联/32 上联时上联带宽 800 GB/s，48/16 时只有 400 GB/s（6.5.1）。
- DGX H100 的 NVLink Switch System 把多台服务器接入同一 NVLink 交换网络，每节点只把一半 NVLink 带宽引出节点（2:1 收敛），最多连接 256 张 H100（6.5.1、6.5.3）。
- 第五代 NVLink：每方向 900 GB/s（每 GPU 1800 GB/s 双向合计），见第 7 章 7.6.4 及其脚注。
- 规格表（ai-infra-book 快照 `references/files/specs/nvidia-nvlink-spec.md` 的 "Specifications" 节，标注 Preliminary）：

| 代际 | 每 GPU NVLink 带宽 | 每 GPU 最大链路数 | 架构 | Switch GPU 域 | Switch 总聚合带宽 |
|---|---|---|---|---|---|
| 第四代 | 900 GB/s | 18 | Hopper | 8 | 7.2 TB/s |
| 第五代 | 1,800 GB/s | 18 | Blackwell | 8 / 72 | 130 TB/s（NVL72） |
| 第六代 | 3,600 GB/s | 36 | Rubin | 8 / 72 | 260 TB/s（NVL72） |

  同一快照说明每颗 NVLink Switch 内含 SHARP 引擎，用于网络内归约和 multicast 加速。由表可推每条链路约 50 GB/s 双向（900/18），即每方向约 25 GB/s，与书中"NVSwitch 端口每方向 25 GB/s"一致（推算，非原文）。
- 演进（6.5.3）：DGX-1 用 NVLink 混合立方体网格（部分 GPU 直连，其余需中转）→ DGX-2 用 16 张 V100 + NVSwitch（把"是否直连"变成"交换层能同时转发多少"）→ DGX A100 八卡 + NVSwitch → GB200 NVL72 机柜级 NVLink。
- 6.5.5 提到 NVLink 属于缓存一致互联，目录开销随对端数线性增长，把这类互联的规模限制在几十个对端，NVL72 的 72 张 GPU 正处在这个范围（书中观点，与 UB 不维护跨主机一致性对比）。

### 与 PCIe 对比

- 书中 DGX H100 例：每张 GPU 经 PCIe Gen5 x16 连到主机，每方向 64 GB/s；两颗 CPU 之间最多 4 条 UPI（6.5.1 的 NUMA 例子）。GPU 间若只能走 PCIe，跨 NUMA 时还要过 UPI，缓冲位置和环的次序会改变 UPI 上的流量。
- 第 7 章对照配置：4 张 A100 PCIe，卡间 PCIe Gen4 P2P 每方向 32 GB/s，网卡与插槽共享 32 GB/s；在这种机器上分层归约反而比连续环慢（7.2.1 讨论），说明"本地互联要比网卡快足够多，分层才合算"。
- NCCL 内部带宽模型（`src/graph/topo.h`，搜索用的估计值，非规格）：`SM90_NVLINK_BW 20.6`、`SM100_NVLINK_BW 40.1`（每条 NVLink）、`PCI_BW 12.0`（注释 PCI Gen3 x16）、`NET_BW 12.0`（注释 100Gbit）；`INTEL_P2P_OVERHEAD(bw)` 注释说明 Intel CPU 把 GPU P2P 流量拆成 64B TLP，消耗更多 PCIe 带宽。
- 路径类型：NVLink 直连是 `PATH_NVL`，PCIe 依距离是 `PATH_PIX/PXB/PHB/SYS`；NCCL 默认 P2P 只到 `PATH_PXB`（`ncclTopoCheckP2p`）。

### 多节点 NVLink（MNNVL）

- `src/mnnvl.cc::ncclMnnvlCheck`：要求 cuMem 开启、所有 rank 支持 fabric handle（`fabricHandleSupport`）且 NVML 报告 fabric 状态 `NVML_GPU_FABRIC_STATE_COMPLETED`；按 `fabricInfo.cliqueId` 划分 clique（同一 NVLink 域内的 GPU 组）。
- 可用时把 `ncclCuMemHandleType` 设为 `CU_MEM_HANDLE_TYPE_FABRIC`（`mnnvl.cc:89`），跨节点的 GPU 也能像同机 P2P 一样导入对端显存。相关参数：`NCCL_MNNVL_ENABLE`（默认 2，自动）、`NCCL_MNNVL_UUID`、`NCCL_MNNVL_CLIQUE_ID`、`NCCL_MNNVL_CROSS_CLIQUE`（`src/init.cc:871-873、1055`）。
- 日志中 P2P 连接会显示 `via P2P/MNNVL`（`p2p.cc::p2pSendSetup`）。

### 交换机内归约（SHARP 思想）

- 第 7 章 7.2.1 "在网归约"：交换芯片沿聚合树归约多个端口的数据再分发回去，端点不必多次发送同一份数据；收益主要在服务器数较多时减少轮数和字节数。
- NVSwitch 上的对应物是 NVLink SHARP（NCCL 称 NVLS）：GPU 对一个 multicast 地址做 `multimem.ld_reduce`，由交换机完成归约（[02-nccl-p2p-shm-nvls.md](02-nccl-p2p-shm-nvls.md)）。

## 性能要点与常见坑

- 更大的 NVLink 域不一定让单个 TP 组更快：书中 6.5.3 算例，Qwen3-32B 的 TP16 即使放进同一 NVLink 交换域仍慢于单机 TP8（环轮数翻倍而每卡 KV 读取不再减少）；大互联域的价值更多在于容纳更多实例或更大模型。
- NVLink 交换网络的上联收敛比（如 2:1）决定跨节点 NVLink 的割集带宽，做通信库设计时要按割集算下界，而不是按单卡 NVLink 带宽。
- PCIe 机器上 P2P 是否可用受 ACS、IOMMU、BIOS 与 CPU 型号影响；NCCL 会在 NVLink 相连但 NVML 报告 P2P 不可用时直接报错（`ncclTopoCheckP2p`），这通常是硬件或驱动问题。
- `nvidia-smi topo -m`、`nvidia-smi nvlink --status`（待确认：本地无 NVIDIA 驱动环境，命令输出未实测；确认方式：在 GPU 机器上执行）是核对 NVLink 拓扑与链路状态的第一步。

## 面试自测

1. 问：NVSwitch 相对 DGX-1 式直连拓扑解决了什么问题？代价是什么？
   答：把"某对 GPU 是否直连、需不需要中转"变成"交换层能同时转发多少数据"，任意两卡等距，all-to-all 不再因为多跳而放大链路负载；代价是交换芯片、线缆、功耗和成本，以及域规模受交换芯片端口数限制，超出后要加一级交换并面临上联收敛。
2. 问：为什么说 NVLink 是内存语义而 RDMA 是消息语义？对通信库实现有何影响？
   答：NVLink 域内对端显存可以映射到本 GPU 地址空间，SM 直接 load/store/原子操作，同步靠内存上的 flag；RDMA 需要注册内存、建 QP、提交请求和轮询完成。通信库在 scale-up 侧可以把数据搬运和归约融合进 kernel（LL/LL128 依赖写入顺序/原子性），scale-out 侧需要 proxy 或 GPU 发起网络把请求交给网卡。
3. 问：H100 8 卡机内与跨机带宽差多少？对 TP/DP 的放置意味着什么？
   答：书中数据：每卡 NVLink 每方向 450 GB/s，网卡每方向 50 GB/s，差约 9 倍。所以频繁的 TP 归约放在 NVLink 域内，DP 梯度同步、PP 阶段交接放到跨节点；分层归约把跨节点阶段压缩成少数轮。
4. 问：MNNVL 与普通单机 P2P 在 NCCL 中的差别？
   答：MNNVL 下 GPU 跨主机但处在同一 NVLink 域（clique），需要 cuMem + fabric handle 导出/导入显存（`CU_MEM_HANDLE_TYPE_FABRIC`），NCCL 要在 init 时检查 NVML fabric 状态与 clique id；单机 P2P 可用 POSIX fd 或 legacy CUDA IPC。
5. 问：交换机内归约（SHARP/NVLS）什么时候收益明显？
   答：书中分析：两台服务器时与分层环差别很小（只省一次启动）；服务器/参与者越多，环需要的字节数是在网归约的 2(S-1)/S 倍，还多出更多轮启动，收益越明显；小消息主要节省轮数带来的延迟。还要考虑交换机归约引擎的吞吐和支持的数据类型。

## 待深入

- 待确认：NVL72 机柜内 NVSwitch 托盘数量与每托盘芯片数；确认方式：阅读 ai-infra-book `references/files/specs/nvidia-gb200.md`。
- 待确认：`nvidia-smi topo -m` 中 `NODE` 与 NCCL 路径类型的对应；确认方式：在真实机器上对比 `nvidia-smi topo -m` 与 `NCCL_TOPO_DUMP_FILE` 导出的 XML。
- ai-infra-book 6.5.4（TPU 环面）与 6.5.5（UB/CloudMatrix）中的拓扑权衡，用于对比 NVSwitch 路线。
