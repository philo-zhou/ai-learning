# NCCL 拓扑探测与通道搜索（graph/）

> 层次：中层 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）、ai-infra-book@3bdcb4fc 第 6 章 6.5.1 节

## 一句话

NCCL 先把本机 GPU、PCIe 交换芯片、CPU(NUMA)、NVSwitch、网卡组织成一张带带宽的图（可导出/导入为 XML），再对每对节点算出"最差的一段是什么类型"的路径（LOC/NVL/NVB/C2C/PIX/PXB/P2C/PXN/PHB/SYS/NET），最后在这张图上用回溯搜索找出若干条带宽尽量高的 ring/tree/NVLS 通道。

## 在调用链中的位置

- 上层：`src/init.cc::initTransportsRank` 依次调用 `ncclTopoGetSystem` → `ncclTopoComputePaths` → `ncclTopoTrimSystem` → `ncclTopoComputePaths` → `ncclTopoSearchInit` → `ncclTopoCompute(各 pattern)` → `ncclTopoPreset/Postset`（见 [01-init-bootstrap.md](01-init-bootstrap.md)）。
- 本模块：`src/graph/` 下 `xml.cc`、`topo.cc`、`paths.cc`、`search.cc`、`connect.cc`、`rings.cc`、`trees.cc`。
- 下游消费者：transport 选择（`ncclTopoCheckP2p`、`ncclTopoCheckNet`、`ncclTopoCheckGdr`，见 [../02-scale-up/02-nccl-p2p-shm-nvls.md](../02-scale-up/02-nccl-p2p-shm-nvls.md)）、tuning 代价模型（`bwIntra/bwInter/typeIntra/typeInter`）。Ring/Tree 算法本身见 [../03-ccl-algorithms/README.md](../03-ccl-algorithms/README.md)。

## 核心概念

### 节点类型（`src/graph/topo.h`）

`GPU`、`PCI`、`NVS`（NVSwitch）、`CPU`（注释：实际是 NUMA 域）、`NIC`、`NET`、`GIN`、`RMA`、`DEV`、`CXB`。字符串表 `topoNodeTypeStr` 在 `src/graph/topo.cc:33`。

### 链路类型与路径类型

`topo.h` 里刻意让 link 与 path 编号对齐（注释 "We want link types and path types to match as much as possible"）：

| 值 | 路径类型 | 含义（`src/include/graph.h` 注释） |
|---|---|---|
| 0 | `PATH_LOC` | 自己 |
| 1 | `PATH_NVL` | 经 NVLink |
| 2 | `PATH_NVB` | 经 NVLink，借道中间 GPU |
| 3 | `PATH_C2C` | 经 C2C（如 Grace-Hopper 的 CPU-GPU 互联） |
| 4 | `PATH_PIX` | 至多经过一个 PCIe 桥 |
| 5 | `PATH_PXB` | 经过多个 PCIe 桥，但不经过 PCIe Host Bridge |
| 6 | `PATH_P2C` | GPU 经 C2C 到 CPU 再经 PCIe 到网卡 |
| 7 | `PATH_PXN` | GPU 经中间 GPU 到网卡（rail-local 聚合发送） |
| 8 | `PATH_PHB` | 经 PCIe 与 Host Bridge（通常是 CPU） |
| 9 | `PATH_SYS` | 经 PCIe 与 NUMA 间互联（QPI/UPI） |
| 10 | `PATH_NET` | 经网络 |
| 11 | `PATH_DIS` | 不连通 |

字符串表 `topoPathTypeStr = {"LOC","NVL","NVB","C2C","PIX","PXB","P2C","PXN","PHB","SYS","NET","DIS"}`（`src/graph/topo.cc:35`）。数值越大越"远"，NCCL 用 `path->type <= level` 判断是否允许某种 transport。`nvidia-smi topo -m` 的 NV#/PIX/PXB/PHB/SYS/NODE 与之大体对应（待确认：`NODE` 与 NCCL 的哪一类对应；确认方式：在真实机器上对比 `nvidia-smi topo -m` 与 `NCCL_TOPO_DUMP_FILE` 导出的 XML 以及 `NCCL_DEBUG_SUBSYS=GRAPH` 的路径打印）。

### 带宽常量（`src/graph/topo.h`，单位 GB/s，NCCL 内部模型值）

`SM80_NVLINK_BW 20.0`、`SM90_NVLINK_BW 20.6`、`SM100_NVLINK_BW 40.1`（每条 NVLink）、`PCI_BW 12.0`（注释 PCI Gen3 x16）、`NET_BW 12.0`（注释 100Gbit）、`SKL_QPI_BW 10.0` 等。注意这些是搜索与代价模型用的"有效带宽"估计，不是硬件规格。

### 搜索 pattern（`src/include/graph.h`）

`NCCL_TOPO_PATTERN_BALANCED_TREE(1)`、`SPLIT_TREE(2)`、`TREE(3)`、`RING(4)`、`NVLS(5)`、`COLLNET_DIRECT(6)`。`struct ncclTopoGraph` 的输出字段：`nChannels`、`bwIntra/bwInter`、`typeIntra/typeInter`、`sameChannels`、`intra[]`（每通道的 GPU 顺序）、`inter[]`（每通道进出的 NET id）。

## 源码导读

### 1. 拓扑 XML：`src/graph/xml.cc`、`topo.cc::ncclTopoGetSystem`

`ncclTopoGetSystem`（`src/graph/topo.cc:1989`）：

1. 若设 `NCCL_TOPO_FILE`，用 `ncclTopoGetXmlFromFile` 读入；否则尝试默认路径 `/var/run/nvidia-topologyd/virtualTopology.xml`（虚拟化场景）。
2. 只探测本进程的 GPU：`ncclTopoFillGpu`（`xml.cc`）→ `ncclTopoGetXmlFromGpu`，通过 sysfs（`ncclTopoGetXmlFromSys`，沿 PCI 路径向上建父节点直到 CPU）和 NVML（NVLink 链路）补属性。
3. 探测网卡：`ncclTopoProcessNet` → `ncclTopoFillNet`；多端口/多 NIC 可能被合成虚拟 NIC（`ncclTopoMakeVNics`，受 `NCCL_IB_MERGE_NICS` 等影响）。
4. `ncclTopoTrimXml` 删掉没有 `keep="1"` 的分支。
5. `bootstrapIntraNodeAllGather` 交换同节点各 rank 的 XML，`ncclTopoFuseXml` 融合成全节点视图（MNNVL 时按 clique 融合）。
6. 若设 `NCCL_TOPO_DUMP_FILE`，`ncclTopoDumpXmlToFile` 输出（只在 `NCCL_TOPO_DUMP_FILE_RANK` 指定的 rank，默认 0；导出调用在 `src/init.cc:1385` 附近）。
7. `ncclTopoGetSystemFromXml` 把 XML 转成 `ncclTopoSystem`：`ncclTopoAddCpu` → `ncclTopoAddPci` → `ncclTopoAddNic/AddGpu`，再 `ncclTopoAddNvLinks`、`ncclTopoAddPciLinks`、`ncclTopoAddC2c`，`ncclTopoConnectCpus` 连接 NUMA 节点，`ncclTopoFlattenBcmSwitches` 处理 Broadcom PCIe 交换芯片。

### 2. 路径计算：`src/graph/paths.cc`

- `ncclTopoSetPaths`：以每个节点为源做 BFS，记录到其他节点的链路列表 `ncclTopoLinkList`，路径 `bw` 取沿途最小带宽，`type` 取沿途最差类型；途经另一块 GPU 的 NVLink 时类型记为 `PATH_NVB`（`NCCL_NVB_DISABLE=1` 可禁止，见 `paths.cc:94-107`）。
- `ncclTopoComputePaths`（`paths.cc:754`）：对 CPU/DEV/GPU/NIC/GIN/RMA/NVS 各类节点调用 `ncclTopoSetPaths`；GPU 间不能/不愿走 P2P 时用 `addInterStep` 让流量经 CPU 中转（注释 "Divert all traffic through the CPU"），连 SHM 都不行的 peer 标记为不可达待 trim；对 NIC 判断能否借 NVLink 相连的 GPU 做 PXN（注释 "PXN = PCI + NVLink"，`NCCL_PXN_DISABLE` 可关）；不能 GDR 时让 GPU-NIC 流量经 GPU 本地 CPU 中转。
- `ncclTopoTrimSystem`：删掉不属于本 comm 的 GPU 和用不到的 NIC。
- 判定函数（transport 层调用）：
  - `ncclTopoCheckP2p`：默认 `p2pLevel = PATH_PXB`（AMD 且 GPU 数 ≤ 2 时放宽到 `PATH_SYS`），用户可用 `NCCL_P2P_LEVEL` / `NCCL_P2P_DISABLE` 覆盖（`ncclGetUserP2pLevel` → `ncclGetLevel`）；还会用 NVML 检查 P2P 读写状态，NVLink 相连却被禁用时报错提示 `NCCL_IGNORE_DISABLED_P2P`；Ampere(cc 80) 经 NVLink 时默认开启 P2P read。
  - `ncclTopoCheckGdr`：网卡与 GPU 都支持 GDR 且距离不超过 `netGdrLevel`（默认 `PATH_PXB`，C2C 平台可到 `PATH_P2C`），用户用 `NCCL_NET_GDR_LEVEL` 覆盖；发送方向（GDR read）额外受 `NCCL_NET_GDR_READ` 控制。
  - `ncclTopoCheckNet`：同机 GPU 是否反而应该走网络（`NCCL_NET_DISABLE_INTRA`）。
- `ncclGetLevel` 同时接受字符串（`LOC/NVL/PIX/PXB/PHB/SYS...`）和旧整数（`levelsOldToNew = {LOC, PIX, PXB, PHB, SYS, SYS}`）。

### 3. 通道搜索：`src/graph/search.cc::ncclTopoCompute`

`ncclTopoCompute`（`search.cc:1151`）是一个"逐步放宽约束"的循环（大量 `goto search`）：

1. 初始约束：`typeIntra = 最小 GPU-GPU 路径类型`，`typeInter = 最小 GPU-NET 路径类型`，`sameChannels = 1`（NVLS 除外），`crossNic` 取 `NCCL_CROSS_NIC`（默认 2，表示"先不跨 NIC，找不到再允许"）。
2. 若设 `NCCL_GRAPH_FILE`，直接从 XML 载入通道（`ncclTopoGetGraphFromXml`），跳过搜索。
3. 速度档位：从 `speedArrayIntra`/`speedArrayInter`（`search.cc` 中如 `{40.0, 30.0, 20.0, ...}`）选不超过系统最大带宽的档位作为每通道目标带宽。
4. `ncclTopoSearchRec` → `ncclTopoSearchRecGpu` / `ncclTopoSearchRecNet`：回溯尝试 GPU 顺序和网卡选择，每走一步用 `ncclTopoFollowPath` 在链路上"扣减"带宽，扣不下就回退。搜索步数预算由 `search.cc:333-336` 的常量控制：`NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL << 19)`、`NCCL_SEARCH_TIMEOUT (1 << 14)`、`NCCL_SEARCH_TIMEOUT_TREE (1 << 14)`、`NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1 << 8)`。
5. 第一轮（pass 1）找不到满意解时依次放宽：`sameChannels=0` → `typeIntra += 1` → `typeInter += 1` → 允许 `crossNic` → 降低速度档位。
6. 找到解后进入 pass 2：在当前解基础上尝试提高 `bwIntra/bwInter`。
7. `ncclTopoDupChannels`：带宽足够高时把通道复制一倍（`bwIntra` 相应减半），以用满链路。

`ncclTopoPrintGraph` 打印 `Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d`，这是读懂 `NCCL_DEBUG_SUBSYS=GRAPH` 日志的关键行。`NCCL_GRAPH_DUMP_FILE` 可导出搜索结果（`ncclTopoDumpGraphs`），配合 `NCCL_GRAPH_FILE` 复现/固定通道。

### 4. 拼全局通道：`src/graph/connect.cc`、`rings.cc`、`trees.cc`

- `ncclTopoPreset`：用本节点搜出的 `intra[]` 设置每个通道 ring 的 prev/next、tree 的 up/down（节点内部分）。
- AllGather3 后 `ncclTopoPostset`：把各节点通道首尾（`ringRecv/ringSend`，即每节点进出网络的 GPU）串成跨节点 ring；用 `ncclGetBtree`/`ncclGetDtree`（`trees.cc`）构建跨节点的双二叉树；源码注释 "Alternate rings to avoid crossing rails"；处理 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`（`connect.cc:331-332`）。
- `rings.cc::ncclBuildRings` 生成最终 ring 并打印 `Channel %02d/%02d : r0 r1 ...`。

## 性能要点与常见坑

- 虚拟机/容器里 sysfs 看不到真实 PCIe 拓扑时，NCCL 可能把 GPU 都判成 `SYS`，导致 P2P/GDR 被禁用、带宽掉档。对策：提供正确的 `NCCL_TOPO_FILE`。
- PCIe ACS 打开会让 P2P 流量绕到 Root Complex，表现为 P2P 带宽低或数据错误（userguide `troubleshooting/gpu_troubleshooting.rst` 有 "PCI Access Control Services (ACS)" 一节）。
- `NCCL_P2P_LEVEL`/`NCCL_NET_GDR_LEVEL` 推荐用字符串，不要用整数：文档明确说整数编号会随版本变化（`docs/userguide/source/env.rst` 的 NCCL_P2P_LEVEL 节）。
- 通道数不是越多越好：通道数 = kernel 的 block 数，会占 SM；`NCCL_MIN/MAX_NCHANNELS`、`NCCL_MIN/MAX_CTAS` 可调，但要和计算重叠一起评估（ai-infra-book 6.4.3 讨论了通信与计算争用 SM/HBM 的实验）。
- 多 NIC 节点上，同一节点不同 rank 搜到的通道数可能不同，`ncclTopoPostset` 会取最小值对齐；单个 rank 的网卡异常会拖累全局通道数。

## 面试自测

1. 问：PIX、PXB、PHB、SYS 有什么区别？对 NCCL 的影响是什么？
   答：PIX 至多一个 PCIe 桥；PXB 多个桥但不经 Host Bridge；PHB 经过 CPU 的 Host Bridge；SYS 还要跨 NUMA 互联。NCCL 默认 P2P 只允许到 PXB，GDR 默认也到 PXB；超出就退回 SHM/经 CPU 中转，或网络数据先拷到 host 内存。
2. 问：PATH_NVB 和 PATH_PXN 分别解决什么问题？
   答：NVB 是两块 GPU 没有直接 NVLink 时借中间 GPU 的 NVLink 转发，比走 PCIe 快；PXN 是 GPU 把数据经 NVLink 交给与目标网卡同 PCIe 交换芯片的另一块 GPU，再由它发出，实现 rail 对齐、避免跨 PCIe/CPU 访问远端网卡。
3. 问：`ncclTopoCompute` 找不到理想解时按什么顺序放宽？
   答：先允许各通道不同（sameChannels=0），再放宽节点内路径类型，再放宽 GPU-NIC 路径类型，再允许 crossNic，最后降速度档位；有解后 pass 2 尝试往上提带宽。
4. 问：如何固定或复现某台机器的拓扑与通道，用于离线分析？
   答：`NCCL_TOPO_DUMP_FILE` 导出拓扑 XML、`NCCL_GRAPH_DUMP_FILE` 导出通道；之后用 `NCCL_TOPO_FILE`、`NCCL_GRAPH_FILE` 加载。
5. 问：为什么 `nvidia-smi topo -m` 显示 NV18，NCCL 还可能不走 P2P？
   答：NCCL 会再用 NVML 查 P2P 读写状态、`cudaDeviceCanAccessPeer`、legacy IPC/cuMem 可用性和用户 `NCCL_P2P_*` 设置；任何一项不满足都会退回其他 transport（`ncclTopoCheckP2p`、`p2pCanConnect`）。

## 待深入

- 搜索步数预算耗尽（`time == 0`）时得到的是"超时前的最好解"，不一定最优；源码中调试 `printf` 会标 `TIMEOUT`/`PERFECT`（`search.cc` 的 `ncclTopoCompute` 内）。
- `ncclTopoSelectNets` 中 "GPU first / channel first" 两种网卡排序策略（`ncclTopoPrefNetsGpuFirst`、`ncclTopoPrefNetsChannelFirst`）与 `NCCL_MNNVL_SCATTER_NETS_ENABLE`。
- NVLS 与 CollNet Direct 的搜索（`ncclTopoSearchTryNvls`、`ncclTopoSearchTryCollnetDirect`）。
