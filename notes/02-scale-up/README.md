# 02 Scale-up 互联（章节索引）

> 层次：下层 scale-up · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）、ai-infra-book@3bdcb4fc 第 1 章、第 6 章 6.5 节、第 7 章 7.1~7.3 节

## 一句话

scale-up 是"一个互联域内的加速器像共享一块大内存那样协作"（NVLink/NVSwitch、机柜级 NVL72），scale-out 是"多个域通过网卡和交换网络相连"（IB/RoCE）；NCCL 在 scale-up 侧用 P2P（CUDA IPC / cuMem 映射对端显存）、SHM（经主机内存）、NVLS（NVSwitch multicast 与交换机内归约）三类 transport。

## 在调用链中的位置

- 上层：NCCL 的 channel 与 primitives（[../04-nccl-internals/03-channels-protocols-kernels.md](../04-nccl-internals/03-channels-protocols-kernels.md)）读写这些 transport 建立的缓冲。
- 本章：硬件概念 + `third_party/nccl/src/transport/{p2p,shm,nvls,multicast,nvls_ub,mc_arena}.cc`、`src/mnnvl.cc`、`src/ce_coll.cc`。
- 平行：scale-out 侧的 RDMA 与 `net_ib` 见 [../01-rdma/README.md](../01-rdma/README.md)。
- 拓扑如何把路径判成 NVL/NVB/PIX 等见 [../04-nccl-internals/02-topology-graph.md](../04-nccl-internals/02-topology-graph.md)。

## 本章笔记

| 文件 | 主题 |
|---|---|
| [01-nvlink-nvswitch.md](01-nvlink-nvswitch.md) | NVLink/NVSwitch/NVL72 概念、与 PCIe 对比、scale-up 与 scale-out 的分工 |
| [02-nccl-p2p-shm-nvls.md](02-nccl-p2p-shm-nvls.md) | NCCL 的 P2P / SHM / NVLS transport 源码导读、CUDA IPC 与 cuMem、MNNVL、NVLS multicast 与 SHARP 式归约 |

两篇的关系：左边是 01 讲的硬件路径，右边是 02 讲的 NCCL 用法和源码位置（路径相对 `third_party/nccl/src/`）。

```text
  hardware path (01)                        NCCL mechanism (02)
  ──────────────────                        ───────────────────
  NVLink / NVSwitch  ── load/store ────────► P2P       transport/p2p.cc   (CUDA IPC / cuMem)
                     ── multicast + SHARP ─► NVLS      transport/nvls.cc, multicast.cc
  MNNVL (cross-node  ── fabric handle ─────► P2P/MNNVL mnnvl.cc           (CU_MEM_HANDLE_TYPE_FABRIC)
         NVLink)
  PCIe P2P           ── within P2P_LEVEL ──► P2P       (default up to PXB)
  host shared memory ── GPU->host->GPU ────► SHM       transport/shm.cc
  copy engine        ── no SM used ────────► P2P/CE, CE collectives  ce_coll.cc
  NIC (scale-out)    ── RDMA ──────────────► NET       see ../01-rdma/
```

## 与岗位的关系

"基于 scale-up 互联研发通信库"意味着要能回答：给定一个 NVLink/NVSwitch 或自研 scale-up 域，如何（1）建立跨 GPU 的内存映射（IPC、fabric handle），（2）设计 load/store 还是 copy engine 还是 multicast 的数据路径，（3）在 kernel 里做同步与流控，（4）把它接入通信库的拓扑、选算法与 transport 框架。NCCL 的 `p2p.cc`、`nvls.cc`、`mnnvl.cc`、`ce_coll.cc` 正好覆盖这四个问题，是最好的参考实现。

## 面试自测

1. 问：scale-up 与 scale-out 的根本区别是什么？
   答：ai-infra-book 第 1 章的定义：超节点内部互联称 scale-up，让一组加速器以较低通信开销共同计算；超节点之间经网卡与交换网络相连称 scale-out，用于扩大集群。前者通常是内存语义（load/store、直接映射对端显存）、带宽高一个数量级、域规模受交换芯片端口和一致性开销限制；后者是消息/RDMA 语义，规模大，但要处理拥塞、故障与 CPU/网卡提交开销。
2. 问：NCCL 在同一台机器的两块 GPU 之间可能用哪些 transport？由什么决定？
   答：按 `ncclTransports[] = {P2P, SHM, NET, CollNet}` 顺序尝试 `canConnect`：P2P（NVLink 或 PCIe P2P，受 `NCCL_P2P_LEVEL`/拓扑/NVML/IPC 能力决定）、不行则 SHM（同 host 且同 `/dev/shm`）、再不行或拓扑判定网络更优时走 NET；集合通信还可能整体使用 NVLS。

## 待深入

- 自研 scale-up 互联（如 UB）接入通信库时需要新增的抽象：ai-infra-book 6.5.5、7.3.3、7.4.2 对 UB/URMA 的讨论。
- NCCL device API（`src/include/nccl_device/`、LSA/multimem）对 scale-up 编程模型的封装。
