# 01-rdma：RDMA（scale-out 下层）

> 层次：下层 · 来源：third_party/rdma-core@bd3282a、third_party/perftest@05200c2、third_party/nccl@3e46c08、third_party/ai-infra-book@3bdcb4fc（第 6、7 章）

## 一句话

RDMA 是集合通信库跨节点（scale-out）路径的"最后一层软件 + 第一层硬件"：通信库把每一次跨机搬运翻译成 verbs 调用（post WR / poll CQ），由 RNIC 在硬件里完成可靠传输、拥塞控制，再经交换机网络送到对端。

## 在调用链中的位置

```
PyTorch / Megatron / vLLM ...            (05-framework-integration)
        │  allreduce / alltoall / send/recv
NCCL：算法(Ring/Tree) → channel → proxy  (03-ccl-algorithms, 04-nccl-internals)
        │  ncclNet_t 接口：listen/connect/accept/regMr/isend/irecv/test
NCCL net_ib transport                     third_party/nccl/src/transport/net_ib/
        │  ibv_reg_mr / ibv_create_qp / ibv_modify_qp / ibv_post_send / ibv_poll_cq
libibverbs (rdma-core 用户态)             third_party/rdma-core/libibverbs/
        │  provider（如 mlx5）：写 WQE、敲 doorbell（MMIO）、轮询 CQE
        │                                 third_party/rdma-core/providers/mlx5/
RNIC 硬件：传输层（RC 可靠性/重传）、拥塞控制（DCQCN 等）、DMA（含 GPUDirect）
        │
交换机网络：IB 或 RoCEv2 以太网；PFC / ECN / ECMP / 自适应路由
```

- 上层：NCCL 的 `ncclNetIb`（`third_party/nccl/src/transport/net_ib/common.cc`）把 `ncclNet_t` 的函数表填成 `ncclIbConnect`、`ncclIbRegMr`、`ncclIbIsend` 等，NCCL 核心通过这张表调用 IB transport。
- 下层：verbs 通过 provider 驱动直接在用户态与网卡交互（kernel bypass），控制面（建 PD/MR/QP）走内核，数据面（post/poll）不进内核。
- 对 JD 的意义："自研 RDMA 协议、拥塞算法"改的正是 RNIC/交换机这一段，以及 verbs 之上它暴露给通信库的语义（顺序、完成、多路径）。读懂 NCCL 怎么用 verbs，才能判断协议改动会影响通信库的哪些假设。

## 章节索引

| 文件 | 主题 |
|---|---|
| [01-verbs-basics.md](01-verbs-basics.md) | PD/MR/CQ/QP/SRQ/AH、QP 状态机、建链信息交换，rc_pingpong 导读 |
| [02-transport-and-ops.md](02-transport-and-ops.md) | RC/UC/UD/XRC/DC、SEND/RECV vs WRITE/READ/ATOMIC、WRITE_WITH_IMM、inline、selective signaling、doorbell |
| [03-roce-ib-congestion.md](03-roce-ib-congestion.md) | IB vs RoCEv2、PFC/ECN/DCQCN、Go-back-N vs 选择重传、incast/PFC 死锁/ECMP 冲突，与 JD 的关系 |
| [04-perftest.md](04-perftest.md) | ib_write_bw/ib_write_lat/ib_send_bw 用法、参数、结果解读、源码结构 |
| [05-gpudirect-and-nccl-ib.md](05-gpudirect-and-nccl-ib.md) | GPUDirect RDMA（nvidia-peermem / dmabuf）、NCCL net_ib 建链/MR/多 QP/GDR 判断、NCCL_IB_* 环境变量 |

建议阅读顺序：01 → 02 → 04（边读边跑 perftest）→ 03 → 05。

## 面试自测（问题 + 要点答案）

1. **从 `ncclAllReduce` 到网线上的一个 RoCE 包，中间经过哪些层？**
   NCCL 算法 / channel → proxy 线程 → `ncclNet_t`（net_ib）→ verbs `ibv_post_send` → provider 写 WQE、敲 doorbell → RNIC 通过 DMA 读 GPU 显存（GDR），封装 BTH/UDP/IP → 交换机按 DSCP 入无损队列、按 ECMP 选路 → 对端 RNIC 写入对端 GPU 显存，产生 IMM 完成。
2. **RDMA 的 "kernel bypass" 具体绕过了什么？哪些操作仍然要进内核？**
   数据面（post WR、poll CQ、doorbell）在用户态直接访问网卡映射的内存和寄存器；控制面（open device、alloc PD、reg MR、create/modify QP）走内核 uverbs 命令。
3. **集合通信库研发为什么要懂下层协议和拥塞控制？**
   通信库的完成语义、QP 规模、多路径切分、发送调度（incast）都建立在传输协议的假设之上；协议一旦改了（乱序、多路径、选择重传），通信库就要配合修改，而算法层的调度也能反过来减轻拥塞（见 03）。

## 待深入

- 返回 [笔记总入口](../00-roadmap/README.md)；上层调用方见 [04-nccl-internals](../04-nccl-internals/README.md)，scale-up 对照见 [02-scale-up](../02-scale-up/README.md)。
