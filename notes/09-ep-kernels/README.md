# 09 中层：EP 专用通信库（DeepEP LL 与 Hybrid-EP）

> 层次：中层（专用通信库，贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP-hybrid-ep@10d4dd73（`hybrid-ep` 分支，含 V1 normal / low-latency 与 NVIDIA Hybrid-EP）、third_party/DeepEP@93eb6eb2（V2.5）、third_party/nccl@3e46c089（`contrib/nccl_ep`）、third_party/vllm@02653fa2、third_party/sglang@d2a43644

## 一句话

MoE 的 dispatch/combine 是"动态、不均匀、要和 token 重排融合"的 All-to-All-v，通用 `ncclAlltoAll` 做不好，于是有了专用库。其中两套算法最值得吃透：

- **DeepEP low-latency（LL）**：decode 用。每个 (token, top-k 专家) 直接一次 RDMA WRITE 写进对端为它预留的槽位，靠"同一 QP 上数据先到、计数后到"完成同步。不需要 CPU 同步，可被 CUDA Graph 捕获，还能用 recv hook 做到"发出后不占 SM"。
- **Hybrid-EP（NVIDIA）**：训练和 prefill 用。先用一次 AllGather 让所有 rank 拿到全局路由，算出每个 token 的确定落点；然后按"跨节点 RDMA 只发给同号 GPU 一次 → 节点内 NVLink 扇出"分层传输，用 TMA + warp 专业化流水线压低 SM 占用。

## 为什么单独成章

- 第 [05 章 DeepEP 笔记](../05-framework-integration/04-deepep.md) 站在框架视角，讲的是"谁在调用、版本怎么演进"。本章站在**通信库作者**视角，讲 kernel 内部：槽位怎么分、信号怎么发、缓冲区怎么排、网卡怎么驱动。
- 这两套算法已经被 NCCL 官方吸收：`third_party/nccl/contrib/nccl_ep/device/ll_ep.cuh` 与 `hybrid_ep.cuh` 文件头都写着改编自 DeepEP，只是把 NVSHMEM/DOCA 换成了 NCCL GIN/LSA（见 [04](04-lineage-and-selection.md)）。理解原版，就理解了 NCCL EP。

## 本章笔记

| 文件 | 主题 | 面向的问题 |
|---|---|---|
| [01-design-space.md](01-design-space.md) | EP All2All 的设计空间：槽位分配、完成信号、缓冲区、分层、SM 占用 | "设计一个 EP 通信库要做哪几个决定" |
| [02-deepep-low-latency.md](02-deepep-low-latency.md) | DeepEP LL：dispatch/combine kernel、IBGDA、双缓冲、recv hook、LogFMT、NVLink 旁路 | "decode 的 dispatch 为什么能做到 ~100 µs、还不占 SM" |
| [03-hybrid-ep.md](03-hybrid-ep.md) | Hybrid-EP：路由预处理、warp 专业化（dispatch 每 block 3 类 warp group 共 4 warp，combine 5 类 warp group 共 6/11 warp）、DOCA GPUNetIO RDMA、chunk 标志、combine 节点内预归约、permute 融合 | "大 batch 下怎样用 4~16 个 SM 跑满网卡" |
| [04-lineage-and-selection.md](04-lineage-and-selection.md) | 谱系（V1 → Hybrid-EP → NCCL EP → DeepEP V2.5）、横向对比、框架里怎么选 | "面试时怎么把几个方案放在一张表里讲清楚" |

## 读这一章的顺序

1. 先读 [01](01-design-space.md)，带着"五个设计决定"去看源码。
2. 读 [02](02-deepep-low-latency.md)，对照 `third_party/DeepEP-hybrid-ep/csrc/kernels/internode_ll.cu` 逐段看。
3. 读 [03](03-hybrid-ep.md)，对照 `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` 的 `dispatch_kernel` / `combine_kernel`。
4. 读 [04](04-lineage-and-selection.md)，把它们放回 NCCL EP 与 DeepEP V2.5 的演进里。

前置知识：[RDMA 传输与操作](../01-rdma/02-transport-and-ops.md)（RC 的按序语义）、[All2All 算法](../03-ccl-algorithms/04-all2all.md)、[device-initiated 通信](../06-comm-compute-fusion/03-nvshmem-and-device-initiated.md)。

## 两张图记住两种算法

```text
DeepEP LL dispatch（每个 rank 都这样做，所有目的地一视同仁，节点内外都走 RDMA 或 NVLink 直写）

 rank r 的 token t，top-k = {e1, e2, ...}
   ├─ e1 在 rank d1：slot = atomicAdd(本地计数[e1])            ← 本地分配槽位
   │                 RDMA WRITE  t → d1.recv[e1_local][r][slot]
   ├─ e2 在 rank d2：……（同一 rank 上的两个专家也各发一份，不去重）
   └─ 所有 token 发完后：对每个专家 e，RDMA ATOMIC  d.count[e_local][r] = -(n+1)
                         （与数据走同一个 QP → 计数一定晚于数据可见）
 接收端：轮询 count[e_local][r] ≠ 0 → 知道 r 发来 n 个 → 拷进 [num_local_experts, R×N, hidden]
```

```text
Hybrid-EP dispatch（节点 = NVLink 域，peer = 不同节点上同 local rank 的 GPU）

 0. AllGather 全局路由 → scan kernel 算出三张表：
      attn_to_rdma_map  我的 token 要去哪些远端节点
      rdma_to_attn_map  每个节点需要哪些 token
      sparse_to_dense_map  每个 token 在每个本地 rank 输出缓冲里的确定下标
 1. RDMA warp：每个 chunk、每个远端节点，只把需要的 token 写到 peer 的同下标位置，
               最后一个 ATOMIC FA 把该 chunk 的标志 +1
 2. G2S warp：本节点 token 直接读；远端节点的 chunk 先等标志到期 → TMA 读进 SMEM 环形队列
 3. S2G warp：按 sparse_to_dense_map，用 TMA 经 NVLink 写进节点内每个需要它的 rank
```

## 面试自测

1. **问**：LL dispatch 不做 CPU 同步，接收端怎么知道收了多少 token、数据是否到齐？
   **要点**：发送端写完一个专家的所有 token 后，向接收端的 `count[local_expert][src_rank]` 发一个原子写，值编码为 `-(n+1)`（让 0 表示"还没到"）。数据与计数在同一个 RC QP 上，按序执行，计数可见时数据已经落地。接收端只需轮询这个计数。见 [02](02-deepep-low-latency.md)。
2. **问**：LL 为什么能被 CUDA Graph 捕获，代价是什么？
   **要点**：接收缓冲按最坏情况预分配（每个本地专家 × 每个源 rank × `num_max_dispatch_tokens_per_rank` 个槽位），输出形状固定，不需要把计数拷回 CPU 分配张量。代价是显存：DeepSeek-V3 量级、256 专家、每 rank 128 token 时约 1.8 GB；接收方 GEMM 还必须用 `recv_count` 识别有效行。
3. **问**：recv hook 为什么"不占 SM"？什么情况下这个说法不成立？
   **要点**：kernel 只跑发送阶段（`LOW_LATENCY_SEND_PHASE`）就退出，数据由网卡在后台搬；调用 `hook()` 时才再 launch 一次接收阶段。如果开了 NVLink 旁路（`allow_nvlink_for_low_latency_mode`），节点内的数据由 SM 直接 store，发送阶段就要更久地占着 SM，`deep_ep/buffer.py` 的参数说明也提醒它和 hook 重叠"某种程度上不兼容"。
4. **问**：Hybrid-EP 和 DeepEP V1 normal 都是"先 RDMA 到同号 GPU，再 NVLink 转发"，Hybrid-EP 改进了什么？
   **要点**：(1) 先 AllGather 路由，落点是确定的，不用在 kernel 里排队、分配槽位；(2) 数据搬运全用 TMA，warp 专业化成 G2S / S2G / RDMA 流水线，4~16 个 SM 就能跑满网卡；(3) 用 DOCA GPUNetIO 直接在 GPU 上组 WQE；(4) combine 先在节点内预归约，跨节点只发一份；(5) 可选把 permute 融合进 kernel，直接输出按专家分组的布局。
5. **问**：为什么 LL 在同一 rank 的多个专家间不去重，Hybrid-EP 却去重？
   **要点**：LL 的接收缓冲按"专家"切分，每个 (token, 专家) 有独立槽位，发送端不需要知道对端还有哪些专家选了它，路径最短、延迟最低；decode token 少，多发几份的带宽代价可以接受。Hybrid-EP 面向大 batch，带宽是瓶颈，所以跨节点按"节点"去重、节点内按"rank"去重，并把概率向量一起带过去。

## 待深入

- 待确认：同一 RC QP 上 RDMA WRITE 之后的 RDMA ATOMIC，在开启网卡乱序放置/自适应路由时是否仍保证"原子结果可见 ⇒ 之前的 WRITE 数据可见"；确认方式：查阅 IBTA 规范的 ordering 章节和 NVIDIA 网卡文档，或在双机上用 `third_party/DeepEP-hybrid-ep/tests/test_low_latency.py` 开启 AR 做压力测试。
- 待确认：Megatron-LM 中 Hybrid-EP 的调用位置（`HybridEPBuffer`）；本地 vLLM/SGLang 未调用它；确认方式：克隆 NVIDIA/Megatron-LM 后 grep `hybrid_ep`。
- 未运行：本章所有 kernel 都需要 Hopper/Blackwell GPU + IB/RoCE 网卡，本机 not-run；有机器后运行 `tests/test_low_latency.py`、`tests/test_hybrid_ep.py`，把实测带宽/延迟补进 02、03。
