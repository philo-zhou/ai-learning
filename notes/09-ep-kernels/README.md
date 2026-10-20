# 09 中层：EP 专用通信库（DeepEP V2.5 源码精读）

> 层次：中层（专用通信库，贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP@b5a33eef（V2.5）、third_party/DeepInfer@774ce244、third_party/nccl@3e46c089（GIN device API、`contrib/nccl_ep`）、third_party/vllm@02653fa2、third_party/sglang@d2a43644；历史对照引用上游 DeepEP `hybrid-ep` 分支@10d4dd73（未克隆）

> **版本说明（2026-10-21）**：本章以本地克隆的 DeepEP 最新主线 V2.5 为准。V1 LL、V1 normal、Hybrid-EP 只作为对照出现在 [01](01-design-space.md) 与 [05](05-lineage-and-selection.md)，引用写作 `hybrid-ep@10d4dd73/<路径>`，指上游 [`hybrid-ep` 分支](https://github.com/deepseek-ai/DeepEP/tree/10d4dd7377d5bce900fbb4b80cce863764892b95)，本地不再克隆。注意本地 vLLM/SGLang 的 `deepep_low_latency` / `low_latency` 路径调用的仍是 V1 接口，见 [05 的"框架里怎么选"](05-lineage-and-selection.md)。

## 一句话

MoE 的 dispatch/combine 是"动态、不均匀、要和 token 重排融合"的 All-to-All-v，通用 `ncclAlltoAll` 做不好，于是有了专用库。DeepEP V2.5 用一个 `EPBuffer` 覆盖训练、prefill 和 decode，按拓扑分两种模式：

- **direct**（单节点，或关掉 hybrid）：每个 token 对每个目标 rank 发一份，发送端本地 `atomicAdd` 分槽位，NVLink 用 TMA 直写、跨节点用 NCCL GIN 的 `put`；同一个 kernel 里的 notify warp 并行交换计数，结束后由 PDL 衔接的 epilogue 按专家展开。
- **hybrid**（多节点默认）：跨节点只经 RDMA 发给目标节点上的同号 GPU 一次，由它的 forward warp 经 NVLink 扇出；channel 私有 tail 做进度通知、不需要环形队列；combine 沿 dispatch 时记下的 linked list 原路返回，并在三处归约。

decode 不再有单独的 LL kernel，而是 `do_cpu_sync=False`：不等 CPU、输出按容量分配、可 CUDA Graph。

## 为什么单独成章

- 第 [05 章 DeepEP 笔记](../05-framework-integration/04-deepep.md) 站在框架视角，讲的是"谁在调用、版本怎么演进"。本章站在**通信库作者**视角，讲 kernel 内部：槽位怎么分、信号怎么发、缓冲区怎么排、网卡怎么驱动。
- 同类实现已经进了 NCCL：`third_party/nccl/contrib/nccl_ep/device/ll_ep.cuh` 与 `hybrid_ep.cuh` 文件头都写着改编自 DeepEP，DeepEP V2.5 自己也建在 NCCL GIN 上。读懂 V2.5，NCCL 的 device API 怎么用来写 EP 也就清楚了。

## 本章笔记

| 文件 | 主题 | 面向的问题 |
|---|---|---|
| [01-design-space.md](01-design-space.md) | EP All2All 的设计空间：槽位分配、完成信号、缓冲区、分层、SM 占用（V1 / Hybrid-EP / V2.5 对照） | "设计一个 EP 通信库要做哪几个决定" |
| [02-deepep-v25-overview.md](02-deepep-v25-overview.md) | V2.5 总览：调用链、JIT、对称内存与 GIN、token 布局、workspace、`NCCLGin`、`gpu_barrier`、deferred epilogue、SM/QP 估算 | "一次 dispatch 从 Python 到 kernel 经过了什么" |
| [03-deepep-v25-direct.md](03-deepep-v25-direct.md) | direct 模式：notify 与数据并行、发送端分槽位、copy epilogue 展开、combine 的 min(R,k) 行与本地归约 | "单机 8 卡 / decode 时 dispatch 和 combine 怎么做" |
| [04-deepep-v25-hybrid.md](04-deepep-v25-hybrid.md) | hybrid 模式：channel 与 rail、两级 notify、scale-out tail、forward warp、linked list、三级归约 | "多机时怎样让每个 token 跨网只走一次" |
| [05-lineage-and-selection.md](05-lineage-and-selection.md) | 谱系（V1 → Hybrid-EP → NCCL EP → DeepEP V2.5）、横向对比、框架里怎么选 | "面试时怎么把几个方案放在一张表里讲清楚" |

## 读这一章的顺序

1. 先读 [01](01-design-space.md)，带着"五个设计决定"去看源码。
2. 读 [02](02-deepep-v25-overview.md)，同时打开 `third_party/DeepEP/deep_ep/buffers/ep.py` 和 `third_party/DeepEP/csrc/buffers/ep.hpp`，跟一遍 `dispatch` 的调用链；`layout/ep/token.cuh`、`comm/handle.cuh`、`comm/barrier.cuh` 是后面 kernel 的公共基础。
3. 读 [03](03-deepep-v25-direct.md)，对照 `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/dispatch.cuh`（407 行，最该精读）→ `dispatch_copy_epilogue.cuh` → `combine.cuh` → `combine_reduce_epilogue.cuh`。
4. 读 [04](04-deepep-v25-hybrid.md)，对照 `hybrid_dispatch.cuh` 与 `hybrid_combine.cuh`。
5. 读 [05](05-lineage-and-selection.md)，把 V2.5 放回 V1、Hybrid-EP、NCCL EP 的演进里，并看框架怎么选。

`ptx::deduplicate`、`ptx::exchange`、`tma_load_1d`、`mbarrier_*` 第一次遇到时去 `third_party/DeepEP/deep_ep/include/deep_ep/common/ptx.cuh` 查定义，都很短。

前置知识：[RDMA 传输与操作](../01-rdma/02-transport-and-ops.md)（RC 的按序语义）、[All2All 算法](../03-ccl-algorithms/04-all2all.md)、[device-initiated 通信](../06-comm-compute-fusion/03-nvshmem-and-device-initiated.md)（GIN、LSA、强/弱信号）。

## 两张图记住两种模式

```text
direct dispatch（R 个 rank 一层，单节点时全走 NVLink）

 rank r 的 token t，top-k = {e1@d1, e2@d1, e3@d2}
   ├─ 按目标 rank 去重：d1 只发一份（e1、e2 由 d1 的 epilogue 展开成两行）
   ├─ slot = atomicAdd(本地 sender_count[d])                 ← 发送端定落点
   ├─ NVLink 可达：TMA 直写 d.recv[r][slot]；否则 TMA → send[t]，gin.put → d.recv[r][slot]
   └─ 同时 notify warp：统计 → 网格归约 → 把"发给你几个 / 你每个专家几个"写给每个 rank
 结尾：gpu_barrier（flush 全部 QP + QP 0 信号）→ PDL → copy epilogue 按前缀和展开
```

```text
hybrid dispatch（节点 = NVLink 域，rail peer = 其他节点上同 local rank 的 GPU）

 节点 A GPU j，channel c                节点 B GPU j（forwarder）              节点 B GPU k
  按目标节点去重，slot = tail[c][B]
  TMA → send，gin.put<Rail> ──RDMA──► scaleout_recv[A][c][slot]
  每 6 个 token red_add_rel(tail) ──► signaled_tail[c][A]
                                       轮询 tail，取 ≤6 个，按目标 GPU 去重
                                       atomicAdd → slot'，TMA ──NVLink──► scaleup_buffer[j][slot']
                                       记 token_metadata_at_forward、linked list 下标
 combine：k 沿 linked list 把部分和写回 B/j → B/j 重放 metadata、节点内归约 → RDMA 回 A/j → A/j 跨节点归约
```

## 面试自测

1. **问**：DeepEP V2.5 还有 LL kernel 吗？decode 怎么做到低延迟、可 CUDA Graph？
   **要点**：没有。同一个 `EPBuffer.dispatch` 传 `do_cpu_sync=False`：kernel 不等 CPU，输出按 `num_max_tokens_per_rank × R` 的容量分配，真实数量留在 GPU 上的 `psum_num_recv_tokens_per_expert`。失去的是 V1 LL recv hook 的零 SM 等待，替代手段是 `defer_epilogue`。见 [02](02-deepep-v25-overview.md)、[03](03-deepep-v25-direct.md)。
2. **问**：V2.5 怎么保证"看到完成就能读数据"？
   **要点**：direct 和节点内一级靠 `gpu_barrier`：flush 所有 QP（RC 本地完成 = 对端已 ACK）、grid sync，再用 QP 0 的 `SignalInc` 互相通知；NVLink 部分用对称计数器的 `red.release`。hybrid 跨节点一级靠 tail 信号与数据同 GIN context，依赖强信号语义。见 [02](02-deepep-v25-overview.md) 第 9 节、[04](04-deepep-v25-hybrid.md) 第 4 节。
3. **问**：direct 和 hybrid 的落点分别由谁决定？为什么都不需要接收端先回复？
   **要点**：direct 由发送端对每个目标 rank 的本地计数器 `atomicAdd`；hybrid 跨节点由 channel 私有 tail（寄存器递增），节点内由 forwarder `atomicAdd`。接收区按最坏情况预分配，每个源（或每个源 × channel）独占一段，所以发送端自己就能定位置。
4. **问**：combine 的数据为什么能直接写到源 rank 上"原始 token 的那一行"？
   **要点**：dispatch 时 token 的 metadata 里带着 `src_token_global_idx = rank × M + token`，copy epilogue 把它存进 `recv_src_metadata`；combine 按它把结果写到源 rank 接收缓冲的第 `token` 行，行的第一维只需要 min(R,k)（direct）或 min(S,k)（hybrid）格。
5. **问**：hybrid 为什么对多 rail / 多 plane 网络友好？
   **要点**：跨节点流量只在各节点同号 GPU 之间（`ncclTeamTagRail`、`NCCL_GIN_CONNECTION_RAIL`），每张卡的流量只走自己那条 rail；同一 token 去同一节点多张卡只跨网一次。direct 需要 FULL 连接，多 plane 网络上会因 GIN 不可用而失败。

## 待深入

- 待确认：NCCL GIN 强/弱信号在 DeepEP 所需版本（NCCL ≥ 2.32.3）中的默认映射，以及开启网卡乱序放置/自适应路由后 tail 信号的可见性保证；确认方式：读对应 NCCL 版本 `ncclDevCommCreate` 与 GDAKI 后端实现，或双机开 AR 压测 `tests/ep/test_ep.py`。
- 待确认：DeepJIT 的编译缓存与多进程并发编译；确认方式：把 `deepseek-ai/DeepJIT` 加入 `repos.lock` 后阅读。
- not-run：本章所有 kernel 都需要 Hopper/Blackwell GPU（DeepInfer 需 SM100）+ IB/RoCE 网卡；有机器后运行 `third_party/DeepEP/tests/ep/test_ep.py`（单机 direct、双机 hybrid），把实测带宽/延迟与 `EP_BUFFER_DEBUG=1` 打印的 SM/channel 数补进 03、04。
