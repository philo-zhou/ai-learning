# EP All2All 的设计空间：五个必须做的决定

> 层次：中层 · 来源：third_party/DeepEP-hybrid-ep@10d4dd73、third_party/DeepEP@93eb6eb2、third_party/nccl@3e46c089（`contrib/nccl_ep`）、ai-infra-book 第 7、9 章

## 一句话

任何 EP dispatch/combine 实现都要回答五个问题：**落点由谁决定、完成怎么通知、缓冲区按什么上界分配、走直连还是分层、用多少 SM 搬数据**。DeepEP LL 和 Hybrid-EP 是这五个问题的两组典型答案。

## 在调用链中的位置

```text
MoE 层：router → topk_idx / topk_weights
   │  dispatch(x, topk_idx)            combine(expert_out, handle, topk_weights)
   ▼                                    ▲
EP 通信库（DeepEP LL / Hybrid-EP / NCCL EP / DeepEP V2.5）
   │  决定：槽位、信号、缓冲、拓扑、SM
   ▼
GPU 直驱网卡（NVSHMEM IBGDA / DOCA GPUNetIO / NCCL GIN）+ NVLink load/store 或 TMA
```

## 核心概念

### 决定 1：落点由谁决定（槽位分配）

接收端事先不知道会收到多少 token、从谁那里收。有三种做法：

| 做法 | 谁定落点 | 代表 | 代价 |
|---|---|---|---|
| **发送端本地原子分配**：发送端对"目标专家"计数器 `atomicAdd`，得到在该专家、该源 rank 分区里的槽位号 | 发送端 | DeepEP LL：`atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)`（`third_party/DeepEP-hybrid-ep/csrc/kernels/internode_ll.cu` dispatch 发送段） | 每个 (专家, 源 rank) 都要预留最大容量；接收端拿到的是"分区 + 计数"，要再整理 |
| **先交换计数再分配**：先跑一个 notify/layout 阶段交换每对 rank 的数量，接收端算前缀和后分配 | 双方协商 | DeepEP V1 normal（`get_dispatch_layout` + notify） | 多一轮通信；要把计数同步给 CPU 才能分配输出张量，和 CUDA Graph 冲突 |
| **全局路由先 AllGather，各自算出确定下标** | 所有人用同一份全局视图算出同一结果 | Hybrid-EP：`Executor::allgather_routing_map` + `scan` kernel 生成 `sparse_to_dense_map`（`third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/executor/executor.cu`、`backend/hybrid_ep_backend.cuh`） | 多一次 AllGather（路由表很小：token 数 × 专家数个 bool 或 top-k 下标）；换来的是 kernel 里不需要任何排队与原子分配 |

这个决定直接对应前面讨论过的 SEND/RECV 与 WRITE 的区别：EP 库**全部用 RDMA WRITE（内存语义）**，所以必须由某一方提前知道远端地址。上面三种做法就是三种"提前知道"的方式。

### 决定 2：完成怎么通知（信号）

单边 WRITE 不会在接收端产生 CQE，接收端要靠**轮询内存里的某个标志**判断数据到齐。几种实现都用同一个技巧：**数据和标志走同一个 RC QP，标志排在数据后面**，利用 RC 按序执行，让"标志可见"蕴含"数据可见"。

| 实现 | 数据 | 标志 | 粒度 |
|---|---|---|---|
| DeepEP LL dispatch | `nvshmemi_ibgda_put_nbi_warp`（RDMA WRITE），QP 号 = 目标本地专家号 | `nvshmemi_ibgda_amo_nonfetch_add` 写 `-(n+1)`，同一 QP 号 | 每 (专家, 源 rank) 一个计数 |
| DeepEP LL combine | 同上 | 原子加 1 到 `rdma_recv_flag[global_expert]` | 每 (专家, 源 rank) 一个标志 |
| Hybrid-EP dispatch | `doca_gpu_dev_verbs_wqe_prepare_write` 若干个 | 同一批 WQE 末尾一个 `ATOMIC_FA` 加 1 | 每 (chunk, 源节点) 一个 64 位标志，期望值逐次递增 |
| NCCL EP LL | `ncclGin::put` | `ncclGin_SignalAdd{signalId, numTokensSent + 1}` 随最后一次 put 一起带上 | 每 (专家, 源 rank) 一个 GIN signal |

为什么 LL 写 `-(n+1)` 而不是 `n`：缓冲区在用之前会被清零，0 被用来表示"还没到"；如果某个源 rank 一个 token 都不发，计数也必须是非零才能让接收端停止等待。

Hybrid-EP 用"期望值递增"而不是"用完清零"：每次调用后 `update_expected_value_kernel` 把期望值加一个增量（`hybrid_ep_backend.cuh`），标志永远单调增加，省掉每轮清零和清零后的同步。

### 决定 3：缓冲区按什么上界分配

| 实现 | 上界 | 规模（量级） |
|---|---|---|
| DeepEP LL | 每 (本地专家, 源 rank) 预留 `num_max_dispatch_tokens_per_rank` 个槽位；发送/接收/信号各两份（奇偶双缓冲） | DeepSeek-V3 量级（hidden 7168、256 专家、每 rank 128 token）约 1.8 GB/rank（按 `csrc/config.hpp` 的 `LowLatencyLayout` 公式算出，见 [02](02-deepep-low-latency.md)） |
| Hybrid-EP | 假设所有 token 都路由到同一个 rank：`max_tokens_per_rank × ranks_per_node × num_nodes` 行 | 文档给出 EP64、4096 token/rank、BF16 时约 4 GB/rank（`third_party/DeepEP-hybrid-ep/docs/Hybrid-EP_Implementation.md` §4.2） |

两者都选择**最坏情况预分配**。DeepEP 的 README 也建议"自己实现时考虑按最大容量分配固定缓冲，比队列更简单、更快"（`third_party/DeepEP-hybrid-ep/README.md` Notices 一节）。代价是显存，好处是没有队列管理、没有死锁、形状固定可被 CUDA Graph 捕获。

### 决定 4：直连还是分层

```text
直连（DeepEP LL）               分层（V1 normal、Hybrid-EP）
 src ──RDMA/NVLink──> dst         src ──RDMA──> 远端节点同号 GPU ──NVLink──> dst
 每个 (token, 专家) 一份           每个 (token, 远端节点) 跨网一份，节点内再扇出
 延迟：1 跳                        延迟：2 跳 + 中转等待
 跨网字节：N·k·d                   跨网字节：≈ N·(选中的远端节点数)·d
```

- decode 的 token 少（每 rank 几十到一百多），总字节不大，**跳数**决定延迟，所以选直连。
- prefill/训练的 token 多（每 rank 数千），网卡带宽是瓶颈，**跨网字节**决定时间，所以选分层：同一个 token 去往同一节点的多个 GPU 时只过一次网卡；而且只发给同号 GPU，流量天然落在同一个 rail 上（rail-optimized 组网友好）。

### 决定 5：用多少 SM、怎么用

| 实现 | 搬数据的方式 | SM 占用 |
|---|---|---|
| DeepEP LL | RDMA 部分由网卡搬；SM 只负责 FP8 量化、写 WQE、敲门铃；接收阶段由 SM 拷贝整理 | launch 时 `num_sms = ceil(num_experts / num_warp_groups)`，几乎占满整卡，但只占很短时间；有 recv hook 时发送和接收之间不占 SM |
| V1 normal | SM 上的 warp 用 LD/ST 搬 NVLink 数据 | 典型 20~24 个 SM（`Buffer.set_num_sms(24)`，vLLM 中 `num_sms = 20`） |
| Hybrid-EP | TMA（`cp.async.bulk`）搬 SMEM↔GMEM，warp 只发指令 | 4~16 个 SM 即可接近跑满网卡（`third_party/DeepEP-hybrid-ep/docs/README_Hybrid-EP.md` 性能表） |

SM 占用决定了通信能不能和计算重叠：通信 kernel 占的 SM 越少，同时运行的 GEMM 能用的就越多。

## 源码导读

| 决定 | 看哪里 |
|---|---|
| 槽位（发送端原子分配） | `third_party/DeepEP-hybrid-ep/csrc/kernels/internode_ll.cu`：`dispatch` kernel 的 "Issue IBGDA sends" 段 |
| 槽位（全局确定下标） | `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh`：`scan` kernel；`executor/executor.cu`：`allgather_routing_map` |
| 信号（同 QP 按序） | `third_party/DeepEP-hybrid-ep/csrc/kernels/ibgda_device.cuh`：`nvshmemi_ibgda_put_nbi_warp`、`nvshmemi_ibgda_amo_nonfetch_add`（都按 `qp_id` 取 `ibgda_get_rc`） |
| 缓冲区布局 | `third_party/DeepEP-hybrid-ep/csrc/config.hpp`：`struct LowLatencyLayout`；Hybrid-EP 见 `docs/Hybrid-EP_Implementation.md` §4 |
| SM/warp 布局 | `hybrid_ep_backend.cuh` 中 `dispatch_kernel`、`combine_kernel` 前的注释 |

## 性能要点与常见坑

- **QP 数量要和本地专家数对齐**（LL）：每个本地专家用一个 QP，`EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts)`；README 示例也要求 `num_qps_per_rank = num_experts // group.size()`。QP 少了会断言失败，多个 warp 抢同一个 QP 的门铃也会变慢。
- **"标志可见 ⇒ 数据可见"依赖 RC 按序**：换成 UD、多 QP 喷洒，或把数据和标志放在不同 QP 上，这个推理就不成立了，要么加 fence，要么改用 WRITE_WITH_IMM 看 CQE。
- **最坏情况缓冲很大**：LL 下 `num_max_dispatch_tokens_per_rank` 每翻一倍，接收缓冲也翻一倍；README 建议它小于 256。

## 面试自测

1. **问**：设计 EP 通信库时，为什么几乎都用 RDMA WRITE 而不是 SEND/RECV？
   **要点**：SEND 的落点由接收端 RQ 的 FIFO 顺序决定，需要预贴 WQE、按序消费，难以和 GPU 直驱、批量小消息、固定布局结合；WRITE 由发送端指定地址，配合预分配缓冲 + 同 QP 后续标志，就能让接收端只轮询内存。
2. **问**：直连和分层各在什么场景占优？用一个式子说明。
   **要点**：直连跨网字节约 $N k d$、1 跳；分层约 $N \bar{m} d$（$\bar{m}$ 为每个 token 选中的远端节点数，$\bar m \le k$）、2 跳。decode 时 $N$ 小，固定延迟 α 主导，选直连；prefill 时 $N$ 大，带宽项主导，选分层。
3. **问**：标志用"清零后写入"和"期望值递增"各有什么问题？
   **要点**：清零要在下一轮开始前完成并同步，否则会把旧值当成新到达（LL 用奇偶双缓冲、在本轮里清下一轮的缓冲来解决）；期望值递增不需要清零，但要求每轮每个标志都被恰好加一次，空 chunk 也要补齐（Hybrid-EP G2S warp 结尾的 "Update residue flags"）。

## 补充：V1 normal 怎样把计数交给 CPU

`third_party/DeepEP-hybrid-ep/csrc/deep_ep.cpp` 在构造 Buffer 时用 `cudaMallocHost(..., cudaHostAllocMapped)` 分配 `moe_recv_counter`，并取得它的设备指针 `moe_recv_counter_mapped`。dispatch 前 host 把它置为 -1，`notify_dispatch` kernel（`csrc/kernels/intranode.cu` / `internode.cu`）交换完计数后直接把总数写进这块零拷贝内存，host 在 `while (true)` 里轮询它变成非负值，然后才分配 `recv_x`。这就是"normal 模式阻塞 CPU、不兼容 CUDA Graph"的具体来源。

## 待深入

- 待确认：V1 normal 跨节点 kernel 里 `kRDMASender`、`kRDMAAndNVLForwarder`、`kNVLReceivers` 等 warp 角色之间的队列协议（head/tail 指针）；确认方式：读 `third_party/DeepEP-hybrid-ep/csrc/kernels/internode.cu` 中 `enum class WarpRole` 之后的 dispatch 主体。
