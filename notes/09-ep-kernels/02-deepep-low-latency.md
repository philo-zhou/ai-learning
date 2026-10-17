# DeepEP low-latency（LL）：decode 的直连 RDMA dispatch/combine

> 层次：中层（贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP-hybrid-ep@10d4dd73（`csrc/kernels/internode_ll.cu`、`ibgda_device.cuh`、`csrc/config.hpp`、`csrc/deep_ep.cpp`、`deep_ep/buffer.py`）、third_party/vllm@02653fa2、third_party/sglang@d2a43644、third_party/nccl@3e46c089（`contrib/nccl_ep/device/ll_ep.cuh`）

## 一句话

LL 把每个 (token, top-k 专家) 变成一次 GPU 直接发起的 RDMA WRITE，写进对端"为这个专家、这个源 rank"预留的固定槽位；所有 token 发完后，再在同一个 QP 上发一个原子写告诉对端"我一共发了 n 个"。接收端只轮询这个计数，不需要 CPU 参与。整个过程形状固定，可被 CUDA Graph 捕获；配合 recv hook，发送和接收之间不占 SM。

## 在调用链中的位置

```text
vLLM  fused_moe/prepare_finalize/deepep_ll.py   self.buffer.low_latency_dispatch(...) / low_latency_combine(...)
SGLang layers/moe/token_dispatcher/deepep.py     buffer.low_latency_dispatch(...)       / buffer.low_latency_combine(...)
   │
deep_ep/buffer.py  Buffer.low_latency_dispatch / low_latency_combine        （Python 封装，返回 handle 与 hook）
   │
csrc/deep_ep.cpp   Buffer::low_latency_dispatch                             （选奇偶缓冲、选 stream、构造 launcher(phases)）
   │
csrc/kernels/internode_ll.cu  dispatch<...> / combine<...> / combine_overlappable<...>   （CUDA kernel）
   │
csrc/kernels/ibgda_device.cuh  nvshmemi_ibgda_put_nbi_warp / amo_nonfetch_add  （GPU 直接写 mlx5 WQE、敲门铃）
   │                         nvshmemi_get_p2p_ptr != 0 时改走 NVLink 直接 store
   ▼
ConnectX 网卡（RC QP，每个本地专家一个 QP）  /  NVLink
```

框架侧的配置：vLLM 的 `DeepEPLLAll2AllManager` 设置 `num_qps_per_rank = num_local_experts`、`allow_nvlink_for_low_latency_mode=True`（`third_party/vllm/vllm/distributed/device_communicators/all2all.py`）；`Buffer.__init__` 据此设置 `NVSHMEM_IB_ENABLE_IBGDA=1`、`NVSHMEM_IBGDA_NUM_RC_PER_PE`、`NVSHMEM_QP_DEPTH`（默认 1024）、`NVSHMEM_DISABLE_P2P`（`third_party/DeepEP-hybrid-ep/deep_ep/buffer.py`）。

## 核心概念

### 1. 接口与输出布局

```python
recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    x,                      # [num_tokens, hidden] BF16，num_tokens ≤ num_max_dispatch_tokens_per_rank
    topk_idx,               # [num_tokens, num_topk] int64，-1 表示不选
    num_max_dispatch_tokens_per_rank, num_experts,
    use_fp8=True, return_recv_hook=True)
# recv_x:     [num_local_experts, num_ranks * num_max, hidden]（FP8 时另带 scale，列主序以便 TMA）
# recv_count: [num_local_experts]，每个本地专家实际收到的 token 数
combined, event, hook = buffer.low_latency_combine(expert_out, topk_idx, topk_weights, handle, ...)
```

- 输出是 **"按专家分组、尾部带空洞"** 的三维张量：第二维的容量是 `num_ranks × num_max`（最坏情况：所有源 rank 的所有 token 都选了这个专家），各源 rank 发来的 token 被紧凑地拼在前面，只有前 `recv_count[e]` 行有效（见下文接收阶段）。下游 grouped GEMM（如 masked GEMM）必须用 `recv_count` 跳过无效行。vLLM 的调用点在 DBO 开启时通过 `dbo_maybe_run_recv_hook` 推迟执行 hook（`deepep_ll.py`）。
- `handle = (src_info, layout_range, num_max, hidden, num_experts)`：`src_info` 记录每个收到的 token 在源 rank 的下标，`layout_range` 记录每个 (本地专家, 源 rank) 段的起点和长度。combine 用它把结果送回原位置。

### 2. 缓冲区：四块对称内存 × 奇偶两份

`csrc/config.hpp` 的 `struct LowLatencyLayout` 在一块 NVSHMEM 对称堆上划出：

```text
[ signal 0 | signal 1 | send 0 | send 1 | recv 0 | recv 1 ]
 signal:  num_experts 个 int（dispatch 的 count 与 combine 的 flag 复用同一块）
 send:    max(num_max × dispatch_msg, num_experts × num_max × combine_msg)
 recv:    num_experts × num_max × max(dispatch_msg, combine_msg)
 dispatch_msg = 16 B 头 + max(hidden×2, hidden + hidden/128×4)      （BF16 与 FP8 取大）
 combine_msg  = hidden/128 × 4 B（LogFMT 的 min/max 元数据）+ hidden×2
```

- **奇偶双缓冲**：`Buffer::low_latency_dispatch` 每次调用执行 `layout.buffers[low_latency_buffer_idx ^= 1]` 切换（`csrc/deep_ep.cpp`）。本轮 kernel 顺手把**下一轮**要用的 signal 区清零（`next_clean`），这样不需要额外的清零 kernel 和同步。代价写在 docstring 里：同一时刻最多只能持有两次 LL 调用的结果张量。
- **规模估算**（hidden 7168、256 专家、num_max 128）：dispatch_msg 14352 B，combine_msg 14560 B，send 与 recv 各约 455 MiB，双份合计约 1.8 GiB。这就是 README 建议 `num_max_dispatch_tokens_per_rank` 小于 256 的原因。
- 从 normal 模式切到 LL 前必须调用 `clean_low_latency_buffer`，因为 LL 依赖 signal 区的初值为 0。

### 3. dispatch 发送阶段：一个 SM 处理若干 token，一个 warp 负责一个 top-k 目标

kernel 配置：`num_warp_groups = ceil(num_experts / num_device_sms)`，`num_warps_per_group = 32 / num_warp_groups`，grid 大小约等于整卡 SM 数，每个 block 1024 线程以内。

```text
for token_idx in (sm_id, sm_id + num_sms, ...):           # 每个 SM 轮流处理 token
    所有发送 warp 一起：读 BF16 → 每 128 通道求 amax → 量化成 FP8 E4M3，scale 写在消息尾部
    消息格式：int4 头{src_token_idx, …} | 量化后的 hidden | scales
    warp w（w < num_topk）：dst_expert = topk_idx[token][w]
        slot = atomicAdd(atomic_counter_per_expert[dst_expert], 1)       # 本地分配槽位
        dst_ptr = recv[dst_local_expert][my_rank][slot]                    # 对端地址按公式算出
        if 对端可经 NVLink 访问:  warp 内 int4 拷贝直接 store 过去
        else:                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, msg, bytes, dst_rank, qp=dst_local_expert)
        atomic_finish_counter_per_expert[dst_expert] += 1
最后一个 warp：统计本 SM 负责的那几个专家一共要发多少 token（遍历整个 topk_idx）
```

几个要点：

- **不去重**：如果一个 token 的两个 top-k 专家在同一个 rank 上，它会被发两次，分别落到两个专家的分区里。代价是多一份带宽，好处是每个槽位只属于一个专家，接收端不用再拆分。decode token 少，这个代价可以接受。
- **对端地址是算出来的**，不需要任何握手：`recv` 区在所有 rank 上布局一致（对称堆），偏移只依赖 (本地专家号, 源 rank, 槽位)。
- **"全部发完"的判定技巧**：`FINISHED_SUM_TAG = 2048`（`csrc/kernels/configs.cuh`）。计数 warp 对每个专家加 `TAG − 应发数`，SM 0 再给每个专家加一个 `TAG`，每个发送 warp 发完一条就加 1。当某个专家的计数达到 `2 × TAG` 时，说明所有该发的 token 都已发出。这样计数和发送可以并行开始，谁先谁后都不影响结果。

### 4. dispatch 计数：与数据同 QP，写 `-(n+1)`

```text
等 atomic_finish_counter_per_expert[e] == 2 × FINISHED_SUM_TAG
dst = count[e_local][my_rank]   on rank(e)
RDMA：nvshmemi_ibgda_amo_nonfetch_add(dst, -n-1, rank(e), qp=e_local)   # 与数据同一个 QP
NVLink：st_release_sys_global(dst, -n-1)
```

数据 WRITE 和计数 ATOMIC 都由 `ibgda_get_rc(dst_pe, qp_id)` 选 QP，`qp_id` 都是目标本地专家号，所以落在同一个 RC QP 上。RC 在接收端按 PSN 顺序执行请求，计数被执行时，前面的数据 WRITE 已经执行完毕。这正是"数据先到、计数后到"的保证来源（可见性细节见"待深入"）。

### 5. dispatch 接收阶段：每个 warp group 负责一个 (本地专家, 源 rank)

```text
responsible = sm_id × num_warp_groups + warp_group_id  →  (local_expert, src_rank)
sub-warp 1 的 lane 0：
    while ((c = ld_acquire_sys(count[local_expert][src_rank])) == 0);   # 自旋等计数
    n = -c - 1
    begin = atomicAdd(packed_recv_count[local_expert], n)                 # 在输出里紧凑排列
    layout_range[local_expert][src_rank] = pack(n, begin)
named barrier 同步 warp group
各 sub-warp 并行：把 n 条消息的头（src_token_idx）、数据、scale 拷到 recv_x[local_expert][begin + i]
```

下图对照一个本地专家 e 的两块内存：左边是 RDMA 落点（对称堆里的 `rdma_recv_x`，按源 rank 分区、每区固定 `num_max` 个槽位），右边是整理后返回给框架的 `packed_recv_x`（各段按到达顺序紧凑拼接）。偏移公式与 `internode_ll.cu` 中发送端算 `dst_ptr`、接收端算 `rdma_recv_x_uint8` 的写法一致：

```text
rdma_recv_x, local expert e                       packed_recv_x[e]
msg idx = (e*num_ranks + src)*num_max + slot      (num_ranks*num_max rows)
┌──────────────────────────────┐                  ┌──────────────────────────┐
│ src 0 │ n0 used │  empty     │ ─┐               │ src 2 segment (n2 rows)  │ ← begin = 0
├──────────────────────────────┤  │               ├──────────────────────────┤
│ src 1 │ n1 used │  empty     │ ─┼── copy ─────► │ src 0 segment (n0 rows)  │ ← begin = n2
├──────────────────────────────┤  │               ├──────────────────────────┤
│ src 2 │ n2 used │  empty     │ ─┘               │ src 1 segment (n1 rows)  │ ← begin = n2+n0
├──────────────────────────────┤                  ├──────────────────────────┤
│ ...                          │                  │ unused tail              │ ← recv_count[e] 之后无效
└──────────────────────────────┘                  └──────────────────────────┘
rdma_recv_count[e*num_ranks + src] = -(n+1)       layout_range[e][src] = pack(n, begin)
```

图中段的先后只是一种可能的到达顺序；哪个源 rank 的计数先被看到，哪个就先抢到 `begin`。

- 输出里各源 rank 的段是**按到达顺序紧凑拼接**的（`atomicAdd` 抢起点），所以 `recv_count` 之前的行全部有效。
- FP8 的 scale 按"列主序"写出，方便下游 GEMM 用 TMA 加载；`use_ue8m0` 时把 4 个 UE8M0 打包成一个 int。
- 可选统计：`cumulative_local_expert_recv_stats` 累计每个专家收到的 token 数（给 EPLB 用），`dispatch_wait_recv_cost_stats` 累计等待每个源 rank 的时钟周期（定位慢 rank，README 提到的 DeepXTrace 就基于它）。

### 6. combine：反向直连 + 接收端加权求和

发送阶段（每个 warp group 负责一个 (目标 rank, 本地专家)）：

```text
for token in layout_range[local_expert][dst_rank]:        # dispatch 时从 dst_rank 收到的那些 token
    src_idx = src_info[token]                             # 它在 dst_rank 上的原始下标
    dst_ptr = recv[global_expert][src_idx]                # 对端按 (全局专家, 原 token) 编址
    TMA 三级流水：GMEM → SMEM →（可选 LogFMT 编码）→ RDMA 发送缓冲 或 直接经 NVLink 写到对端
    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf, bytes, dst_rank, qp=local_expert)
最后：flag[global_expert] += 1   （同一 QP 上的原子加）
```

接收阶段：

```text
等本 SM 负责的那些专家的 flag 非零 → grid sync
每个 token：一个 TMA 加载 warp 依次把 top-k 个槽位 recv[topk_idx[t][i]][t] 读进 SMEM
            若干归约 warp 读出并乘以 topk_weights[t][i] 累加 → BF16 → TMA 写回 combined_x[t]
```

- combine 的接收缓冲按 **(全局专家, 原 token 下标)** 编址，所以接收端知道"第 t 个 token 的第 i 个专家结果"在哪里，直接按 top-k 读取并归约，不需要额外的索引交换。
- **加权求和在接收端（token 的原 rank）做**，发送端不做预归约。
- **LogFMT**（`use_logfmt=True`）：每 128 通道统计对数域的 min/max，满足条件时把 BF16 编码成 10 位对数格式（每 256 bit 压成 160 bit），否则保留 BF16；元数据放在消息头部（`logfmt_encode` / `decode_and_accumulate`）。它压缩的是 combine 的 $d$。
- **zero-copy**：`get_next_low_latency_combine_buffer(handle)` 返回下一轮 combine 的 RDMA 发送缓冲视图，专家 GEMM 直接把输出写进去，combine 就省掉一次 GMEM→缓冲区的拷贝。

### 7. recv hook：发送与接收拆成两次 launch

`csrc/deep_ep.cpp` 把 kernel 包成 `launcher(phases)`：

```text
return_recv_hook=False：launcher(SEND | RECV)，在 comm stream 上跑，compute stream 等它
return_recv_hook=True ：launcher(SEND)，直接在当前 compute stream 上跑，立刻返回
                        hook = lambda: launcher(RECV)
```

```text
时间 →
compute stream:  [dispatch SEND(批 A)] [attention(批 B)] [hook(): dispatch RECV(批 A)] [expert GEMM(批 A)] ...
网卡:                 └──── RDMA 在后台搬批 A 的 token ────┘
```

- SEND 阶段只做量化、写 WQE、敲门铃，很快退出；两次 launch 之间网卡在后台搬数据，**没有任何 kernel 占着 SM 自旋等待**。这就是 README 说的"不占 SM 的通算重叠"。SGLang 的 TBO（`third_party/sglang/python/sglang/srt/batch_overlap/two_batch_overlap.py`）和 vLLM 的 DBO（`third_party/vllm/vllm/v1/worker/ubatching.py`）都用它重叠两个 micro-batch。
- 只有 SEND|RECV 同时跑时才需要 `cg::this_grid().sync()`，因此需要 cooperative launch。
- 前提是数据真的由网卡搬：开启 NVLink 旁路后，节点内目的地的数据是 SEND 阶段由 SM 拷过去的，SEND 阶段变长。

### 8. combine_overlappable：等 GEMM 一个专家算完就发一个专家

`combine_overlappable` kernel 多了 `src_signals` / `src_signal_expect_value` 参数：发送阶段对每个本地专家先 `wait_signal(src_signals + local_expert_idx, expect)`，等专家 GEMM 把这个专家的输出算完再开始发送；多个 warp group 合作发送同一个专家的 token。SGLang 在 Blackwell 上的 single-batch overlap 会传入 `overlap`、`src_signals`、`src_signal_expect_value`（`third_party/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py`）。从代码看，这个 kernel 的完成标志只实现了 NVLink 路径（RDMA 分支是 `EP_DEVICE_ASSERT(0)`），即只适用于整个 EP 组都在一个 NVLink 域内的情况。

### 9. IBGDA：GPU 线程怎样直接驱动网卡

`nvshmemi_ibgda_put_nbi_warp`（`csrc/kernels/ibgda_device.cuh`，改编自 NVSHMEM）是一个 warp 协作完成的 RDMA WRITE：

1. **查 key**：各 lane 根据本地/远端地址所在的对称堆分块，查出 lkey、rkey 和远端虚拟地址；跨分块时拆成多个 WQE（最多 32 个，一个 lane 一个）。
2. **预留 WQE 槽**：lane 0 对 QP 的 `resv_head` 做 `atomicAdd`，多个 warp 可以并发往同一个 SQ 里写。
3. **填 WQE**：每个 lane 往 SQ 环里写一个 mlx5 WQE：控制段（opcode = RDMA_WRITE、QPN、`CQ_UPDATE`）、远端地址段（raddr、rkey）、数据段（laddr、lkey、长度），共 48 B。
4. **提交**：`__threadfence()` 保证 WQE 写完；用 CAS 推进 `ready_head`，保证按预留顺序提交；每 4 条消息（`kNumRequestInBatch`）才真正敲一次门铃：更新 doorbell record（`ibgda_update_dbr`），再把控制段的前 8 字节写到 BlueFlame 寄存器（`ibgda_ring_db`）。

这就是第一节问答里 Requester 那一侧的"软件部分"：WQE、门铃、PSN 都由网卡硬件按 RC 语义处理，只是填 WQE 和敲门铃的不再是 CPU，而是 GPU 线程。原子操作用 `MLX5_OPCODE_ATOMIC_MASKED_FA` 的 32 位扩展形式（`ibgda_write_amo_add_wqe`）。

### 10. NVLink 旁路

`nvshmemi_get_p2p_ptr(ptr, rank, dst_rank)` 查 `peer_heap_base_p2p`：如果对端可经 NVLink P2P 访问，就返回映射后的地址，kernel 改用 warp 内 `st_na_global` 直接写；否则返回 0，走 IBGDA。这是 README 中 2025.06.05 那条更新（PR #173）："低延迟 kernel 尽可能使用 NVLink"。是否启用由 `allow_nvlink_for_low_latency_mode` 控制，它最终设置 `NVSHMEM_DISABLE_P2P`。

## 源码导读

| 路径 | 函数 / 结构 | 作用 |
|---|---|---|
| `third_party/DeepEP-hybrid-ep/deep_ep/buffer.py` | `Buffer.__init__`、`low_latency_dispatch`、`low_latency_combine`、`get_next_low_latency_combine_buffer`、`clean_low_latency_buffer` | Python 接口、NVSHMEM 环境变量、返回值语义 |
| `third_party/DeepEP-hybrid-ep/csrc/deep_ep.cpp` | `Buffer::low_latency_dispatch` | 奇偶缓冲切换、stream 选择、`launcher(phases)` 与 `recv_hook` |
| `third_party/DeepEP-hybrid-ep/csrc/config.hpp` | `LowLatencyBuffer`、`LowLatencyLayout` | 对称缓冲区布局与大小 |
| `third_party/DeepEP-hybrid-ep/csrc/kernels/internode_ll.cu` | `dispatch<...>`、`combine<...>`、`combine_overlappable<...>`、`clean_low_latency_buffer` | 三个核心 kernel |
| `third_party/DeepEP-hybrid-ep/csrc/kernels/ibgda_device.cuh` | `nvshmemi_ibgda_put_nbi_warp`、`nvshmemi_ibgda_amo_nonfetch_add`、`ibgda_post_send`、`nvshmemi_get_p2p_ptr` | GPU 侧组 WQE、敲门铃、NVLink 判定 |
| `third_party/DeepEP-hybrid-ep/csrc/kernels/configs.cuh` | `FINISHED_SUM_TAG`、`LOW_LATENCY_SEND_PHASE`、`LOW_LATENCY_RECV_PHASE` | 常量 |
| `third_party/DeepEP-hybrid-ep/tests/test_low_latency.py` | `test_main` | 正确性与性能测试，覆盖 hook / zero-copy / LogFMT 组合 |
| `third_party/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py` | `low_latency_dispatch` / `low_latency_combine` 调用点 | vLLM 用法 |
| `third_party/nccl/contrib/nccl_ep/device/ll_ep.cuh` | `ncclGin::put` + `ncclGin_SignalAdd` | 同一算法的 NCCL GIN 移植版 |

## 性能要点与常见坑

- **README 的参考数据**（H800 + CX7 400 Gb/s IB，128 token/batch、hidden 7168、top-8、FP8 dispatch、BF16 combine）：dispatch 从 EP8 的 77 µs 到 EP256 的 194 µs，combine 从 114 µs 到 360 µs；EP 扩大后单卡 RDMA 带宽稳定在约 39~40 GB/s。厂商自测数据，本机未复现。
- **QP 数必须 ≥ 本地专家数**，否则 kernel 断言失败；也不要让 QP 远多于专家数，门铃竞争和 QP 上下文都会变多。
- **网络隔离**：README 建议用 IB 虚拟通道（`NVSHMEM_IB_SL`）把 normal、LL 和其他流量分开；自适应路由在重负载下打开、轻负载下关掉。
- **槽位溢出**：任何 rank 的 `num_tokens` 都不能超过 `num_max_dispatch_tokens_per_rank`，否则会写穿别的源 rank 的分区。
- **结果张量复用缓冲**：返回的 `recv_x` 指向奇偶缓冲之一，第三次调用会覆盖第一次的结果。
- **未定义行为的 PTX**：为了性能，读 volatile 数据时用了 `ld.global.nc.L1::no_allocate.L2::256B`，只在 Hopper 上验证过；其他平台出错时用 `DISABLE_AGGRESSIVE_PTX_INSTRS=1` 关闭（README Notices）。

## 面试自测

1. **问**：画出 LL dispatch 的完整时序：从 `topk_idx` 到接收端 `recv_count` 可读。
   **要点**：发送 SM 量化 → 每个 top-k 一个 warp 本地 `atomicAdd` 取槽位 → GPU 写 WQE、敲门铃（或 NVLink store）→ 发送完成计数达到 `2×TAG` → 同 QP 原子写 `-(n+1)` → 接收端自旋看到非零 → `atomicAdd` 取紧凑起点 → 拷贝整理。全程没有 CPU 参与。
2. **问**：如果把数据 WRITE 放在 QP0、计数 ATOMIC 放在 QP1，会出什么问题？
   **要点**：不同 QP 之间没有顺序保证，计数可能先于数据被执行，接收端读到不完整的 token；需要在发送端等数据 WRITE 完成（轮询 CQ）后再发计数，延迟会增加一个 RTT。
3. **问**：LL combine 为什么把接收缓冲按 (全局专家, 原 token) 编址？
   **要点**：原 rank 知道每个 token 选了哪几个专家（`topk_idx`），按这个编址可以直接定位 k 个结果做加权和，不需要 dispatch 时的 `src_info` 反查；代价是接收缓冲要按 `num_experts × num_max` 预留。
4. **问**：recv hook 在什么情况下反而让性能变差？
   **要点**：没有其他可重叠的计算时，拆成两次 launch 只是多了一次 kernel 启动；启用 NVLink 旁路时，SEND 阶段的数据由 SM 拷贝，hook 节省 SM 的好处变小；另外两次 LL 调用之间不能持有超过两份结果。
5. **问**：从网卡视角看，GPU 发起的 RDMA WRITE 和 CPU 用 `ibv_post_send` 发起的有什么不同？
   **要点**：网卡看到的 WQE、门铃、RC 协议处理完全一样；区别在于谁写 WQE、谁敲门铃。IBGDA 要求 SQ、doorbell record 和 BlueFlame 寄存器能被 GPU 访问（映射到 GPU 地址空间），并且 GPU 侧要自己处理多 warp 并发写 SQ（预留槽 + 按序提交）。

## 待深入

- 待确认：RC 上 "WRITE 后 ATOMIC" 的可见性在乱序放置网卡上的保证（同 [README](README.md) 待深入第一条）。
- 待确认：vLLM 传入的 `allow_mnnvl`、`enable_shrink` 参数在本地 `hybrid-ep` 分支的 `Buffer.__init__` 中不存在，说明 vLLM 对接的是另一个 V1 版本；确认方式：在上游 `main` 的 V1 历史提交中查 `deep_ep/buffer.py`。
- 未运行：`tests/test_low_latency.py` 需要多卡 Hopper + IB，本机 not-run。
