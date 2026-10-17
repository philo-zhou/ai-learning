# Hybrid-EP：路由预计算 + 分层 RDMA/NVLink + TMA warp 专业化流水线

> 层次：中层（贯穿下层 RDMA/NVLink） · 来源：third_party/DeepEP-hybrid-ep@10d4dd73（`csrc/hybrid_ep/`、`deep_ep/hybrid_ep_buffer.py`、`docs/Hybrid-EP_Implementation.md`、`docs/README_Hybrid-EP.md`）、third_party/nccl@3e46c089（`contrib/nccl_ep/device/hybrid_ep.cuh`）

## 一句话

Hybrid-EP 是 NVIDIA 为大 batch（训练、prefill）写的 EP 通信实现：先 AllGather 一次全局路由，用 `scan` kernel 让每个 rank 算出所有 token 的确定落点；dispatch 时每个 token 跨节点只经 RDMA 发给 **peer（远端节点上 local rank 相同的 GPU）** 一次，到达后再由 TMA 经 NVLink 扇出到节点内需要它的 rank；combine 反过来，先在节点内预归约再跨节点发送。kernel 是常驻（persistent）、warp 专业化的流水线，搬数据全用 TMA，4~16 个 SM 就能接近跑满网卡。

## 在调用链中的位置

```text
训练框架（Megatron 等，本地未克隆，待确认）
   │  HybridEPBuffer.dispatch / dispatch_with_permute / combine / combine_with_unpermute
   ▼
deep_ep/hybrid_ep_buffer.py   路由整理（topk 或 routing_map）、token 槽位对齐、handle 管理
   │
csrc/hybrid_ep/hybrid_ep.cu   HybridEPBuffer（C++）：缓冲区分配与注册、JIT、调 Executor
csrc/hybrid_ep/executor/      Executor：metadata_preprocessing → dispatch/combine 三段式
csrc/hybrid_ep/jit/           按配置生成 .cu → nvcc → dlopen
   │
csrc/hybrid_ep/backend/hybrid_ep_backend.cuh   scan / dispatch_kernel / combine_kernel
   │   节点内：TMA cp.async.bulk 读写经 CUDA IPC 或 fabric（MNNVL）映射的远端缓冲
   │   节点间：DOCA GPUNetIO 在 GPU 上组 RDMA WRITE/ATOMIC WQE（或实验性的 NIXL/UCX）
   ▼
NVLink / NVSwitch（含 MNNVL）   ConnectX 网卡（每个 block 对每个远端节点一个 QP）
```

"节点"在 Hybrid-EP 里指 **NVLink 域**，不是操作系统意义上的机器：在 GB200 NVL72 这类 MNNVL 系统上，一个节点可以有 36 或 72 个 rank。`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` 可以手动指定（`docs/Hybrid-EP_Implementation.md` §9.2）。

## 核心概念

### 1. 三段式执行：预处理 → 核心 kernel → 后处理

| API | 预处理 | 核心 | 后处理 |
|---|---|---|---|
| `metadata_preprocessing` | — | AllGather 路由 + `scan` kernel | — |
| `dispatch` | 多节点时把输入 D2D 拷进已注册的 RDMA 缓冲 | `dispatch_kernel` | 从缓冲 D2D 拷出给 PyTorch |
| `dispatch_with_permute` | 同上 | `dispatch_kernel` | permute kernel（或融合进 dispatch） |
| `combine` | 把专家输出 D2D 拷进缓冲 | `combine_kernel` | — |
| `combine_with_unpermute` | unpermute kernel（或融合进 combine） | `combine_kernel` | — |

（摘自 `docs/Hybrid-EP_Implementation.md` §5，与 `executor/executor.cu` 中的 `cudaMemcpyAsync` 调用一致。）

### 2. 路由预计算：一次 AllGather 换来确定的落点

`Executor::allgather_routing_map`（`executor/executor.cu`）把每个 rank 的路由（`routing_map` 布尔矩阵或 `topk_idx`）AllGather 到所有 rank。多节点时用 `torch.distributed.all_gather_into_tensor`（走 NCCL）；单节点且开启 `enable_custom_allgather` 时用自研 allgather。

随后 `scan` kernel（`backend/hybrid_ep_backend.cuh`）对全局路由做前缀和，生成三张表，这就是 dispatch handle 的前几项：

| 表 | 形状（概念） | 含义 | 谁用 |
|---|---|---|---|
| `attn_to_rdma_map` | [本 rank 的 token, 远端节点数] | 我的这个 token 是否需要发往某个远端节点（那个节点上至少有一个它选中的专家） | RDMA warp：决定发哪些 token |
| `rdma_to_attn_map` | [token, 节点] | 某个节点（含本节点）是否需要某个源 token | G2S warp：决定加载哪些 token |
| `sparse_to_dense_map` | [源 token, 本节点 rank 数] | 这个 token 在本节点每个 rank 的输出缓冲里的下标（-1 表示不需要） | S2G warp：决定写到哪个 rank 的哪一行 |

因为所有 rank 用同一份全局路由算出同样的结果，**kernel 里不需要任何原子分配槽位，也不需要先交换计数**。这正是 [01](01-design-space.md) 决定 1 的第三种做法。

代价有两个：

- 每次路由变化都要多一次 AllGather 和 `scan`；路由表很小（token 数 × 专家数个 bool），远小于 token 数据本身。
- 想知道"本 rank 实际收到多少 token"就要把计数同步回 host（默认 blocking 模式）。CUDA Graph 下用 `non_blocking=True`，由用户给出估计的 `num_permuted_tokens`，超出时丢弃多余 token 并置 `overflow_flag`（§7.2）。

### 3. dispatch kernel：每个 block 4 个 warp，三类角色

`dispatch_kernel` 前的注释（`hybrid_ep_backend.cuh`）给出了布局：

```text
每个 CUDA block（常驻一个 SM）：
  ├─ RDMA N2N warp group（1 warp，仅多节点）   N2N_warp_group_device_function
  ├─ 节点内 G2S warp group（1 warp）           G2S_warp_group_device_function
  └─ 节点内 S2G warp group（多节点 2 warp / 单节点 3 warp）  S2G_warp_group_device_function
token 按 chunk（默认 64 个 token，NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API）切分，chunk 在 block 间轮转分配
```

**RDMA warp（发往远端 peer）**，对每个 chunk、每个远端节点：

1. 把这个 chunk 的 `attn_to_rdma_map` 读进 SMEM，warp 内 `__reduce_add_sync` 算出要发几个 token。
2. lane 0 用 `doca_gpu_dev_verbs_reserve_wq_slots` 一次性预留 `token 数 × 每 token WQE 数 + 1` 个 WQE 槽（每个 token 一个 WRITE 写数据；前向时再加一个写概率向量；FP8 时再加一个写 scale）。
3. 32 个 lane 用 `__ballot_sync` + `__popc` 给需要发送的 token 分配连续的 WQE 下标，各自填 `RDMA_WRITE` WQE。**远端地址 = peer 缓冲基址 + 源 token 下标 × hidden**：每个源 token 在 peer 上有固定位置，不紧凑排列。
4. 最后由 lane 0 追加一个 `ATOMIC_FA`，把 peer 上"(源节点, chunk)"对应的 64 位标志加 1，然后 `mark_wqes_ready` + `submit_db` 敲门铃。
5. 所有 chunk 发完后，轮询 CQ 等这些 WQE 完成。

这里和 LL 用的是同一个技巧：**数据 WRITE 与标志 ATOMIC 在同一个 QP 上、标志排在最后**（见 [01](01-design-space.md) 决定 2）。

**G2S warp（把要处理的 token 读进 SMEM）**：只有一个线程工作。对每个 chunk，按"本节点、本节点−1、本节点−2……"的顺序轮转各个源节点：

- 源节点是本节点：数据就在本 rank 的输入缓冲里，直接读；
- 源节点是远端：先用 `ld.relaxed.sys.global` 自旋等标志达到 `expected_flag_value`，再读 RDMA 缓冲里对应的 chunk。

只加载 `rdma_to_attn_map` 标记为"本节点需要"的 token：用 `cp.async.bulk` 把 token、概率、scale 搬进 SMEM 环形队列（`NUM_OF_STAGES_DISPATCH_API` 级），通过 mbarrier 与消费者交接。

**S2G warp（经 NVLink 扇出）**：按 `sparse_to_dense_map`，把 SMEM 里的每个 token 用 TMA 写到节点内每个需要它的 rank 的 `expert_output_token[下标]`（远端缓冲通过 CUDA IPC 或 fabric handle 映射）。一个 token 被同节点多个 rank 需要时，在 SMEM 里只有一份，向多个 rank 各写一次。

下图是一个 dispatch block 内部三类 warp 的数据流（多节点）：RDMA warp 自成一路往外发；G2S 与 S2G 通过 SMEM 环形队列做生产者-消费者，每级一对 mbarrier（源码注释："1st for producer->consumer, 2nd for consumer->producer"）。

```text
                  ┌──────────────── one dispatch block (1 SM, 4 warps) ──────────────────┐
 input buffer ────┼─► RDMA warp ── WRITE x tokens + ATOMIC_FA ───────────────────────────┼──► peer rdma buffer
 (registered)     │                                                                      │    (remote node)
                  │                                                                      │
 input buffer ────┼─► G2S warp ─cp.async.bulk─► SMEM ring ─────► S2G warps ──TMA store───┼──► expert_output_token[r]
 rdma buffer  ────┼─►  (spin on flag)          NUM_OF_STAGES     (2 multi-node)          │    (NVLink, IPC/fabric)
 (from peers)     │                            full/empty mbar                           │
                  └──────────────────────────────────────────────────────────────────────┘
```

- 左侧"rdma buffer"即参数里的 `rdma_inter_node_group_token`，G2S 等的标志是 `rdma_inter_node_group_flags`（按源节点和 chunk 编号定位）；右侧 `expert_output_token[r]` 是节点内各 rank 的输出缓冲指针数组（`hybrid_ep_backend.cuh` 中 `dispatch_kernel_param_t`）。
- 同一个 block 既是发送方（RDMA warp 把本 rank 的 chunk 发给远端 peer），也是接收方（G2S 消费远端 peer 发来的 chunk）。dispatch 的共享内存结构里，mbarrier 只用于 token 环形队列、`sparse_to_dense_map` 的乒乓缓冲和 S2G 组内同步，RDMA warp 与 G2S/S2G 之间没有 SMEM 交接。

```text
节点 A 的 rank 0                         节点 B（peer = rank 0）                节点 B 内其他 rank
 token t ──RDMA WRITE（只发一次）──> B.rank0.rdma_buffer[t] ──TMA/NVLink──> B.rank{1,3,…}.expert_output[dense_idx]
                   └─ ATOMIC_FA: flag[A][chunk] += 1     G2S 等 flag 到期
```

### 4. combine kernel：节点内预归约 → 跨节点 → 最终归约

`combine_kernel` 前的注释给出了每个 block 的五类 warp group（多节点共 11 个 warp，单节点 6 个）：

| warp group | 作用（多节点） |
|---|---|
| 节点内 G2S（1 warp） | 读本节点各 rank 的专家输出（这些输出要发往某个远端节点） |
| 节点内归约（4 warp） | 对"同一个远端 token"在本节点内的多个专家结果做 FP32 累加（**预归约**），结果写进 RDMA 发送缓冲 |
| RDMA N2N（1 warp） | 等某个 chunk 预归约完（SMEM mbarrier），把它 RDMA WRITE 到 token 原节点的 peer，末尾 ATOMIC 标志 |
| 节点间 G2S（1 warp） | 读本节点直接算出的结果和远端 peer 发来的预归约结果 |
| 节点间归约（4 warp） | 做最终归约，经 NVLink 写回 token 原来所在的 rank |

下图按一个跨节点 token 的路径把五类 warp group 串起来（多节点）。每个 block 两条流水都有；对某一个 token 而言，上一行发生在专家所在的节点 B，下一行发生在 token 原节点 A 上的 peer GPU：

```text
node B (experts)
                ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
 expert outputs │ intra G2S   │    │ intra red    │    │ RDMA N2N     │  RDMA WRITE
 of B's ranks ─►│ (1 warp)    ├───►│ (4 warps)    ├───►│ (1 warp)     ├──+ ATOMIC ──┐
                └─────────────┘SMEM└──────────────┘    └──────────────┘             │
                                    FP32 pre-reduce     waits per-chunk mbarrier    │
                                    → rdma send buf                                 │
node A (token home, peer GPU)                                                       │
                ┌─────────────┐    ┌──────────────┐                                 │
 local results ►│ inter G2S   │    │ inter red    │  TMA / NVLink                   │
 rdma buffer ──►│ (1 warp)    ├───►│ (4 warps)    ├──────────────► home rank of t   │
   ▲            └─────────────┘SMEM└──────────────┘                                 │
   └────────────────────────────────────────────────────────────────────────────────┘
```

单节点时没有上一行（intra G2S / intra red / RDMA 都不启用），只剩 inter G2S（2 warp）与 inter red（4 warp，两条流水），共 6 个 warp（`combine_kernel` 前的注释）。

- 节点内归约 warp 按"同一个 chunk 号发往各个远端节点，再下一个 chunk 号"的顺序产出，RDMA warp 按相同顺序消费。注释给出的理由是：最终归约需要所有节点的同一个 chunk 号，这样排序能让它们尽早凑齐。
- 预归约把跨节点 combine 的流量从"每个 (token, 远端专家) 一份"降到"每个 (token, 远端节点) 一份"，和 dispatch 的去重对称。
- combine 只支持 BF16，累加在 FP32 中进行。反向时（`BACKWARD_COMBINE`）概率向量也一起归约。

### 5. 每个 rank 的缓冲区

最坏情况预分配，假设所有 token 都路由到同一个 rank（`docs/Hybrid-EP_Implementation.md` §4.2）：

```text
max_tokens = max_tokens_per_rank × ranks_per_node × num_nodes
节点内：token_buffer = max_tokens × hidden × dtype；prob_buffer = max_tokens × (experts_per_rank × ranks_per_node) × 4 B
节点间：rdma_token_buffer = max_tokens_per_rank × (num_nodes − 1) × hidden × dtype（每个远端节点一份，按源 token 下标定位）
```

- 文档的例子：EP64、256 专家、BF16、hidden 7168、4096 token/rank，纯 MNNVL（64×1）约 3.75 GB/rank，RDMA（8×8）约 4 GB/rank。
- `use_shared_buffer=True`（默认）时 dispatch 与 combine 共用节点内缓冲，显存减半，代价是两次 Hybrid-EP 操作不能重叠。
- 缓冲注册（RDMA MR 注册、IPC/fabric handle 导出、`all_gather` 交换、打开对端 handle）很慢，只在初始化或扩容时做；`max_num_of_tokens_per_rank` 要一次给够。

### 6. 同步原语：期望值递增，不清零

- RDMA 标志：每次 dispatch/combine 后 `update_expected_value_kernel` 把 `expected_rdma_flag_value` 加一个增量，发送端每次都对标志做 `ATOMIC_FA +1`。标志单调增加，不需要清零。某些 chunk 本轮没有数据要发时，G2S warp 在末尾直接把这些"残余标志"写成期望值（"Update residue flags"），保证下一轮的期望值仍然对得上。
- NVLink 域内屏障：`device_sync_kernel` 对域内 rank 0 上的一个 u32 做系统域原子加，再轮询到期望值；用奇偶位区分相邻两轮。
- 跨节点屏障：`rdma_sync_kernel`。
- permute 融合时，dispatch 的 S2G warp 写完一个 chunk 就置 `intra_node_expert_output_chunk_flags`，permute block 等标志后开始重排，同样用单调递增的 `expected_permute_flag_value`（§8.1）。

### 7. permute 融合与非阻塞模式

- 注册缓冲不归 PyTorch 缓存分配器管理，dispatch 结果本来就要拷一次到普通张量。`dispatch_with_permute` 把这次拷贝和"按专家重排 + `pad_multiple` 对齐"合在一起，输出可直接喂给 grouped GEMM；`combine_with_unpermute` 做相反的事。
- `fuse_permute_dispatch=True` 时，kernel grid 在 dispatch block 后面追加若干 permute block（block 专业化），按 chunk 标志流水执行，省掉一次独立 kernel 和一遍 GMEM 往返。这需要编译宏 `HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE`，融合版和非融合版编成两个 `.so`。

### 8. 节点间后端：DOCA GPUNetIO 或 NIXL

- 默认 DOCA 路径：GPU 侧通过 `doca_gpu_dev_verbs_*` 直接操作 QP（预留 WQE、填 WRITE/ATOMIC、敲门铃、轮询 CQ），与 IBGDA 同类，都是 GPU 直驱网卡。QP 创建时沿用 NCCL 的 `NCCL_IB_GID_INDEX`、`NCCL_IB_ADDR_FAMILY`、`NCCL_IB_TC`（RoCE 配置），GPU 到网卡的映射可以自动探测，也可以用 `HYBRID_EP_NIC_MAPPING` 指定（§4.1）。
- NIXL 路径（实验性，`USE_NIXL=1`）：经 UCX 的 GPU Direct Async 通道发送，元数据靠 etcd 交换；数据 put 带 `defer` 标志，最后的原子信号 `NODELAY`，一次门铃刷出（`hybrid_ep_backend.cuh` 中 NIXL 版 `N2N_warp_group_device_function` 的注释）。官方说明性能尚未与 DOCA 持平。

### 9. JIT

`HybridEpConfigInstance`（hidden、每 rank 最大 token、每 rank 专家数、节点数、每节点 rank 数、流水级数、block 数等）作为模板参数生成 `.cu`，调用 `nvcc` 编成 `.so`，再 `dlopen`。缓存在 `HYBRID_EP_CACHE_DIR`（默认 `$HOME/.deepep/hybrid_ep/jit`）。所有 rank 编译同样的 kernel，先写带 rank 和时间戳的临时文件，再原子 `rename`（§6）。

## 源码导读

| 路径 | 函数 / 结构 | 作用 |
|---|---|---|
| `third_party/DeepEP-hybrid-ep/deep_ep/hybrid_ep_buffer.py` | `HybridEPBuffer.dispatch`、`dispatch_with_permute`、`combine`、`combine_with_unpermute` | Python 接口、token 槽位对齐（`_resolve_token_slots`）、路由填充 |
| `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/executor/executor.cu` | `Executor::allgather_routing_map`、metadata preprocessing、dispatch/combine 前后处理 | 三段式编排 |
| `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` | `scan`、`dispatch_kernel`、`combine_kernel`、`N2N_warp_group_device_function`、`G2S_warp_group_device_function`、`S2G_warp_group_device_function`、`intra_node_red_warp_group_device_function`、`inter_node_red_warp_group_device_function`、`device_sync_kernel`、`update_expected_value_kernel` | 全部 kernel |
| `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/buffer/internode_doca.cu`、`intranode.cu` | RDMACoordinator / NVLCoordinator | 缓冲注册、QP 建立、handle 交换 |
| `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/allocator/allocator.cu` | fabric（`cuMemCreate`/`cuMemExportToShareableHandle`）与 IPC 两种分配 | MNNVL 支持、NVLink 域探测 |
| `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/config.cuh` | `BufferConfig`、`HybridEpConfigInstance` | 配置与 JIT 模板参数 |
| `third_party/DeepEP-hybrid-ep/tests/test_hybrid_ep.py`、`test_graphed_hybrid_ep.py` | — | 功能、CUDA Graph 测试 |
| `third_party/nccl/contrib/nccl_ep/device/hybrid_ep.cuh` | — | 同一设计在 NCCL GIN 上的移植（NCCL EP 的 HT 模式） |

## 性能要点与常见坑

- **参考数据**（`docs/README_Hybrid-EP.md`，厂商自测，algbw）：
  - H100、EP16、8 个 SM：FP8 dispatch 约 45 GB/s（kernel-only 62），BF16 dispatch 约 58，combine 约 57；对照 DeepEP V1 normal 在 H800 EP16 约 43 GB/s RDMA 瓶颈带宽（不同平台，只能粗看量级）。
  - B200 单节点 8 卡、16 个 SM：Hybrid-EP BF16 dispatch 536 GB/s，DeepEP 同 SM 数为 348 GB/s，DeepEP 要 48 个 SM 才到 646 GB/s。
  - 这类数字取决于路由分布（测试用随机均匀路由）、SM 数和平台，本机均未复现。
- **chunk 大小**：默认 64 token；不用融合路径时可以试 128（三个 `NUM_OF_TOKENS_PER_CHUNK_*` 环境变量要一起改）。融合模式要求三者相等。
- **hidden 很大时**减少流水级数（`NUM_OF_STAGES_DISPATCH_API` 等），否则 SMEM 放不下。
- **概率向量用 TMA 搬**，要求 16 B 对齐，即每个节点的专家数（`experts_per_rank × ranks_per_node`）是 4 的倍数（kernel 内静态断言的注释）。
- **非阻塞模式会丢 token**：`num_permuted_tokens` 估小了会丢弃溢出部分，必须检查 `overflow_flag`（handle 的最后一项）。
- **RDMA 拓扑探测依赖 NVML**：容器构建时用 stub，运行时必须注入真实的 `libnvidia-ml.so.1`。

## 面试自测

1. **问**：Hybrid-EP 为什么要先 AllGather 路由？不做行不行？
   **要点**：做了之后每个 rank 都能算出每个 token 在每个目标 rank 上的确定下标（`sparse_to_dense_map`），kernel 里就不需要原子分配槽位、不需要先交换计数，S2G 可以直接 TMA 写到最终位置，接收端也不需要整理。不做就退回 V1 normal 的"notify 交换计数 + 队列"方案，多一轮同步，还要管理队列的 head/tail。
2. **问**：dispatch 跨节点为什么只发给 peer（同 local rank 的 GPU）？
   **要点**：(1) 同一个 token 去往一个节点的多个 rank 时只过一次网卡；(2) 在 rail-optimized 组网中同号 GPU 的网卡接在同一个 rail 上，跨节点流量不需要穿过 spine 换 rail；(3) 节点内扇出交给带宽高一个数量级的 NVLink。
3. **问**：combine 的节点内预归约省了什么，代价是什么？
   **要点**：跨节点流量从每个 (token, 专家) 一份降到每个 (token, 节点) 一份；代价是多一级流水（节点内 G2S → 归约 → RDMA），以及归约顺序变化带来的数值差异（BF16 输入、FP32 累加，仍有微小差别）。
4. **问**：为什么说 Hybrid-EP 省 SM？省在哪里？
   **要点**：数据搬运全部由 TMA（`cp.async.bulk`）异步完成，一个线程就能发起整块拷贝；warp 专业化后每个角色只需 1~4 个 warp，一个 block 只有 4 个（dispatch）或 11 个（combine）warp；RDMA 由网卡搬。相比之下 V1 normal 用 warp 的 LD/ST 一个 int4 一个 int4 地搬，需要多得多的线程。
5. **问**：标志为什么设计成"每轮期望值递增"？
   **要点**：省掉清零和清零后的屏障；发送端每轮固定做一次 `ATOMIC_FA +1`，接收端只比较是否等于本轮期望值。前提是每个标志每轮都恰好加一次，所以空 chunk 要由本地补齐。

## 待深入

- 待确认：`scan` kernel 中 `sparse_to_dense_map` 的前缀和在多个 block 间如何拼接（`tmp_state_t` 的 decoupled look-back 或两遍扫描）；确认方式：读 `hybrid_ep_backend.cuh` 中 `scan` 的完整实现。
- 待确认：DOCA GPUNetIO 的 `doca_gpu_dev_verbs_submit_db` 与 IBGDA 门铃写法的差异（是否用 BlueFlame、是否批量）；确认方式：DOCA SDK 文档或 `csrc/hybrid_ep/backend/ibvcore.h`。
- 待确认：Megatron-LM 如何接入 `HybridEPBuffer`；确认方式：克隆 Megatron-LM 后 grep。
- 未运行：`tests/test_hybrid_ep.py` 需要 Hopper/Blackwell 多卡 + IB/RoCE（多节点还需 DOCA 或 NIXL），本机 not-run。
