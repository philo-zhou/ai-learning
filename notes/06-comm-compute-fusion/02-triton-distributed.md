# Triton-distributed 导读

> 层次：跨层 · 来源：third_party/Triton-distributed@7908e4ea（ByteDance Seed 开源）、ai-infra-book 第 5.5.4 节
>
> 重要提示：以下所有可运行命令都需要 NVIDIA/AMD GPU 与 NVSHMEM，本机（macOS）不可运行，本文只做源码阅读。

## 一句话

Triton-distributed 是基于 OpenAI Triton 的"分布式编译器"：在 Triton 里加入少量跨 rank 原语（`wait` / `notify` / `symm_at` 等）并直接暴露 NVSHMEM device API，让开发者在同一个 Triton kernel 里写"算 tile + 发通知 / 等通知"，从而用 Python 写出接近手写 CUDA 库水平的通算融合算子。README 称其可与 Distributed-GEMM、FLUX 对标。

## 在调用链中的位置

- 上层：`triton_dist/layers/nvidia/` 里的 `TP_MLP`、`TP_Attn`、EP MoE 等层，被 `triton_dist/models/` 的 Qwen3 等模型和 E2E Engine 调用（见 `third_party/Triton-distributed/docs/getting-started/e2e/e2e_dense.md`）。
- 中层：`triton_dist/kernels/nvidia/` 里的融合算子（`ag_gemm`、`gemm_rs`、EP all2all 等）。
- 下层：NVSHMEM（对称堆、put/get/signal，见 [03](03-nvshmem-and-device-initiated.md)）、CUDA copy engine、CUDA stream。

## 核心概念

### 1. 仓库结构（实际目录，commit 7908e4ea）

| 路径（均在 `third_party/Triton-distributed/` 下） | 作用 |
|---|---|
| `python/triton_dist/language/` | 分布式语言扩展：`distributed_ops.py`（`wait`、`consume_token`、`rank`、`num_ranks`、`symm_at`、`notify`）、`extra/`（`libshmem_device.py` 及各后端：`cuda/libnvshmem_device.py`、`hip/librocshmem_device.py`、`hip/libmori_shmem_device.py`、`maca/libmxshmem_device.py`） |
| `python/triton_dist/kernels/nvidia/` | NVIDIA 平台融合与通信算子 |
| `python/triton_dist/kernels/amd/`、`metax/` | AMD、沐曦平台对应算子 |
| `python/triton_dist/layers/nvidia/` | 把算子包成可替换的层（TP/EP/SP/PP） |
| `python/triton_dist/models/` | 模型与推理 Engine |
| `python/triton_dist/mega_triton_kernel/` | MegaTritonKernel：整个模型层融合为一个 persistent kernel |
| `python/triton_dist/test/nvidia/` | 每个算子的正确性与性能测试 |
| `python/triton_dist/benchmark/` | `bench_allgather_gemm.py`、`bench_tp_mlp.py` 等 |
| `python/triton_dist/autotuner.py`、`tune.py` | 自动调优（`triton_dist.tune.autotune`） |
| `python/triton_dist/utils.py` | NVSHMEM 初始化与张量分配（`init_nvshmem_by_torch_process_group`、`nvshmem_create_tensor(s)`、`nvshmem_barrier_all_on_stream`、`initialize_distributed`） |
| `python/triton_dist/profiler_utils.py` | `perf_func`、`group_profile` |
| `lib/`、`include/TritonDistributed/`、`plugin/` | 编译器侧（dialect、转换、Triton 3.7 out-of-tree 插件）。README 称默认分支以插件形式对着上游 Triton 3.7 构建 |
| `shmem/nvshmem_bind/`、`rocshmem_bind/`、`mxshmem_bind/` | 各 SHMEM 库的绑定 |
| `FlashComm/` | 独立的 CUDA 通信库（EP overlap、TMA 等），按 README 新闻属于较新增量，本文不展开 |
| `tutorials/` | 10 个教程（见下） |
| `docs/` | `primitives.md`、`testing.md`、`build.md`、`getting-started/` 等 |

### 2. 原语（以 `docs/primitives.md` 与 `distributed_ops.py` 实际代码为准）

**低级原语（`triton_dist.language`，别名常写作 `dl`）**

- `rank(axis)`、`num_ranks(axis)`：上下文查询。
- `symm_at(ptr, rank)`：把本地对称内存指针映射为对端 rank 上对应地址，之后可直接 `tl.load/tl.store` 访问对端（NVLink 域内）。
- `wait(barrierPtrs, numBarriers, scope, semantic, waitValue=1)`：在一组信号地址上自旋等待，返回一个 token。`scope` 取 `gpu/cta/sys`，`semantic` 取 `relaxed/acquire/release/acq_rel`（见 `distributed_ops.py` 中的映射表）。
- `consume_token(value, token)`：把 token "挂"到某个值（如指针或 TMA descriptor）上，使后续使用该值的加载在数据依赖上排在 wait 之后，防止编译器把加载提前到 wait 前。
- `notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node")`：向某 rank 的信号地址写入/累加信号；`sig_op` 为 `set` 或 `add`，`comm_scope` 为 `gpu/intra_node/inter_node`。

**NVSHMEM device 原语（`triton_dist.language.extra.libshmem_device`）**：`my_pe`、`n_pes`、`putmem_*`（block/warp/nbi 变体）、`putmem_signal_*`、`getmem_*`、`signal_op`、`signal_wait_until`、`fence`、`quiet`、`barrier_all*`、`sync_all*`、`remote_ptr` 等，完整清单在 `third_party/Triton-distributed/docs/primitives.md`。

**高级（tile-centric）原语**：`docs/primitives.md` 写明"高级原语将在 MLSys 2025 之后发布"，对应论文 TileLink（README 新闻）。待确认：本 commit 中高级原语是否已落地；确认方式：在 `python/triton_dist/language/` 下搜索论文中的 tile 级通信接口名。

### 3. 典型编程模式（生产者-消费者 + 信号）

```text
通信侧（copy engine / 通信 kernel / 远端 put）  --写数据-->  对称内存
                                               --置信号-->  signal[分片]
GEMM 侧（消费者 kernel）: 每个 tile 开始前 dl.wait(signal[所需分片]) + consume_token
```

调度上，GEMM 侧按"本地分片优先、其他 rank 按到达顺序"重排 tile（`swizzle`），这样前几个 tile 无需等待。

## 源码导读（路径 + 函数 + 作用）

### AllGather + GEMM

文件：`third_party/Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py`

- `AllGatherGEMMTensorParallelContext`：持有对称内存的 AG 缓冲（`symm_workspaces`）、信号（`symm_barriers`）、两条异步 stream（`ag_intranode_stream`、`ag_internode_stream`）、`max_gemm_sm`、`all_gather_method`。
- `create_ag_gemm_context`、`ag_gemm`（带 `triton_dist.tune.autotune`）：对外入口。
- `rowise_ag_gemm_dispatcher`：主流程。让通信 stream `wait_stream(current_stream)`；调用 `cp_engine_producer_all_gather_intra_node` / `..._inter_node` 在通信 stream 上启动 AllGather（用 copy engine 与 NVSHMEM）；随后在主 stream 启动 GEMM kernel；最后 `current_stream.wait_stream(...)` 合流。Hopper 及以上（`get_device_capability()[0] >= 9`）走 persistent 版本，并把 GEMM 使用的 SM 数设为总 SM 数减去 `n_nodes - 1`（留给跨节点通信）。
- `kernel_consumer_gemm_persistent` / `kernel_consumer_gemm_non_persistent`：消费者 GEMM。核心几行是：对 tile 做 `pid_m` 的 rank swizzle（`nnodes == 1` 时按 `rank` 循环移位；多节点用 `threadblock_swizzle_allgather_gemm_kernel`），算出该 tile 的 A 行段落在哪些 rank 分片上（`rank_beg`、`rank_end`），然后 `dl.wait(ready_ptr + rank_beg, ..., "gpu", "acquire", waitValue=ready_value)` 并 `dl.consume_token(a_desc, token)`，再做 TMA 加载与 `tl.dot`。
- `copy_and_barrier_all_intra_node_kernel`、`local_copy_and_barrier_all`：把本地分片拷进 AG 缓冲并做跨 rank 屏障。
- 相关：`ag_gemm_threadblock_swizzle.py`、`allgather.py`（`AllGatherMethod` 枚举：`All2All_IntraNode`、`Ring1D_IntraNode`、`Ring2D_InterNode` 等；`get_auto_all_gather_method` 在全互联 NVLink 时选 All2All）。
- AllGather producer 的 copy-engine 部分：`allgather.py` 的 `cp_engine_producer_all_gather_inter_node` 中，对节点内每个目标用 `cudaMemcpyAsync` 搬运，再用 `_set_signal_cuda` 在目标 rank 的信号槽置位，跨节点部分用 `_wait_eq_cuda` 等收到后再转发。

### GEMM + ReduceScatter

文件：`third_party/Triton-distributed/python/triton_dist/kernels/nvidia/gemm_reduce_scatter.py`

- `GEMMReduceScatterTensorParallelContext`、`create_gemm_rs_context`：含 `rs_stream`、`num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms`。
- `kernel_gemm_rs_producer_persistent`：GEMM 是"生产者"。每个 tile 完成并写出后，对覆盖到的每个 rank 行段递增原子计数器（`tl.atomic_add(..., sem="release", scope="gpu")`）；当某行段的所有 tile 都完成（计数达到 `tiled_m_size * num_pid_n`），调用 `dl.notify(barrier_ptr + counter_id, rank, signal=1, comm_scope="gpu")`。`FUSE_SCATTER` 分支则直接用 `dl.symm_at(c_ptr, cur_rank)` 把结果 `tl.store` 到对应 rank 的缓冲。
- `kernel_gemm_rs_producer_persistent_reduce_st`、`kernel_gemm_rs_producer_non_persistent`：另两种变体。
- `gemm_rs_op`（主入口）：先在主 stream 上启动 GEMM 生产者 kernel；`fuse_scatter=False` 时在 `rs_stream` 上调用 `reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)` 做归约，再让主 stream `wait_stream(rs_stream)`；`fuse_scatter=True` 时 GEMM 已把结果写到各 rank，之后用 `nvshmem_barrier_all_on_stream` 夹住 `ring_reduce` 本地归约。reduce-scatter 的各种实现在 `reduce_scatter.py`（如 `reduce_scatter_ring_push_1d_intra_node_ce` 走 copy engine、`..._sm` 走 SM；`num_rs_sms` 属性决定通信占 SM 数）。
- `gemm_rs` 带 autotune，`get_gemm_rs_config_space` 给出搜索空间。

### MoE 与 EP all2all

- `kernels/nvidia/low_latency_all_to_all.py`：`all_to_all_kernel`（`triton_dist.jit`），README 以它为例。逻辑：每个 program 负责一个目标 rank，`libshmem_device.putmem_nbi_block` 把数据块、splits、scale 推到对端对称缓冲，用 `call_count % 2` 做双缓冲，`putmem_signal_nbi_block` 或 `signal_op` 置信号，`fence()`，再由线程 0 `signal_wait_until` 等对端信号。入口 `fast_all_to_all`。README 称在 32 张 H800 上 137 us，对比 DeepEP 的 182 us（数字取自 README，未复现；且版本与现 DeepEP V2.5 不同）。
- `kernels/nvidia/ep_a2a.py`：`kernel_dispatch_token`、`kernel_combine_token`、`ep_dispatch_token_inplace`、`ep_combine_token_inplace`（高吞吐 EP）。
- `kernels/nvidia/ep_all2all_fused.py`：`tile_kernel_dispatch_token_intra_node`、`tile_kernel_moe_grouped_gemm_nk_const`、`tile_kernel_gather_combine_token_intra_node` 等——把 dispatch、grouped GEMM、combine 放进同一套 tile kernel 里，是 MoE 通算融合的代表。
- `ep_a2a_intra_node.py`、`low_latency_all_to_all_v2.py`、`all_to_all_single_gemm.py` 等为变体。
- TP MoE：`ag_moe`（`threadblock_swizzle_ag_moe_triton.py`）、`moe_reduce_rs.py`、`moe_reduce_ar.py`、`allgather_group_gemm.py`、`group_gemm.py`。
- 其他融合：`gemm_allreduce.py`、`allreduce.py`（含 `one_shot`、`two_shot`、`double_tree`、`one_shot_multimem` 方法，见 `docs/testing.md`）、序列并行 `sp_ag_attention_*`、`sp_ulysess_*`、`flash_decode.py`。
- 层封装：`layers/nvidia/tp_mlp.py`（`TP_MLP`）、`tp_attn.py`、`tp_moe.py`、`ep_moe.py`、`ep_a2a_layer.py`、`ep_ll_a2a_layer.py`、`gemm_allreduce_layer.py`。

### 教程（`third_party/Triton-distributed/tutorials/`，由易到难）

`01-distributed-notify-wait.py`（原语）→ `02-intra-node-allgather.py`（copy engine + NVSHMEM 做 AllGather）→ `03-inter-node-allgather.py` → `04-deepseek-infer-all2all.py` → `05-intra-node-reduce-scatter.py` → `06-inter-node-reduce-scatter.py` → `07-overlapping-allgather-gemm.py` → `08-overlapping-gemm-reduce-scatter.py`，另有 AMD 版 `09`、`10`。建议阅读顺序即此。

### MegaTritonKernel

`python/triton_dist/mega_triton_kernel/`（`core`、`kernels`、`models`、`tasks`、`test`）：把一个模型层甚至整个 decode 步编译进单个 persistent kernel，减少 launch 与 kernel 间空泡。文档 `docs/getting-started/megakernel/megakernel.md` 给出 8×H800 TP=8 下 Qwen-32B 单步 decode：torch+cudagraph 10.80 ms、triton_dist_AR+cudagraph 9.18 ms、mega_triton_kernel 7.41 ms（文档数字，未复现）。与书中第 5.5.4 节的 persistent kernel 思想一致：任务队列 + 完成标志，少 launch、按块重叠。

## 文档给出的运行方式（均需 GPU，本机不可运行）

来源：`third_party/Triton-distributed/docs/build.md`、`docs/testing.md`、`README.md`。

- 环境：Python ≥3.11、CUDA ≥12.4、Torch ≥2.8，推荐 NVIDIA PyTorch 容器；AMD 需 ROCm 7.1。依赖 NVSHMEM（`docs/prepare_nvshmem.md`、`scripts/build_nvshmem_from_src.sh`，后者默认 `NVSHMEM_IBGDA_SUPPORT=1`）。
- 启动：`bash scripts/launch.sh <test.py> [args]`，可用 `--nproc_per_node=N`。
- 例（取自 `docs/testing.md`）：
  - AG+GEMM：`bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check`（加 `--autotune` 可调优）。
  - GEMM+RS：`bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check`（`--fuse_scatter` 切换变体）。
  - EP A2A：`NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check`；低延迟版 `test_ep_ll_a2a.py`。
  - 教程：`bash scripts/launch.sh tutorials/07-overlapping-allgather-gemm.py`。
- 常用环境变量（`docs/testing.md` 表）：`NVSHMEM_SYMMETRIC_SIZE`（对称堆大小）、`NVSHMEM_DISABLE_CUDA_VMM`、`NVSHMEM_IBGDA_SUPPORT`、`USE_TRITON_DISTRIBUTED_AOT`、`CUDA_DEVICE_MAX_CONNECTIONS`。
- 测试脚本的共同结构（以 `test_ag_gemm.py` 为例）：`assert_allclose`（`python/triton_dist/test/utils.py`）与 torch 基线对比、`perf_func`（`python/triton_dist/profiler_utils.py`）测时、`group_profile` 抓 profile。

## 性能要点与常见坑

- 小 shape 融合不一定赢：E2E 文档中 Attn prefill/decode 的 AG+RS 方案慢于 Torch AllReduce（0.961、0.985），而 GEMM+AllReduce 在同类 shape 上快 1.39–1.44 倍（8×H800，文档数字）。
- 对称堆不够会初始化失败：调大 `NVSHMEM_SYMMETRIC_SIZE`（`docs/testing.md` Troubleshooting）。
- allreduce 卡死：测试脚本有 `--verify_hang` 超时参数。
- 信号复用：EP kernel 用 `call_count % 2` 双缓冲信号与数据，避免上一轮的信号被误判为本轮（`low_latency_all_to_all.py`）。
- 版本漂移：默认分支追踪 Triton 3.7 的插件构建，3.4 线在分支 `triton-v3.4`；README 提示 Ascend 需用 3.4 分支。

## 面试自测

1. Triton-distributed 在 Triton 之上加了哪几个核心原语？各自作用？
   - 要点：`wait`（自旋等信号，返回 token）、`consume_token`（把 token 绑到值上建立依赖）、`notify`（写信号）、`symm_at`（对端指针）、`rank/num_ranks`；加上 NVSHMEM device 函数（putmem/signal/fence）。
2. 为什么需要 `consume_token`？
   - 要点：`wait` 本身没有数据依赖，编译器可能把后面的加载提前；把 token 与待加载的指针/descriptor 绑定后，加载必须在 wait 之后。
3. 画出 AG+GEMM 的执行时间线，说明谁是生产者、谁是消费者、依赖如何表达。
   - 要点：通信 stream（copy engine 或 NVSHMEM kernel）写 AG 缓冲并置分片信号；GEMM tile 在加载 A 前等待所需分片信号；本地分片优先的 swizzle 使首批 tile 无需等待；两 stream 末尾 `wait_stream` 合流。
4. GEMM+ReduceScatter 里，如何判断可以通知归约侧？
   - 要点：按 rank 行段给每段一个原子计数器，tile 写完后 release 语义递增；计数达到该段 tile 总数的那个 tile 负责 `notify`。
5. GEMM 与通信的 SM 怎么分？
   - 要点：GEMM SM = 总 SM − 通信 SM（AG 跨节点 `n_nodes−1`，RS 的 `num_rs_sms`）；节点内 AG 走 copy engine，几乎不占 SM。
6. MegaTritonKernel 与普通通算融合的区别？
   - 要点：不止融合通信和相邻 GEMM，而是把整层/整模型任务调度进一个 persistent kernel，依靠任务队列与完成标志按块推进，省 launch 与空泡。

## 待深入

- 待确认：`lib/` 与 `plugin/` 中 distributed dialect 的 lowering 细节（`wait` 如何降到 PTX 自旋与内存序）；确认方式：读 `lib/Conversion` 与 `plugin/lib`。
- 待确认：`FlashComm/` 的 EP overlap 设计；确认方式：读 `FlashComm/csrc/ep` 与 `FlashComm/tests/ep_overlap`。
- 待确认：本机不可运行；若要实测，需 8 卡 NVLink 机器与 NVSHMEM。
