# 通算融合算子设计检查清单

> 层次：跨层 · 来源：third_party/Triton-distributed@7908e4ea、third_party/DeepEP@93eb6eb2、third_party/nccl@3e46c089、ai-infra-book 第 5.5.4、6.4.3、6.4.4、9.4.3、10.3.3 节
>
> 本文是方法清单，具体条目都落到前三篇已导读的代码上；凡属一般工程经验而非本地源码证据的，会注明"经验"。

## 一句话

设计一个通算融合算子，先回答"值不值得融"（收益上限），再回答六件事：切多细、靠什么信号表达依赖、分多少 SM、内存序对不对、拓扑怎么映射、怎么证明正确与更快。

## 在调用链中的位置

- 输入：上层层（TP MLP、MoE 层）给出的 shape、并行度、拓扑。
- 输出：一个可被 `layers/` 替换的算子，以及正确性测试 + 基准报告。
- 参考实现：[02-triton-distributed.md](02-triton-distributed.md)。分类见 [01-overlap-taxonomy.md](01-overlap-taxonomy.md)，底层机制见 [03-nvshmem-and-device-initiated.md](03-nvshmem-and-device-initiated.md)。

## 核心清单

### 0. 先估收益上限（动手前）

- [ ] 测基线的通信时间 $T_{comm}$、计算时间 $T_{comp}$、以及两者并发时的实际时间（书 6.4.3：不能用单独时间相减）。
- [ ] 理想流水收益 ≈ $\min(T_{comm}, T_{comp})$ 减去填充/收尾；若 $T_{comm}$ 远小于 $T_{comp}$（或相反），收益上限就很低（书 5.5.4：两个阶段耗时越接近，按块重叠省得越多；差距大时只剩少一次 launch）。
- [ ] 检查通信是否瓶颈在共享出口（NIC、NVLink 出口）；若是，让出 SM 不会减少字节数（书 6.4.4）。
- [ ] 小 shape / 小消息：先看是否只需减少 launch（CUDA Graph、persistent kernel、更低延迟的小消息算法）。Triton-distributed 的 E2E 文档显示 Attention 小 shape 上 AG+RS 拆分反而更慢，而 GEMM+AllReduce 更快（`third_party/Triton-distributed/docs/getting-started/e2e/e2e_dense.md`）。

### 1. 切分粒度

- [ ] 通信分片粒度 = 依赖信号粒度；常见取"每个 rank 一个分片"（AG+GEMM 里 `ready_ptr + rank_beg`）或"每个 tile 行段一个计数器"（GEMM+RS）。
- [ ] 粒度太细：信号数量与自旋开销增加、小消息 NIC 效率低（书 7 章提到的报文速率问题）；太粗：流水填充/收尾暴露，首块 dispatch 与末块 combine 不可重叠（书 9.4.3）。
- [ ] tile 边界与分片边界不对齐时，一个 tile 可能跨多个分片（代码里 `rank_beg` 到 `rank_end` 的范围等待），要保证全部覆盖。
- [ ] padding：Grouped GEMM 对每个专家按 tile 补齐，会让硬件执行的行数远大于有效 token（书 9.4.3：3.5–4 倍），融合时要决定补齐位置（`do_zero_padding` 在 DeepEP README 中的语义）。
- [ ] 不均匀（MoE 的 splits）：dispatch 数据量随 batch 变化，需要先交换计数/偏移，再组织接收空间；融合算子中通常把 splits 与 payload 一起放进同一次通信流程（见 `low_latency_all_to_all.py`）。

### 2. 依赖信号设计

- [ ] 明确每个信号的**生产者**、**消费者**、**语义值**（`set` 到固定值 vs `add` 计数；`dl.notify` 的 `sig_op`）。
- [ ] 信号复用：多次调用之间如何区分？Triton-distributed 的 EP all2all 用 `call_count` 当信号值并做 `call_count % 2` 双缓冲；其他做法是每轮清零（要保证清零与上一轮读取之间的顺序）。
- [ ] 完成语义：单个 put 完成 vs 所有此前 put 完成。NCCL GIN 明确区分强/弱信号（`third_party/nccl/src/include/nccl_device/gin.h`）：要让一个信号代表"一批数据到齐"必须用强信号或先 flush。
- [ ] 避免死锁：消费者等的信号，其生产者是否一定会被调度？persistent kernel 或占满全部 SM 的 kernel 会让"生产者 kernel 没有 SM 可用"，经验上要给通信侧预留 SM（见 3）。同 stream 串行时，等信号的 kernel 永远等不到后发的生产者。
- [ ] 超时与可观测：DeepEP 有 GPU 端超时（`num_gpu_timeout_cycles`，见 `third_party/DeepEP/csrc/kernels/comm/context.cpp`）；Triton-distributed 测试有 `--verify_hang`（`docs/testing.md`）。自己写时也应有超时断言，而不是无限自旋。

### 3. SM 分配与资源争用

- [ ] 明确公式：计算 SM = 总 SM − 通信 SM；参考 AG+GEMM 的 `gemm_sm = max_gemm_sm - internode_ag_sm`、GEMM+RS 的 `num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms`。
- [ ] 可以不用 SM 的路径优先：节点内 AllGather 走 copy engine；DeepEP bucket all-gather `num_sms=0`；NCCL `NCCL_CTA_POLICY`。归约一般仍需 SM（或 NVLS/multimem）。
- [ ] 通信 SM 数不要超过让链路饱和所需（DeepEP 解析估算 `get_theoretical_num_sms`，并允许覆盖）；扫描 SM 数得到"计算时间 vs 通信时间"曲线，取交点附近。
- [ ] HBM 带宽争用：DMA / 通信读写 HBM 与 GEMM 竞争（书 9.4.3：通信慢 49% 即可抹平两 micro-batch 流水收益）；必要时错峰、降低通信块大小或让 CE 与 SM 的访问不同时集中。
- [ ] persistent kernel 的 `NUM_SMS` 与 grid：grid 不应超过可同时驻留的 CTA 数，否则生产者与消费者互等（经验）。
- [ ] 若有 CUDA Graph，注意 stream 捕获与对称内存分配不能在捕获中做（Triton-distributed 代码里有注释"don't allocate memory on other stream: error-prune"，见 `gemm_reduce_scatter.py`）。

### 4. 内存序与 fence

- [ ] 生产者：写数据 → release（或 fence）→ 写信号。消费者：读信号 → acquire → 读数据。Triton-distributed：`dl.wait(..., "gpu", "acquire")` 与 `tl.atomic_add(..., sem="release", scope="gpu")`、`dl.notify` 成对使用。
- [ ] scope 选对：同 GPU 内用 `gpu`；跨 GPU 经 NVLink 需要 `sys` 级别可见性或 NVSHMEM 的 fence/quiet（`dl.notify` 的 `comm_scope` 区分 `gpu / intra_node / inter_node`）。
- [ ] 跨节点 put 与 signal 的顺序：用 put-with-signal 原语（`putmem_signal*`）或 NCCL GIN 的强信号，而不是"先 put 再普通写一个 flag"。
- [ ] 数据依赖：wait 之后的加载必须真的排在 wait 之后。Triton 里用 `dl.consume_token(value, token)` 把 token 绑到被加载的指针/descriptor 上。
- [ ] TMA / 异步拷贝：TMA store 要 commit 与 wait 后才能发信号（DeepEP `hybrid_dispatch.cuh` 里先 `tma_store_commit`/`tma_store_wait` 再发起 put）。
- [ ] buffer 重用：发送完成（flush / local completion）前不能覆盖源缓冲；接收端消费完前不能被下一轮写入（双缓冲或显式 credit）。
- [ ] stream 边界：通信 stream 与计算 stream 之间用 `wait_stream` / event（`rowise_ag_gemm_dispatcher`）；DeepEP 的 `previous_event`、`EventOverlap.wait()` 同类。张量要 `record_stream` 避免被提前释放（DeepEP 的 `tensors_to_record`，见 `third_party/DeepEP/csrc/buffers/ep.hpp`）。

### 5. 拓扑感知

- [ ] 区分域内（NVLink/NVSwitch，可用指针直接 load/store）与跨域（RDMA）；NCCL 里对应 LSA team 与 Rail team（`ncclTeamLsa`、`ncclTeamRail`）。
- [ ] 全互联 NVLink 机器用 all2all 式 AllGather，否则用 ring（Triton-distributed 的 `get_auto_all_gather_method` 依据 `has_fullmesh_nvlink`）。
- [ ] 多节点用两级：节点内 NVLink + 节点间 rail（每个 local rank 与同号 rank 互联）；DeepEP hybrid 模式即此。
- [ ] tile 调度要感知拓扑：多节点 AG+GEMM 用 `threadblock_swizzle_allgather_gemm_kernel`，让最先到达的数据最先被消费。
- [ ] 一台机器 8 卡共用一张 NIC 与每卡一张 NIC 的差别可能决定"必须靠重叠"还是"不依赖重叠"（书 10.3.3）。
- [ ] QP/连接数与每秒请求率：device-initiated 的请求率受 SM 数、QP 数、队列深度限制（DeepEP `ginContextCount`、`ginQueueDepth`）。
- [ ] RDMA 层面的流量隔离（SL / traffic class）：DeepEP 通过 `EP_DEFAULT_RDMA_SL`、`EP_OVERRIDE_RDMA_SL` 设置；多租户/多业务混跑时要关注。

### 6. 正确性测试

- [ ] 与参考实现对比：同输入分别跑融合算子与 `torch.distributed` + `torch.matmul` 的非融合版本，用 `assert_allclose`（`third_party/Triton-distributed/python/triton_dist/test/utils.py`）；融合算子累加顺序不同，用 rtol/atol（`test_ag_gemm.py` 中 AG+GEMM 用 atol/rtol=1e-3）而非逐位相等；有明确保证的路径（如不改变归约顺序）可做 bitwise 比较（Triton-distributed README 新闻提到 FlashComm MXFP8 dispatch 做了 bitwise 校验）。
- [ ] 多次迭代与压力：同一算子循环调用、变化 shape、变化 splits（含空专家、全落在一个专家），验证信号复用无串扰（Triton-distributed allreduce 测试用 `--stress`）。
- [ ] 异步扰动：给某个 rank 人为 sleep（`rowise_ag_gemm_dispatcher` 有 `straggler_option`，调用 `torch.cuda._sleep`）以暴露缺失的等待。
- [ ] 多拓扑：1 节点 2/4/8 卡、多节点；有无 NVLink 全互联。测试命令里的 `--nproc_per_node` 可调（`docs/testing.md`）。
- [ ] 死锁检测：超时 + 打印每个 rank 所等信号值。
- [ ] 只在有 GPU 的机器上可跑；本机仅能做静态检查。

### 7. 性能基准方法

- [ ] 指标：端到端算子时间（毫秒或微秒）、对基线 speedup、有效 TFLOPS（可与理论 `get_tensorcore_tflops` 对比）、有效通信带宽（对 NVLink/IB 峰值）。`test_ag_gemm.py` 同时引用 `get_tensorcore_tflops` 与 `get_intranode_max_speed_gbps`。
- [ ] 基线要公平：同一 shape、同一 dtype、同一进程组；基线至少包括"非融合串行"与"stream 级重叠"两种，才能说明 tile 级融合额外带来多少。
- [ ] 计时：warmup + 多次迭代 + CUDA event 或 `perf_func`（`profiler_utils.py`）；多 rank 取最慢者；如要排除 L2 缓存影响，可用 `perf_func_with_l2_reset`。
- [ ] 扫参数：M（batch×seq）、N、K、tile 大小、SM 分配、通信方法；使用 autotune（`triton_dist.tune.autotune`）并保留搜索空间配置与最终选中配置。
- [ ] 分解 timeline：用 profiler（`group_profile`、intra-kernel profiler，文档 `docs/getting-started/profiler/intra_kernel_profiler.md`）看通信与计算是否真的并行、哪里出现空泡。
- [ ] 与书中口径对齐：NCCL 的 algbw / busbw 定义（书 6.4.3）；busbw 是归一化指标，不能直接当 NIC 吞吐。
- [ ] 记录环境：GPU 型号、驱动、CUDA、NCCL / NVSHMEM 版本、拓扑（`nvidia-smi topo -m`）、关键环境变量（如 `NVSHMEM_SYMMETRIC_SIZE`、`NCCL_CTA_POLICY`）。

## 性能要点与常见坑

- 看到"重叠了"就以为"变快了"：必须比较并发时的整体时间。
- 只在大 shape 上测：小 shape 才是延迟敏感场景，融合可能反向。
- 把信号当"全部完成"：弱信号只保证单个 put。
- 忘记 stream 间依赖导致偶发错误：在 GPU 上难复现，需 straggler 注入。
- 版本漂移：Triton-distributed 追踪 Triton 3.7 插件构建；DeepEP 绑定 NCCL ≥ 2.32.3；复现实验要锁版本（见 `docs/references/third-party.md`）。

## 面试自测

1. 你要把 AllGather 和后面的 GEMM 融合，先做哪几件事？
   - 要点：先测通信/计算/并发时间估计收益上限；再定分片粒度与信号；定 SM 分配；写内存序；做正确性与基准。
2. 融合算子的正确性为什么不能用逐位相等？什么情况下可以？
   - 要点：分块累加或归约顺序变化导致浮点结果不同；使用容差；若融合保持相同的归约顺序且不引入新的累加，可做 bitwise 比对。
3. 通信占多少 SM 合适？怎么定？
   - 要点：不是越多越好；通信速率饱和即可；扫描 SM 数得到计算/通信时间曲线取交点；可用 copy engine 减少占用。
4. 描述一次 GEMM+ReduceScatter 的信号设计，以及如何避免一个 rank 读到另一个 rank 尚未写完的数据。
   - 要点：每个目标行段一个计数器，tile 完成后 release 语义递增；达到总数的 tile `notify` 对应 rank；消费者 `wait` 带 acquire；跨 rank 写要有 scope 足够的 fence。
5. 如何证明融合算子比"stream 级重叠"更好，而不是只比串行好？
   - 要点：基线至少三档：串行、stream 重叠、融合；同 shape、同拓扑、多次迭代取最慢 rank；再用 timeline 展示空泡变化。
6. 融合 kernel 为什么可能死锁？如何防？
   - 要点：消费者等待的生产者拿不到执行资源（SM 被占满、stream 顺序、persistent grid 过大）、信号被上轮残留值误触发；防法：预留通信 SM、控制 grid、信号带轮次/双缓冲、GPU 端超时。

## 待深入

- 待确认：Triton-distributed 的 `wait` 在编译器中具体降级成怎样的 PTX（自旋 + 内存序）；确认方式：读 `third_party/Triton-distributed/lib/Conversion` 与 `plugin/lib`。
- 待确认：本章 05（框架集成）、07（性能排查）建好后，补充"如何在 vLLM / SGLang 里落地融合算子"与 nccl-tests 口径的对照链接。
- 待深入：在有 GPU 的机器上按 [02](02-triton-distributed.md) 的命令实际跑 `test_ag_gemm.py --case check` 与 `test_gemm_rs.py --check`，把基线、stream 重叠、融合三档数据填回本篇。
