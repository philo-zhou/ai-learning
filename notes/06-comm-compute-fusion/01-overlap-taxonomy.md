# 通算重叠 / 融合分类

> 层次：跨层 · 来源：third_party/Triton-distributed@7908e4ea、third_party/DeepEP@b5a33eef、third_party/nccl@3e46c089、ai-infra-book 第 5.5.4、6.4.3、6.4.4、7.3.2、9.4.3、10.3.3 节

## 一句话

按"重叠粒度"和"谁发起通信"两条轴，通算重叠可以分五层：stream 级、算子拆分流水（tile 级）、kernel 内融合（device-initiated）、copy-engine 卸载，以及贯穿其中的"通信占多少 SM"权衡。粒度越细、融合越深，收益上限越高，实现与正确性成本也越高。

## 在调用链中的位置

- 上层框架（PyTorch DDP/FSDP、Megatron、vLLM/SGLang 的 TP/EP 层）决定用哪一种重叠方式。
- 下层是 NCCL（host-initiated，stream 语义）、NVSHMEM / NCCL device API（device-initiated）、CUDA copy engine、NIC。
- 本文是后三篇的总纲：02 讲 Triton 写 tile 级融合，03 讲 device-initiated 通信底座，04 讲设计检查清单。

## 核心概念

### 分类总览

| 层级 | 重叠粒度 | 谁发起通信 | 典型例子 | 主要代价 |
|---|---|---|---|---|
| A. stream 级 | 整个集合通信 vs 整段计算 | 主机（NCCL 提交到另一条 stream） | DDP 梯度分桶、FSDP/ZeRO-3 参数预取 | 受依赖粒度限制；通信 kernel 占 SM、争 HBM |
| B. 算子拆分流水 | tile / 分片 | 主机或 kernel | AllGather+GEMM、GEMM+ReduceScatter、MoE 的 micro-batch 流水 | 需要分片间的依赖信号；小 shape 反而变慢 |
| C. kernel 内融合 | 一个 kernel 内算完即发 | GPU 线程（device-initiated） | DeepEP dispatch/combine、Triton-distributed 的 GEMM+RS、NCCL device API 的 RMSNorm 融合示例 | 要分出 SM；需要 symmetric memory、内存序保证 |
| D. copy-engine 卸载 | 分片 | 主机或 CE | Triton-distributed AllGather 的 cp_engine 路径、DeepEP bucket all-gather（`num_sms=0`） | 只适合搬运不适合归约；有 CE 调度延迟 |
| E. SM 数权衡 | 横切 | — | DeepEP `num_sms`、Triton-distributed 的 GEMM SM 数 = 总 SM − 通信 SM | 通信越快、计算越慢，要联合调 |

下图把前三层的时间线放在一起对比：串行、A 层 stream 级重叠、B 层 tile 级流水、C 层 kernel 内融合（块数 4 只是示意）。

```text
time ───────────────────────────────────────────────────────►

(1) 串行：同一 stream 上先通信、后计算
comm     ┌────────────────┐
         │ AllGather      │
         └────────────────┘
gemm                      ┌────────────────┐
                          │ GEMM           │
                          └────────────────┘

(2) A. stream 级：整个集合通信放到另一条 stream，与无依赖的计算同时跑
comp     ┌────────────┐┌────────────┐
         │ comp #1    ││ comp #2    │
         └────────────┘└────────────┘
comm                   ┌────────────┐
                       │ comm #1    │   ← 通信在另一条 stream，与 comp #2 并行
                       └────────────┘

(3) B. tile 级流水：第 k 块到了就先算第 k 块
comm     ┌──────┐┌──────┐┌──────┐┌──────┐
         │ c0   ││ c1   ││ c2   ││ c3   │
         └──────┘└──────┘└──────┘└──────┘
comp             ┌──────┐┌──────┐┌──────┐┌──────┐
                 │ t0   ││ t1   ││ t2   ││ t3   │   ← 首块填充、末块收尾仍暴露
                 └──────┘└──────┘└──────┘└──────┘

(4) C. kernel 内融合：一个 kernel 内，算完一块线程直接 put / signal
SM-comp  ┌──────┐┌──────┐┌──────┐┌──────┐
         │ t0   ││ t1   ││ t2   ││ t3   │
         └──────┘└──────┘└──────┘└──────┘
SM-comm          ┌──────┐┌──────┐┌──────┐┌──────┐
                 │ p0   ││ p1   ││ p2   ││ p3   │   ← 无主机往返，但要划出 SM
                 └──────┘└──────┘└──────┘└──────┘
```

### A. stream 级 overlap

- **DDP 梯度分桶**：反向逐层产生梯度，多个梯度合并成一个桶，桶内梯度都算完才发起 AllReduce。书中第 10.3.3 节：桶越大越可能等最后一个梯度，桶越小启动次数越多。另外，先算 $dX$ 让前一层更早反向，先算 $dW$ 则本层梯度通信更早启动，计算顺序本身也是调度变量。
- **FSDP / ZeRO-3 预取**：执行某层前预取它的参数，反向一层结束立即归约该层梯度。书中同节算例：48 卡每 micro-batch 链路时间约 1.50 s、计算约 6.53 s，逐层看大部分通信落在计算之内，真正暴露的只有一步里第一次 AllGather 和最后一次 ReduceScatter。书中还指出，共用一张网卡时"按期完成押在重叠上"，每卡一张网卡则不依赖重叠。
- 机制：通信放在另一条 CUDA stream，用 event 或 `wait_stream` 建依赖。Triton-distributed 的 AG+GEMM 外层也是这个模式，见 `third_party/Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py` 中 `rowise_ag_gemm_dispatcher` 里的 `wait_stream` 调用。
- 局限：依赖粒度是"整个张量"，且 NCCL kernel 默认占用 SM（见 E）。
- 待确认：PyTorch DDP `bucket_cap_mb`、FSDP `forward_prefetch` / `backward_prefetch` 的具体实现；原因：PyTorch 源码未克隆；确认方式：克隆 pytorch 后查 `torch/nn/parallel/distributed.py` 与 `torch/distributed/fsdp`。

### B. 算子拆分流水（tile 级）

- 思路：把"先 AllGather 整个输入再做 GEMM"改成"哪一块到了就先算哪一块"。
- **AllGather + GEMM**：每个 rank 持有 `[M/world, K]` 的输入分片，GEMM 需要完整的 `[M, K]`。拆成：通信侧按 rank 分片收集并置位"该分片就绪"信号；GEMM 侧每个 tile 在加载 A 之前等待它所覆盖的分片信号。本地分片不需要等，所以 tile 调度会让每个 rank 先算自己已有的分片（swizzle）。实际代码见 02 文档对 `kernel_consumer_gemm_persistent` 的导读。

下图示意 AllGather+GEMM 的依赖信号与 swizzle：本 rank 先算本地分片，其余分片的 tile 在对应 ready 信号置位后才加载 A（world=4、本 rank=1、分片到达顺序均为示意）。

```text
comm    ┌─────┐┌─────┐┌─────┐
        │ S2  ││ S3  ││ S0  │   ← 通信侧：每个分片到齐就置 ready[Sk]
        └─────┘└─────┘└─────┘
gemm    ┌─────┐┌─────┐┌─────┐┌─────┐
        │ S1  ││ S2  ││ S3  ││ S0  │   ← GEMM 侧：覆盖该分片的 tile
        └─────┘└─────┘└─────┘└─────┘
        (local)   ▲ 加载 A 之前 wait ready[S2]；本地分片 S1 不用等
```

- **GEMM + ReduceScatter**：反过来，GEMM 一个 tile 算完，就按其所在行段通知归约侧可以开始。Triton-distributed 中用原子计数器统计某个 rank 对应的 tile 是否全部完成，完成后再 `notify`（见 02）。
- **MoE 两阶段流水**：dispatch → 专家计算 → combine，拆成两个 micro-batch，第一批计算时第二批 dispatch。书中第 9.4.3 节：理想值 0.827 ms → 0.581 ms，但若 DMA 与 GEMM 争 HBM 使通信各慢 49%，收益归零。
- 局限：拆分带来额外的启动次数与同步；Triton-distributed 的 E2E 文档（`third_party/Triton-distributed/docs/getting-started/e2e/e2e_dense.md`）明确写到，Attention 部分 shape 太小时 AG+RS 拆分反而比单个 AllReduce 慢（speedup 0.961、0.985），此时应改用 GEMM+AllReduce。
- 正是这类"依赖信号 + 分片调度"的设计，构成 JD 里"设计通信算子与计算算子的融合方案"的主体内容。

### C. kernel 内融合（device-initiated）

- 传统 host-initiated：GPU 算完 → CPU 轮询或事件 → CPU 提交通信 → GPU 再继续。书中第 6.4.4 节：CPU 代理线程路径控制信号要穿越 PCIe 三次；GPU SM 发起则两次；NIC 上处理器则为零。
- device-initiated：kernel 里的线程直接 put/get/signal 到对端 symmetric memory。好处是"数据一算完就能发"，并省掉主机往返（书中第 7.3.2 节）；代价是要划出一部分 SM 构造请求、轮询完成。

下图对比两种发起方式的控制路径：host-initiated 每次都要回到 CPU，device-initiated 由 kernel 内线程直接发起。

```text
host-initiated
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ GPU compute │────►│ CPU poll or │────►│ CPU submits │────►│ GPU resumes │
│ kernel done │     │ event wait  │     │ comm        │     │ next kernel │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

device-initiated
┌───────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ tile k done   │────►│ thread put / signal │────►│ compute tile k+1 │   ← put 写入对端 symmetric memory，不回到主机
└───────────────┘     └─────────────────────┘     └──────────────────┘
```

- 例子：
  - `third_party/DeepEP/csrc/kernels/ep/dispatch.hpp` 与 `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh`（kernel 内 `gin.put`，注释里写 "Issue IBGDA requests"）。
  - NCCL device API 示例 `third_party/nccl/docs/examples/07_kernel_fusion/README.md`：RMSNorm 与 reduce-scatter、all-gather 融合，用 LSA（NVLink 域内直接指针访问）、GIN（GPU-initiated networking）或混合模式。该 README 自述示例不以性能为目标。
- 细节在 [03-nvshmem-and-device-initiated.md](03-nvshmem-and-device-initiated.md)。

### D. copy-engine 卸载

- Copy engine（CE）是 GPU 上独立于 SM 的 DMA 引擎，做 `cudaMemcpyAsync` 类搬运不占 SM。
- Triton-distributed：AllGather 的 producer 走 CE（函数名 `cp_engine_producer_all_gather_intra_node`，位于 `third_party/Triton-distributed/python/triton_dist/kernels/nvidia/allgather.py`，内部调用 `cudaMemcpyAsync`），教程 02 标题即 "Use copy engine and NVSHMEM primitives for AllGather"（`third_party/Triton-distributed/tutorials/README.md`）。这样 GEMM 能用满几乎所有 SM。
- DeepEP：README 写明 bucket 的 all-gather 由 copy engine 驱动（`num_sms=0`），reduce-scatter / all-reduce 仍需 SM（`third_party/DeepEP/README.md`）；同时声明 EP dispatch/combine 需要 SM，不支持零 SM 的 RDMA EP。
- NCCL：有 `NCCL_CTA_POLICY` 环境变量，取值 DEFAULT/EFFICIENCY/ZERO（`third_party/nccl/docs/userguide/source/env.rst`）；`third_party/nccl/src/ce_coll.cc` 里有 CE 集合实现与 `NCCL_CE_CHUNK_SIZE`、`NCCL_CE_INTRA_GPU_MEMCPY_ENABLE` 等参数。具体哪些集合在什么拓扑下走 CE，待确认；确认方式：读 `src/ce_coll.cc` 与 `src/include/ce_coll.h` 以及文档 `docs/userguide/source` 中 CTA policy 一节（`cta_policy_flags`）。
- 规律：搬运类（AllGather、Broadcast、AllToAll）适合 CE；需要加法的归约通常仍要 SM 或交换机内归约（NVLS）。书中第 6.4.4 节还提到昇腾 CCU 是同一思路的专用单元。

### E. 通信占 SM 数的权衡

- 通信线程/channel 越多，单独跑越快，但抢走计算资源。书中第 6.4.3 节：AutoCCL 用含并发干扰的实测时间调线程、通道、分块；重计算干扰下 AllGather 带宽 18.26 → 32.44 GB/s（论文数字，非本地复现）。
- Triton-distributed：`gemm_sm = max_gemm_sm - internode_ag_sm`（AG+GEMM，见 `rowise_ag_gemm_dispatcher`）；GEMM+RS 的 `num_gemm_sms = NUM_SMS - rs_ctx.num_rs_sms`（见 `third_party/Triton-distributed/python/triton_dist/kernels/nvidia/gemm_reduce_scatter.py` 的 `create_gemm_rs_context`）。
- DeepEP V2.5：SM 与 QP 数通过 `get_theoretical_num_sms` 解析估算，"no more auto-tuning needed"，也允许每次调用传 `num_sms` 覆盖（`third_party/DeepEP/README.md`）。
- 这是一个联合优化问题：总时间 ≈ max(计算时间(SM 数), 通信时间(SM 数))，最优点通常在两者接近处。

## 性能要点与常见坑

- 重叠收益要看"同时运行"的实测时间，不能用单独时间相减（书 6.4.3）。
- 拆得过细：launch/同步开销和小消息效率下降；拆得过粗：流水填充/收尾（第一块 dispatch 和最后一块 combine）暴露。
- 同一 stream 上通信与计算会串行，要确认真的在不同 stream 且无隐式依赖。
- 争用的不只是 SM，还有 HBM 带宽（书 9.4.3）和 NIC / 互联出口（书 6.4.4：出口字节数不会因让出 SM 而减少）。
- 融合 kernel 要保持语义顺序：书 6.4.4 的例子，RMSNorm 必须在归约完成后做，否则结果不同。

## 面试自测

1. DDP 的梯度分桶为什么既不能太大也不能太小？
   - 要点：桶大 → 要等最后生成的梯度，通信开始晚；桶小 → 通信启动次数多、小消息效率低。取折中，且依赖反向的产生顺序。
2. AllGather+GEMM 做 tile 级重叠，依赖信号怎么设计？
   - 要点：按 rank 分片为粒度置位 ready 信号；GEMM tile 加载 A 前，等待该 tile 行范围覆盖的分片信号；本地分片无需等待，tile 顺序按 rank 做 swizzle 先算本地/先到的分片。
3. 为什么说 device-initiated 通信更快？它的代价是什么？
   - 要点：省掉 GPU→CPU→NIC 的控制路径往返；数据一算完即可发。代价是占 SM、需要对称内存与显式内存序、调试困难。
4. copy engine 适合什么、不适合什么？
   - 要点：适合纯搬运（AllGather、拷贝）且希望让出 SM；不能做归约；有额外调度延迟，小消息未必占优。
5. 如何选择通信占用的 SM 数？
   - 要点：不是越多越好；通信带宽有上限（NIC/NVLink 带宽饱和后再加 SM 无益）；计算与通信时间接近处最优；用含并发干扰的实测来调，如 AutoCCL 思路或 DeepEP 的解析估算。
6. 什么情况下拆分 AllReduce 为 AG+RS 做重叠反而变慢？
   - 要点：shape 很小时，拆分的额外 launch 与同步超过重叠收益；Triton-distributed E2E 文档中 Attention 的 AG+RS 方案出现 0.961/0.985 的 speedup 即是此类例子。

## 待深入

- 待确认：Megatron / PyTorch 中 TP 通算重叠（如 userbuffer、async TP）的具体实现；原因：对应源码未克隆；确认方式：克隆后搜索 `async_tp`、`overlap`。
- 待确认：NCCL 各集合何时走 CE；确认方式：见 D 节。
