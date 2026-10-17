# 并行策略与通信模式

> 层次：上层 · 来源：third_party/ai-infra-book@3bdcb4fc 第 6 章（`manuscripts/06-超节点.md`）、第 7 章（`manuscripts/07-数据中心网络.md`）、第 9 章（`manuscripts/09-分布式推理.md`）、第 10 章（`manuscripts/10-训练系统.md`）；third_party/vllm@02653fa、third_party/sglang@d2a4364 作代码佐证

## 一句话

每种并行方式切分一个维度，就会在切分边界上“欠”一次通信：DP 欠梯度、TP 欠部分和、SP 欠序列分片、PP 欠激活、EP 欠 token 路由、CP 欠远端 K/V；CCL 工程师要能把它们折算成“原语 × 消息大小 × 次数 × 是否在关键路径”。

## 在调用链中的位置

- 上层：模型结构（层数 L、hidden h、头数、专家数 E、top-k）+ 并行配置（tp/pp/dp/ep/cp size）+ 负载形状（batch、序列长度、prefill/decode）。
- 本层：并行层把这些映射成进程组上的集合通信调用（见 [02](02-vllm-communication.md)、[03](03-sglang-communication.md)）。
- 下层：NCCL AllReduce/AllGather/ReduceScatter/Send-Recv/AlltoAll，或 DeepEP 这类专用 kernel（见 [04](04-deepep.md)）。

## 核心概念

### 1. 各并行方式产生的通信

记号：b=batch，s=序列长度，h=hidden，L=层数，d=组大小，N=参数量，dtype 字节数 β（BF16 为 2）。

| 并行 | 切分维度 | 产生的通信 | 单次消息量级 | 次数 | 关键路径？ |
|---|---|---|---|---|---|
| DP（训练） | batch | 反向后梯度 AllReduce（通常分桶） | 每桶 MiB 级；PyTorch DDP 默认桶 25 MiB（见 05） | 每步若干桶 | 可与反向重叠，只有最后一个桶暴露 |
| DP（推理） | 请求 | 无模型内通信（独立副本） | — | — | — |
| ZeRO-1/2 | 优化器状态 / 梯度 | 梯度 ReduceScatter + 参数 AllGather（替代一次 AllReduce） | 与 DP 同量级 | 每步 | 大部分可重叠 |
| ZeRO-3 / FSDP | 参数也分片 | 前向前 AllGather 参数、反向前再 AllGather、反向后 ReduceScatter 梯度 | 每层参数量 × β | 每 micro-batch 3 次/层 | 预取可隐藏，第一次 AG 和最后一次 RS 暴露 |
| TP（Megatron 式） | 层内矩阵（列切 + 行切） | 每层 Attention 后、MLP 后各一次 AllReduce（前向）；训练反向再各一次 | b·s·h·β | 前向 2L 次 | **是**，与计算串行依赖 |
| SP（配合 TP） | 序列（LayerNorm/Dropout 等逐 token 算子） | 把 TP 的 AllReduce 拆为 ReduceScatter + AllGather | 同 TP，但激活显存 /d | 同 TP | 是，但可与 GEMM 融合重叠 |
| PP | 层 | stage 间点对点 Send/Recv 激活（反向传梯度） | b·s·h·β / micro-batch 数 | 每 micro-batch 每边界一次 | 气泡是主问题，单次传输一般不大 |
| EP（MoE） | 专家 | dispatch：All-to-All 把 token 发到专家所在卡；combine：反向 All-to-All 加权合并 | ≈ 每 token top-k × h × β，按目的卡不均匀 | 每 MoE 层 2 次（前向） | **是**，且受负载不均（skew）放大 |
| CP（长上下文） | 同一序列的注意力 | Ring 传递 K/V（P2P），或 AllGather K/V，或合并 softmax 统计量 | 每块 K/V | 每层 d-1 轮 | 可与块内注意力计算重叠 |
| 推理 DP-Attention + EP（DeepSeek 式） | attention 按请求 DP，MoE 按专家 EP | attention 后 gather/scatter token 到 MoE 组，再 All-to-All | 变长 | 每层 | 是 |

把上表的前向通信标到一个 transformer 层（Attention 子块 + MLP/MoE 子块）上，看每种并行在哪个位置"欠"通信：

```text
  from stage s-1 ──► PP: Recv(activation)                      [P2P]
          │
 ┌────────▼─────────────────────┐
 │ Norm                         │     SP: Norm 只算本卡的序列分片
 │ QKV proj   (column-parallel) │ ◄── SP: 先 AllGather 补全序列；TP 按头切，无通信
 │ core attn                    │ ◄── CP: 块间 Ring P2P 传 K/V（d-1 轮）
 │ O proj     (row-parallel)    │ ──► TP: AllReduce 部分和（SP: 改为 ReduceScatter）
 └────────┬─────────────────────┘
         (+) residual
 ┌────────▼─────────────────────┐
 │ Norm                         │     SP: 同上，按序列分片
 │ dense MLP:                   │
 │   gate/up  (column-parallel) │ ◄── SP: AllGather
 │   down     (row-parallel)    │ ──► TP: AllReduce（SP: ReduceScatter）
 │ ---- or MoE: --------------- │
 │   router -> top-k            │
 │   dispatch                   │ ◄─► EP: All-to-All，token 发往专家所在 rank
 │   expert FFN (grouped GEMM)  │
 │   combine                    │ ◄─► EP: All-to-All，送回并按 top-k 权重求和
 └────────┬─────────────────────┘
         (+) residual
          │
          ▼
  to stage s+1 ──► PP: Send(activation)                        [P2P]

  不在单层前向里的：DP 训练在反向时按桶 AllReduce 梯度；ZeRO-3/FSDP 每层前向前、
  反向前 AllGather 参数，反向后 ReduceScatter 梯度；DP-Attention 在 MoE 前 gather、后 scatter。
```

书中对应位置：
- 并行方式一览（切分维度表）：`third_party/ai-infra-book/manuscripts/06-超节点.md` 6.1.4。
- TP 每层两次 AllReduce、Qwen3-32B 64 层 → 每步 128 次 AllReduce，并给出 `T(p)=T_local/p + 128·T_AR(p)`：同文件 6.2.2。
- SP 把 AllReduce 拆成 RS + AG：6.2.3。CP 的 softmax 统计量合并与 zigzag 负载均衡：6.2.4。PP 气泡 `η=b/(q+b−1)`：6.2.5。EP dispatch/combine 均为 All-to-All、按目的卡去重可把流量减半：6.2.6。
- ZeRO 0~3 每卡显存 `16N, 4N+12N/d, 2N+14N/d, 16N/d`：`third_party/ai-infra-book/manuscripts/10-训练系统.md` 10.2.1。
- ZeRO-3 每 micro-batch 三次集合通信、每次每卡收发 `(d−1)/d·2N`，只有第一次 AG 与最后一次 RS 无法隐藏（MegaScale 结论）：同文件 10.3.3。
- EP 通信量 dispatch `nkfd`、dispatch+combine `2nkfd`，combine 是 dispatch 的转置矩阵：`third_party/ai-infra-book/manuscripts/07-数据中心网络.md` 7.2.3。
- 书中还有可运行的通信量计算器：`third_party/ai-infra-book/calculations/`（如 `calculations/src/infra_calc/topics/dense_communication.py`、`all_to_all.py`、`training_state.py`、`pd_af_handoff.py`），以 `python3 calculations/calc.py <topic>` 运行（用法见 `calculations/README.md`）。

### 2. 消息大小直觉（自己算一遍）

- TP AllReduce 单次 = b·s·h·β。h=4096、BF16：单 token 8 KiB；8192 token 的 prefill 为 64 MiB——同样的层数和次数，消息相差 8192 倍（`06-超节点.md` 6.1.3）。
- Ring AllReduce 每卡发送 `2(d−1)M/d`，`T_ring = 2(d−1)α + 2(d−1)M/(dB)`；与树算法交叉点在书中参数下约 680 KiB（6.4.2）。
- α-β 平衡点：书中 8 卡环、每轮启动 5 μs、50 GB/s 时，载荷项等于启动项的 M≈2 MB。8 KiB 的 decode AllReduce 几乎全是启动开销：72 次串行 AllReduce 把带宽提高 3 倍只省 ~14 μs，把每轮启动从 5 μs 降到 2 μs 却省 ~3.0 ms（`07-数据中心网络.md` 7.6.3）。

### 3. 训练 vs 推理（prefill / decode）

| 维度 | 训练 | Prefill | Decode |
|---|---|---|---|
| TP 消息 | 大（MiB 级），前后向都有 | 大（MiB 级） | 小（KiB 级，b×h×β） |
| 瓶颈 | 带宽 + 重叠 | 带宽 | 延迟（α）、kernel 启动、同步 |
| 主要原语 | AllReduce/RS/AG（DP/FSDP）、TP AR、PP P2P、EP A2A | TP AR、EP A2A（高吞吐模式） | TP AR（custom AR/one-shot）、EP A2A（低延迟模式） |
| 是否 CUDA Graph | 一般否 | 视实现 | 是（要求通信可被 capture、地址固定） |
| 典型优化 | 分桶、重叠、分层 AllReduce、SHARP | 通算重叠、FP8 dispatch | IPC/NVLink one-shot、multimem、AR+RMSNorm 融合、双 batch 重叠 |
| 尾延迟 | 掉队者决定步长 | TTFT | TPOT/p99，一次慢通信放大到每个 token |

框架里的佐证：
- SGLang `DeepEPMode.resolve(is_extend_in_batch)` 在 AUTO 模式下 prefill(extend) 用 normal、decode 用 low_latency：`third_party/sglang/python/sglang/srt/layers/moe/utils.py`（`class DeepEPMode`）。
- vLLM custom allreduce 的 one-stage/two-stage 选择按字节阈值（ws≤4 且 <512 KiB，或 ws≤8 且 <256 KiB 用 one-stage）：`third_party/vllm/csrc/custom_all_reduce.cuh`（`REDUCE_CASE` 宏）。

### 4. PD 分离与 KV cache 传输

- 动机：prefill 计算密集、decode 访存/延迟密集，同机混跑互相干扰；分离后 P 实例算完 prompt，把 KV cache 交给 D 实例生成（`09-分布式推理.md` 9.2.1，DistServe/Splitwise）。
- 数据量：GQA `V_KV = 2·n·L·h_kv·d_h·b`。Qwen3-8B（36 层、8 KV 头、128 维、BF16）为 144 KiB/token，8192 token 共 1.125 GiB，25 GB/s 下约 48 ms；MLA（DeepSeek-V3）为 `n·L·(d_c+d_r)·b`，约 68.6 KiB/token（9.2.2）。
- 实例配比约束：`μ_PD = min(μ_P, μ_D, B_net / V_KV)`——网络带宽直接进入系统吞吐上限（9.2.3）。
- 通信特征：点对点、跨实例（不同 world）、按 KV block 寻址、可按层流水（P 算完一层就开始传）、需要完成标志与双端显存驻留。实现上走 RDMA 传输引擎（NIXL、Mooncake TransferEngine 等），不走 NCCL 集合通信（见 [02](02-vllm-communication.md) KV connector、[03](03-sglang-communication.md) disaggregation）。
- 与 CCL 的关系：这类“GPU 显存 ↔ 远端 GPU 显存、一对一、动态成员”的传输是 RDMA 层的直接应用；要关心 GPUDirect RDMA、MR 注册、多网卡聚合、与 EP/TP 流量的隔离（SL/VL）。

## 源码导读（路径 + 函数 + 作用）

| 并行方式 | 框架代码入口 | 作用 |
|---|---|---|
| TP | `third_party/vllm/vllm/model_executor/layers/linear.py` `RowParallelLinear.forward` | `reduce_results and tp_size > 1` 时调用 `tensor_model_parallel_all_reduce` |
| TP（gather） | 同文件 `ColumnParallelLinear.forward` | `gather_output` 时 `tensor_model_parallel_all_gather` |
| 进程组布局 | `third_party/vllm/vllm/distributed/parallel_state.py` `initialize_model_parallel` | 注释 `ExternalDP x DP x PP x PCP x TP`，TP 是最内层（连续 rank → 同机 NVLink） |
| EP | `third_party/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py` `FusedMoEPrepareAndFinalize` | prepare = dispatch，finalize = combine |
| DP-Attention | `third_party/sglang/python/sglang/srt/layers/dp_attention.py` `_dp_gather` / `dp_scatter` | attention-DP 与 MoE 组之间的 token 搬运 |
| SP 式 RS/AG | `third_party/sglang/python/sglang/srt/layers/layer_boundary/ops.py` `tp_reduce_scatter` / `attn_tp_gather_input` | 层边界上把 AllReduce 换为 RS + AG |
| PD KV | `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py` `KVConnectorBase_V1`；`third_party/sglang/python/sglang/srt/disaggregation/base/conn.py` `BaseKVSender` | KV 传输抽象 |

按 vLLM `initialize_model_parallel` 的 `torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)` 手算一个小例子，看各组落在哪些 rank 上（world=8，dp=2、pp=2、pcp=1、tp=2，假设每节点 4 卡）：

```text
 rank = dp_i*4 + pp_i*2 + tp_i            （TP 是最内层，变化最快）

               ┌────── node 0 (dp_i=0) ──────┐   ┌────── node 1 (dp_i=1) ──────┐
  pp_i = 0     │      r0 ═════ r1            │   │      r4 ═════ r5            │
               │      │        │             │   │      │        │             │
  pp_i = 1     │      r2 ═════ r3            │   │      r6 ═════ r7            │
               └─────────────────────────────┘   └─────────────────────────────┘
                      ═══ TP 组    │ PP 组

  TP : [0,1] [2,3] [4,5] [6,7]      ← 相邻 rank，同机 NVLink（每层 2 次串行 AllReduce）
  PP : [0,2] [1,3] [4,6] [5,7]      ← 同一 dp 副本内相邻 stage
  DP : [0,4] [1,5] [2,6] [3,7]      ← 跨节点（推理 DP 无模型内通信）
  EP : [0,1,4,5] [2,3,6,7]          ← 同一 pp stage 内 DP×PCP×TP 拼成（仅 MoE 模型）
```

## 性能要点与常见坑

- TP 必须放在 NVLink 域内：rank 布局里 TP 放最内层就是为此。TP 跨节点后每层两次串行 AllReduce 直接走 RDMA，decode 延迟会显著恶化。
- TP 大于 KV 头数后 KV 会被复制（`06-超节点.md` 6.2.2），显存收益消失但通信不减。
- EP 的瓶颈经常不是平均带宽而是最热的那张卡（skew）：书中给出 EP8 热点专家 8% skew、EP256 达 4×（`09-分布式推理.md` 9.4.1）；All-to-All 的完成时间由最慢的收/发方决定。
- ZeRO-3 的通信量约为 DP 的 1.5 倍（2 次 AG + 1 次 RS 对比 1 次 AR=RS+AG），但每次都可以和层计算重叠；暴露的部分集中在首尾。
- 训练与推理共用同一套 NCCL，但调优目标相反：训练调大带宽（channel 数、buffer），decode 调小延迟（LL 协议、one-shot、少 channel 占 SM）。
- 只看平均：p99 由偶发的慢就绪或恢复事件决定（`07-数据中心网络.md` 7.6.3 尾延迟例子）。

## 面试自测

1. 问：一个 32 层、h=8192、TP=8 的 dense 模型，decode batch=64、BF16，单步前向 TP 通信多少次、每次多大？
   要点：每层 2 次 AllReduce → 64 次；每次 64×8192×2 B = 1 MiB；Ring 下每卡发送 2·7/8·1 MiB ≈ 1.75 MiB。次数多、单次中等，启动开销与带宽都要看。
2. 问：为什么说 SP 不增加通信量？
   要点：AllReduce = ReduceScatter + AllGather，SP 只是把这两半放到不同算子边界，总字节相同，换来 LayerNorm/Dropout 激活按序列分片的显存收益，并给 GEMM 重叠创造机会。
3. 问：ZeRO-3 相比 DDP 通信量是多少倍、为什么还能用？
   要点：约 1.5 倍；参数 AllGather 可预取、梯度 RS 逐层发出，大部分与计算重叠，只有第一层 AG 和最后一层 RS 暴露。
4. 问：MoE 的 All-to-All 和 TP AllReduce 在通信特征上有什么本质区别？
   要点：A2A 每对 rank 消息大小由路由动态决定、不均匀、事先未知（需要先交换计数/布局）；不做归约；完成时间受最热卡支配；dispatch 和 combine 互为转置。AllReduce 消息规则、可分段流水、各 rank 对称。
5. 问：decode 阶段 TP AllReduce 慢，加带宽有用吗？
   要点：基本没用，消息 KiB 级、启动开销主导；应减少同步次数（融合 AR+Norm）、用 one-shot IPC/multimem、缩短发起路径、用 CUDA Graph 消除 launch 开销。
6. 问：PD 分离后 KV 传输带宽怎么估？
   要点：`V_KV=2nL·h_kv·d_h·b` 乘以 prefill 吞吐（token/s），与网卡带宽比较；`μ_PD = min(μ_P, μ_D, B_net/V_KV)`；可按层流水与 prefill 计算重叠。

## 待深入

- 待确认：Megatron-LM 中 TP/SP 通信与 GEMM 重叠（如 tp-comm-overlap）的具体实现；确认方式：克隆 NVIDIA/Megatron-LM 后查 `megatron/core/tensor_parallel/` 与 `megatron/core/model_parallel_config.py`。
- 待确认：PyTorch FSDP2 的 AllGather/ReduceScatter 调度与预取实现；确认方式：克隆 pytorch/pytorch 后查 `torch/distributed/fsdp/`。
- 用 `third_party/ai-infra-book/calculations/` 的 `dense_communication.py`、`all_to_all.py` 对自己关心的模型实际算一遍通信账本。
