# 一次完整推理：PD 分离 + AF 分离

> 层次：上层 · 来源：third_party/vllm@02653fa、third_party/sglang@d2a4364、third_party/DeepEP@93eb6eb、third_party/ai-infra-book@3bdcb4fc 第 7、9 章；外部结论引用 [调研报告](../../docs/references/2026-10-14-inference-parallelism-ep-research.md) 的 F2、F13、G4，以及 [MegaScale-Infer 阶段调研](../../docs/references/2026-10-17-megascale-infer-phase.md)

## 一句话

以 MoE 模型为对象：一次请求先在 P 节点上把整个 prompt 并行过完 L 层，得到第 1 个输出 token 和每层一份 KV cache；KV cache 经 RDMA 传到 D 节点后，D 节点每步只输入 1 个新 token。AF 分离用在 **decode** 阶段：D 节点里每一层的 attention 和 FFN（expert）分别跑在两组 GPU 上，hidden state 在两组之间每层来回传一次。

## 在调用链中的位置

```text
用户请求 → 网关/路由（选 P 实例、D 实例）
   ├─ P 实例：prefill（整段 prompt，compute-bound）
   │     └─ KV 传输：vLLM KVConnector / SGLang KVSender → Mooncake / NIXL → RDMA
   └─ D 实例：decode（每步 1 token，memory-bound）
         ├─ 常规部署：每张卡同时跑 attention 和 FFN，MoE 层走 EP dispatch/combine
         └─ AF 分离：attention 组 ⇄ FFN 组，每层 M→N、N→M 各一次（MegaScale-Infer 的 M2N）
```

PD 分离解决"prefill 和 decode 抢同一组 GPU"的问题；AF 分离再把 decode 内部拆开，解决"attention 和 FFN 需要的资源不同"的问题。两者互相独立，可以只用其中一个。

### AF 分离的两个前提：MoE 模型、decode 阶段

| 前提 | 原因 |
|---|---|
| **MoE 模型** | dense 模型的 FFN 每个 token 都要算，batch 和 attention 一样大，拆出去收益很小。MoE 的问题是稀疏激活：每个 expert 只分到 `B × k / E` 个 token（E 是 expert 总数），GEMM 太小、读权重的开销摊不开。AF 分离让 M 个 attention 节点的 token 汇聚到同一组 expert 节点，把每个 expert 的 batch 做大。 |
| **decode 阶段** | MegaScale-Infer 论文原文："we focus on the decoding phase"（[调研 M1、M2](../../docs/references/2026-10-17-megascale-infer-phase.md)）。decode 时 attention 读 KV cache、FFN 读 expert 权重，两边都是访存密集、GPU 利用率低，拆开才能分别扩缩容。prefill 一次就有成千上万个 token，expert GEMM 本来就够大、是 compute-bound，拆开没有收益；论文里 prefill 走单独的集群（M3）。 |

容易反过来理解的一点：AF 分离确实给 decode 每层加了 2 次通信，但论文的做法是用 ping-pong micro-batch 把通信藏在计算后面，换来 expert 利用率的提升。所以它是"decode 专用"的优化，而不是"因为 decode 延迟敏感所以不用"。

## 核心概念

### 1. 先纠正几个常见误解

| 误解 | 实际情况 |
|---|---|
| embedding 是"跟模型的长度对齐" | embedding 是查表：每个 token id 映射成一个 d 维向量（d = hidden size）。对齐的是隐藏维度 d。prompt 有 N 个 token 就是 N 行，不用补齐到固定长度。 |
| Q、K、V 都存进 KV cache | 只存 K 和 V。Q 只服务于当前 token 的输出，用完就丢。 |
| attention 算完要发给 FFN 节点，FFN 算完再发回 | 常规部署里两者在同一组 GPU 上顺序执行，不走网络。只有 AF 分离架构才真正"发过去再发回来"。 |
| 最后一层算出"最终的 KV cache" | KV cache 按层存，每层一份，没有"最终的 KV cache"。最后一层输出的是 hidden state，经 lm_head 采样出 token。 |
| decode 每步要输入之前的所有 token | 每个请求每步只输入 1 个新 token；历史 token 以 KV cache 的形式在 attention 里被读取。 |

### 2. 每层的结构：两个子块串联

```text
h ──► Norm ──► Attention ──► (+) ──► Norm ──► FFN / MoE ──► (+) ──► 下一层
│                             ▲      │                       ▲
└────────── 残差 ─────────────┘      └─────── 残差 ──────────┘
```

- **Attention 子块**：唯一让 token 之间交换信息的地方，要读其他 token 的 K/V。它的状态（KV cache）随序列变长。
- **FFN 子块**：每个 token 独立计算，token 之间不交互，没有状态，只有权重。dense 层是一个 FFN；MoE 层是 router 加一组 expert，每个 expert 是一个 FFN（现代模型多为 SwiGLU：`W_down(SiLU(W_gate·x) ⊙ W_up·x)`）。FFN 里没有 softmax，softmax 出现在 attention、router（部分模型用 sigmoid）和输出层。
- AF 分离能成立的前提正是这条差异：attention 有状态、随上下文增长；FFN 无状态、只看权重。两者的资源需求不同，所以可以拆开、各自扩缩容。

### 3. 为什么 K/V 能复用、Q 不用存

第 t 个 token 在某一层的输出：

```text
o_t = Σ_{i≤t} softmax_i( q_t · k_i / √d ) · v_i
```

- 只需要**当前 token 的 q_t**，所以历史 q 不存。
- 需要**全部历史的 k_i、v_i**，所以缓存。
- 历史 K/V 不会变：causal mask 下 token i 只看 1..i，逐层归纳可知它在每一层的 hidden state 都只依赖 1..i，后续 token 进来不影响它。BERT 那种双向 attention 就没有这个性质，KV cache 也就不成立。

### 4. Dispatch 与 Combine：MoE 层里的两次搬运

**它们是什么。** MoE 层用 EP（Expert Parallel）部署时，不同 expert 放在不同 GPU 上，token 却分散在各个 GPU 上。于是每个 MoE 层要搬两次数据：

- **Dispatch（分发）**：router 给每个 token 选出 top-k 个 expert 后，把 token 的 hidden 向量**发到这些 expert 所在的 GPU**。同一个 token 选中的多个 expert 如果在同一张卡上，只发一份。
- **Combine（合并）**：各 GPU 算完本地 expert 后，把结果**发回 token 原来所在的 GPU**，再按 router 给的 gate 权重加权求和，还原成每个 token 一个 d 维输出。

```text
两张卡，EP = 2：GPU0 持有 E0、E1，GPU1 持有 E2、E3；top-k = 2
token a 在 GPU0，router 选 {E0, E2}

        GPU0                                   GPU1
 ┌────────────────┐                     ┌────────────────┐
 │ a  router→E0,E2│                     │                │
 │                │── dispatch: a ─────►│ E2(a)          │  ← a 跨卡发一份
 │ E0(a)          │   （E0 在本卡，不用发）│                │
 │                │◄── combine: E2(a) ──│                │  ← 结果发回 a 的来源卡
 │ out_a = g0·E0(a) + g2·E2(a)          │                │
 └────────────────┘                     └────────────────┘
所有 token 同时这样做，就是两卡之间双向的 All2All-v。
```

- 通信形态是 **All2All-v**：每张卡都可能给每张卡发数据，而且每对卡之间发多少由这一批 token 的路由决定，每次都不一样。
- 数据量：单层 dispatch + combine 约 `2·n·k·f·d`，n 是 token 数，k 是 top-k，f 是跨卡/跨机比例，d 是每份激活的字节数（书 §7.2.3，详见[学习入口](../00-roadmap/00-start-here.md)第 3.2 节）。DeepSeek 的做法是 dispatch 用 FP8、combine 用 BF16。
- 它不是 NCCL 的标准集合通信：路由动态、要先交换"每个 rank 发多少"、还要和 expert GEMM 重叠。所以有 DeepEP 这类专用库，本地 V2.5 的入口是 `third_party/DeepEP/deep_ep/buffers/ep.py` 的 `EPBuffer.dispatch`（L562）和 `EPBuffer.combine`（L770）。

**什么时候用到。**

| 场景 | 是否有 dispatch/combine | 说明 |
|---|---|---|
| dense 层（attention、dense FFN） | 没有 | 没有 expert，不用路由 |
| MoE 层，EP = 1（所有 expert 在一张卡上） | 没有跨卡通信 | 只在卡内按 expert 重排 token（permute / unpermute） |
| MoE 层，只用 TP 切 expert | 没有 | 每张卡持有所有 expert 的一个分片，靠 AllReduce 汇总 |
| **MoE 层 + EP，prefill** | **有，每个 MoE 层各一次** | token 多（成千上万），消息大，追求带宽；DeepEP normal 模式 |
| **MoE 层 + EP，decode** | **有，每个 MoE 层各一次** | 每请求 1 个 token，消息小，追求延迟；DeepEP low-latency 模式 |
| **AF 分离的 decode** | **有，形态变成非对称** | attention 组 → FFN 组相当于 dispatch（M→N），FFN 组 → attention 组相当于 combine（N→M） |

一个 L 层、其中 L_moe 层是 MoE 的模型，每生成一个 token（或每次 prefill）要做 L_moe 次 dispatch 和 L_moe 次 combine，全部在关键路径上。这就是 EP 通信性能直接决定 MoE 推理延迟的原因。

## 完整流程

### 总图

```text
                  ┌───────────── P 实例（prefill）─────────────┐
prompt (N token) ─►│ Embedding → L 层 × [Attn + FFN/MoE] → lm_head │──► 第 1 个 token（返回给用户）
                  │ 每层产出 KV cache[l]                        │
                  └───────────────┬────────────────────────────┘
                                  │ ② KV 传输：L 层 × N token，RDMA，可按层流水
                  ┌───────────────▼──────────── D 实例（decode，AF 分离）─────────────────┐
                  │  Attention 组（M 个节点）              FFN/Expert 组（N 个节点）        │
                  │  持有 KV cache、attention 权重          持有 expert 权重                │
                  │      │  每层：hidden ──── M→N ────►  expert FFN                         │
                  │      │        (+)  ◄──── N→M ────── 结果                               │
                  │  多个 micro-batch ping-pong 交替，让两组 GPU 都不闲着                  │
                  └────────────────────────────────────────────────────────────────────┘
                                  │ 每步 1 个 token，循环到 EOS
                                  ▼
                              流式返回给用户
```

### 阶段一：Prefill（P 实例，常规 EP 部署、不做 AF 分离；以 prompt = `[我, 爱, 北, 京]`、N = 4 为例）

```text
prompt: [我, 爱, 北, 京]
   │ Embedding 查表
h⁰: [4, d]
   │
┌──▼──────────────────────── 第 l 层（l = 1..L）────────────────────────┐
│ ① RMSNorm(h)                                                          │
│ ② Q = h·W_qˡ  K = h·W_kˡ  V = h·W_vˡ          → 4 个 token 一起投影   │
│    对 Q、K 加 RoPE 位置编码                                           │
│ ③ KV cache[l] ← K, V（4 个 token）            ★ 每层各存一份，Q 不存  │
│ ④ softmax(Q·Kᵀ/√d + causal mask)·V             → 4×4 注意力矩阵       │
│ ⑤ ·W_o，残差 h = h + attn_out                                        │
│ ⑥ RMSNorm(h)                                                          │
│ ⑦ FFN，或 MoE：router 选 top-k → [EP dispatch] → expert FFN          │
│    → [EP combine]                                                     │
│ ⑧ 残差 h = h + ffn_out                                               │
└──┬────────────────────────────────────────────────────────────────────┘
   │ 走完 L 层
hᴸ: [4, d] → 只取最后一个位置 → Norm → lm_head → 采样 → "天"
```

- 每层都要重新算 QKV：每层权重不同，输入是上一层的输出。
- N 个 token 一次前向并行算完，全是大 GEMM，compute-bound，决定 TTFT。
- 产物：第 1 个输出 token，以及 L 层 × N token 的 KV cache。
- "天"自己的 K/V 这时还没算，要等 decode 第一步把它作为输入时才算出来。

### 阶段二：KV cache 传输（P → D）

- 数据量（GQA）：`V_KV = 2 · n · L · h_kv · d_h · b`。Qwen3-8B（36 层、8 个 KV 头、128 维、BF16）约 144 KiB/token，8192 token 共约 1.125 GiB；DeepSeek-V3 用 MLA 存压缩 latent，约 68.6 KiB/token（书 9.2.2，详见 [并行与通信模式](01-parallelism-and-comm-patterns.md) 第 4 节）。
- 通信特征：点对点、跨实例（P、D 不在同一个 NCCL world）、按 KV block 寻址、需要完成通知；一般不走 NCCL 集合通信，而是走 Mooncake / NIXL 这类 RDMA 传输引擎（F2、G4）。
- 可以按层流水：P 算完第 l 层就开始传 KV cache[l]，和后面层的计算重叠。
- 在 AF 分离的 D 实例里，KV cache 的接收方是 **attention 组**，因为只有 attention 读 KV cache；FFN 组不需要它。（这是从结构推出来的，MegaScale-Infer 的具体实现待确认，见文末。）
- 系统吞吐上限：`μ_PD = min(μ_P, μ_D, B_net / V_KV)`，网络带宽直接进入上限（书 9.2.3）。

### 阶段三：Decode（D 实例，AF 分离）

以生成"天"之后的下一步为例。输入只有 1 个 token "天"，position = 4，batch 里每个请求各 1 个，共 B 个。

```text
Attention 组                                          FFN / Expert 组
──────────────                                        ────────────────
Embedding: [B, d]
for l in 1..L:
  ① RMSNorm
  ② q, k, v = 投影（每个请求只算 1 个 token）
     RoPE（位置 = 当前长度）
  ③ KV cache[l] 追加 k, v（长度 4 → 5）
  ④ q × K_cache[l]ᵀ → softmax → × V_cache[l]
     （1×5 注意力，其中 4 个 K/V 是复用的）
  ⑤ ·W_o，残差
  ⑥ RMSNorm
  ⑦a router 打分、选 top-k
  ⑦b 按目标 expert 发送 hidden ═══ M→N ═══►  ⑦c 收齐各 attention 节点的 token
                                              ⑦d 本地 expert FFN（Grouped GEMM）
  ⑦f 收回并按 gate 权重加权  ◄═══ N→M ═══     ⑦e 结果发回 token 的来源节点
  ⑧ 残差
lm_head → 采样 → "安" → 作为下一步输入
```

- 每层通信两次（M→N 和 N→M），L 层就是 2L 次，全部在关键路径上。
- **ping-pong 流水**：把 batch 切成多个 micro-batch。FFN 组算 micro-batch 1 时，attention 组算 micro-batch 2，传输也和计算重叠（F13）。要藏住通信，大致需要：attention 一段的耗时 ≈ FFN 一段的耗时，且单次传输耗时小于计算耗时。（这是按流水线常识推的约束，论文给出的具体条件待确认。）
- 和常规 EP 的区别：常规 EP 每张卡既跑 attention 又持有部分 expert，dispatch/combine 是对称的 all-to-all；AF 分离是 M 个 attention 节点对 N 个 expert 节点的**非对称**通信（[学习入口](../00-roadmap/00-start-here.md)第 3、4 节已有对比）。

## Token 用量与"算/复用"对照表

设 decode 第 t 步时，序列里已有 t 个 token。

| 环节 | Prefill（N 个 token） | Decode 一步（每请求） | 在 AF 分离里跑在哪 |
|---|---|---|---|
| Embedding | N 个 | 1 个 | attention 组 |
| Q 投影 | N 个，用完即丢 | 1 个，用完即丢 | attention 组 |
| K、V 投影 | N 个，全部写入 cache | 1 个，追加进 cache | attention 组 |
| 历史 K、V | 无 | t 个，**只读复用，不重算** | attention 组（持有 KV cache） |
| 注意力矩阵 | N × N（causal） | 1 × (t+1) | attention 组 |
| W_o、Norm、残差 | N 个 | 1 个 | attention 组 |
| Router | N 个 | 1 个 | attention 组（选完再发） |
| Expert FFN | N × top-k 份 | 1 × top-k 份 | FFN 组 |
| lm_head | 只算最后 1 个位置 | 1 个 | attention 组 |
| 瓶颈 | 计算（大 GEMM） | 读权重 + 读 KV cache | attention：读 KV cache；FFN：读 expert 权重 |

记一句话：decode 时所有"按 token 计算"的环节都只算新 token；唯一接触历史 token 的是 attention，而且只是读 cache。

## 通信清单

| 位置 | 发生在哪个阶段 | 通信 | 单次数据量（量级） | 关键路径 |
|---|---|---|---|---|
| TP 切分的 attention/FFN 之后 | P、D | AllReduce | `tokens × d × b` | 是 |
| MoE 层前后（常规 EP） | P、D | dispatch / combine（all-to-all） | 约 `tokens × top-k × d × b` | 是 |
| Prefill 结束 / 按层流水 | P → D | KV cache 点对点传输 | `2 · N · L · h_kv · d_h · b` | 影响首 token 之后的衔接，可与计算重叠 |
| 每层 attention ↔ FFN | D（AF 分离） | M→N、N→M 非对称传输 | 约 `B_mb × top-k × d × b`，每层各一次 | 是，靠 ping-pong 掩盖 |

prefill 的 dispatch/combine 一次搬成千上万个 token，追求带宽；decode 一次只有 B 个 token，追求延迟。DeepEP 因此分 normal 和 low-latency 两套 kernel（见 [DeepEP](04-deepep.md)、[EP kernel 专章](../09-ep-kernels/README.md)）。

## 源码导读（路径 + 函数 + 作用）

### PD 分离的 KV 传输

| 框架 | 位置 | 作用 |
|---|---|---|
| vLLM | `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py` 的 `KVConnectorBase_V1`（L181） | KV connector 抽象，分 scheduler 和 worker 两个角色（`KVConnectorRole`，L139） |
| vLLM | 同文件 `start_load_kv`（L333）、`wait_for_layer_load`（L350） | D 侧：开始异步加载 KV；前向到某层时等这一层的 KV 到位，实现按层流水 |
| vLLM | 同文件 `save_kv_layer`（L364）、`wait_for_save`（L386） | P 侧：在 attention 层内发起这一层 KV 的异步保存/发送；`wait_for_save` 每步调用一次，提交保存并做必要同步（异步保存时，源 block 要保留到完成为止） |
| vLLM | 同文件 `get_num_new_matched_tokens`（L516）、`request_finished`（L631） | scheduler 侧：查询有多少 token 的 KV 可以从外部加载；请求结束时决定是否延迟释放 KV block |
| vLLM | `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/`、`third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/` | 两种具体后端；nixl 目录下分 `push_*` / `pull_*` 两种模式，`tp_mapping.py` 处理 P、D 两侧 TP 不一致时的映射 |
| SGLang | `third_party/sglang/python/sglang/srt/disaggregation/base/conn.py` 的 `BaseKVSender`（L160）、`BaseKVReceiver`（L235） | P 侧 `send`、D 侧 `send_metadata`，双方用 `poll` 查 `KVPoll` 状态 |
| SGLang | `third_party/sglang/python/sglang/srt/disaggregation/prefill.py` 的 `SchedulerDisaggregationPrefillMixin`（L548） | P 侧调度循环，算完 prefill 后把请求放进 inflight 队列等传输完成 |
| SGLang | `third_party/sglang/python/sglang/srt/disaggregation/decode.py` 的 `DecodePreallocQueue`（L375）、`DecodeTransferQueue`（L2459） | D 侧先预分配 KV 空间，再等 KV 传完，才把请求放进 decode batch |
| SGLang | `third_party/sglang/python/sglang/srt/disaggregation/mooncake/conn.py` | 默认的 Mooncake 后端 |

### AF 分离

- 本地 vLLM@02653fa、SGLang@d2a4364 中没有找到 attention/FFN 分离部署的实现（按 `afd`、`AFD`、`attention_ffn` 搜索，无相关结果）。本节流程依据调研报告 F13、[MegaScale-Infer 阶段调研](../../docs/references/2026-10-17-megascale-infer-phase.md) 和上面的结构推理。

## 性能要点与常见坑

1. **PD 两侧并行配置不一致**：P 用 TP=8、D 用 TP=4 时，KV 的 head 切分不同，传输时要重排。vLLM nixl 后端用 `tp_mapping.py` 处理这件事。
2. **KV 传输和 prefill 重叠**：不按层流水时，传输时间直接加到"首 token 到第 2 个 token"的间隔上。
3. **D 侧显存预分配**：KV 还没到时就要先留好空间，否则传过来没地方放。SGLang 的 `DecodePreallocQueue` 就是做这件事。
4. **AF 分离的流水平衡**：attention 和 FFN 两段耗时差太多时，ping-pong 会有一侧空等。attention 的耗时随上下文长度涨，FFN 不涨，所以负载变化时配比要跟着调。
5. **AF 分离的通信次数**：每层 2 次、全在关键路径上，单次延迟比带宽更重要。这和 decode EP 的 low-latency 需求是同一类问题。
6. **只有 MoE 层值得拆**：dense FFN 层拆出去收益小（理由见上文"AF 分离的两个前提"）。DeepSeek 一类模型前几层是 dense 层（层数以模型 config 的 `first_k_dense_replace` 为准）。

## 面试自测

1. 问：prefill 每层都要算 QKV 吗？存什么？
   要点：每层都要，因为每层权重不同、输入是上一层输出；N 个 token 一起投影；只存 K、V，Q 用完即丢；KV cache 按层存。
2. 问：decode 每步输入几个 token？历史 token 怎么参与计算？
   要点：每个请求 1 个；投影、FFN、Norm 都只算新 token；历史 token 只在 attention 里以 KV cache 的形式被读取，不重算。
3. 问：为什么 K/V 能缓存、Q 不需要缓存？
   要点：`o_t` 只用 `q_t` 和全部历史 K/V；causal mask 保证历史 token 每层的 hidden 不变，所以 K/V 不变；历史 Q 只服务于已经算完的输出。
4. 问：PD 分离的 KV 传输为什么不走 NCCL？怎么估带宽？
   要点：点对点、跨实例、按 block 寻址、动态建连，NCCL 是固定成员的静态组；`V_KV = 2nL·h_kv·d_h·b`，乘以 prefill 吞吐后和网卡带宽比较，`μ_PD = min(μ_P, μ_D, B_net/V_KV)`。
5. 问：AF 分离的动机是什么？通信和常规 EP 有什么不同？
   要点：attention 有状态、读 KV cache；FFN 无状态、读权重，两者资源需求不同，拆开后可以独立扩缩容、用不同并行策略和异构硬件。常规 EP 是对称 all-to-all；AF 分离是 M→N、N→M 的非对称通信，每层两次，用 ping-pong micro-batch 掩盖。
6. 问：AF 分离的 D 实例里，P 传来的 KV cache 应该落在哪组 GPU？
   要点：attention 组，因为只有 attention 读 KV cache。
7. 问：AF 分离为什么用在 decode、只对 MoE 有意义？
   要点：decode 时每个 expert 只分到 `B·k/E` 个 token，GEMM 太小、访存密集；把多个 attention 节点的 token 汇聚到 expert 节点能把 batch 做大。prefill token 多，expert GEMM 本来就 compute-bound，拆开没收益。dense 模型的 FFN batch 和 attention 一样大，也没收益。

## 待深入

- 待确认：MegaScale-Infer 的 M2N 库怎样建连、怎样省掉 GPU→CPU 拷贝和同步，ping-pong 的 micro-batch 数怎样确定；确认方式：阅读 arXiv 2504.02263 第 4、5 节。
- 待确认：AF 分离的 D 实例接收 KV cache 的具体机制；确认方式：同上论文，或开源实现出现后对照源码。
- 待确认：vLLM nixl 后端 push 与 pull 模式的取舍；确认方式：读 `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py` 与 `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py`。
