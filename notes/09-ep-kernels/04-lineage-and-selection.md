# 谱系、横向对比与选型

> 层次：中层 · 来源：third_party/DeepEP-hybrid-ep@10d4dd73、third_party/DeepEP@93eb6eb2（V2.5）、third_party/nccl@3e46c089（`contrib/nccl_ep`）、third_party/vllm@02653fa2、third_party/sglang@d2a43644

## 一句话

EP 通信库经历了三步：DeepEP V1 用 NVSHMEM 给出 normal（分层、高吞吐）和 LL（直连、低延迟）两种 kernel；NVIDIA 在 `hybrid-ep` 分支用 TMA + 路由预计算重写了高吞吐路径（Hybrid-EP）；随后 NCCL EP 把 LL 和 Hybrid-EP 都移植到 NCCL 的 GIN/LSA 上，DeepEP V2.5 也改建在 NCCL device API 上，并把两种模式统一成 `EPBuffer`。**算法思想没有变，变的是"谁来驱动网卡"和"接口统一到哪一层"。**

## 谱系图

```text
                         DeepEP V1（deepseek-ai，NVSHMEM + IBGDA）
                ┌───────────────┴────────────────┐
         normal（intranode.cu / internode.cu）   low-latency（internode_ll.cu）
         分层：RDMA→同号GPU→NVLink               直连：每 (token,专家) 一次 RDMA/NVLink 写
         notify 交换计数 + 队列，CPU 等待         固定槽位 + 同 QP 计数，CUDA Graph 友好
                │                                         │
     Hybrid-EP（NVIDIA，hybrid-ep 分支）                    │
     路由 AllGather + scan 预计算落点                       │
     TMA warp 专业化，DOCA GPUNetIO / NIXL                  │
                │                                         │
                ▼                                         ▼
     NCCL EP HT 模式                               NCCL EP LL 模式
     contrib/nccl_ep/device/hybrid_ep.cuh          contrib/nccl_ep/device/ll_ep.cuh
     （文件头：adapted from DeepEP）                （ncclGin::put + ncclGin_SignalAdd）
                └─────────────── 都建在 NCCL GIN（RDMA）/ LSA（NVLink）上 ─┘

     DeepEP V2.5（deepseek-ai，main）：移除 V1 与 NVSHMEM，统一为 EPBuffer，
     底层走 NCCL GIN/LSA；多节点有 "direct" 与 "hybrid"（RDMA + NVLink 分层）两种模式；
     不再提供"零 SM 的 RDMA EP"（README：zero-SM RDMA EP is not supported）
```

## 名字辨析："hybrid" 指三种不同的东西

| 叫法 | 所指 | 位置 |
|---|---|---|
| **Hybrid-EP** | NVIDIA 写的整套高吞吐实现（本章 [03](03-hybrid-ep.md)） | `third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/`，上游 `hybrid-ep` 分支；V2.5 README 的致谢列表也链接到这个分支 |
| **NCCL EP 的 hybrid_ep** | Hybrid-EP 在 NCCL GIN/LSA 上的移植，对应 NCCL EP 的 `NCCL_EP_ALGO_HIGH_THROUGHPUT` | `third_party/nccl/contrib/nccl_ep/device/hybrid_ep.cuh`、`include/ep_enums.h` |
| **DeepEP V2.5 的 hybrid mode** | `EPBuffer(allow_hybrid_mode=True)`：多节点时用"RDMA + NVLink 分层"而不是 direct；docstring 说它带宽更高、对多平面/多 rail 网络更友好 | `third_party/DeepEP/deep_ep/buffers/ep.py`、`third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh` |

三者共同的含义是"**分层**：跨节点 RDMA + 节点内 NVLink"。面试时如果对方只说 "hybrid"，先确认指的是哪一个。

## 横向对比

| 维度 | V1 normal | V1 LL | Hybrid-EP | NCCL EP（LL / HT） | DeepEP V2.5 |
|---|---|---|---|---|---|
| 目标工况 | 训练、prefill | decode | 训练、prefill | decode / 训练与 prefill | 统一接口，覆盖两者 |
| 拓扑 | 分层（RDMA→同号 GPU→NVLink） | 直连（节点内可走 NVLink 旁路） | 分层（同 V1 normal） | LL 直连 / HT 分层 | direct 或 hybrid |
| 落点决定 | notify 交换计数，接收端分配 | 发送端本地 `atomicAdd` 槽位 | 全局路由 AllGather + scan | LL 同 V1 LL / HT 同 Hybrid-EP | 待确认（见 V2.5 `dispatch.cuh`/`hybrid_dispatch.cuh` 中的 rank 去重与 slot 分配） |
| 去重 | 按节点、按 rank | 不去重（每个专家一份） | 跨节点按节点、节点内按 rank | 同对应原版 | 有 rank 级去重（源码注释 "Rank choice should do deduplication"） |
| CPU 同步 | 有（等 `moe_recv_counter`） | 无 | 默认有（拿精确计数）；`non_blocking` 时无 | 视模式 | `do_cpu_sync` 可选 |
| CUDA Graph | 不兼容（单节点 `num_worst_tokens` 例外） | 兼容 | `non_blocking` 时兼容 | 兼容 | `do_cpu_sync=False` 时兼容 |
| 网卡驱动 | NVSHMEM（IBRC/IBGDA） | NVSHMEM IBGDA | DOCA GPUNetIO（或 NIXL/UCX） | NCCL GIN（GDAKI 或 Proxy 后端） | NCCL GIN |
| NVLink 搬运 | warp LD/ST | warp LD/ST | TMA | LSA（load/store、TMA） | LSA + TMA |
| 典型 SM | ~20~24 | 整卡、短时；hook 期间 0 | 4~16 | 待确认 | 解析式估算，可覆盖 |
| 依赖 | NVSHMEM | NVSHMEM | DOCA 或 NIXL、rdma-core | NCCL ≥ 2.29、CUDA 13+（依 NCCL EP README） | NCCL ≥ 2.32.3 |

## 框架里怎么选

| 框架 | 配置 | 对应 |
|---|---|---|
| vLLM | `--all2all-backend`（`third_party/vllm/vllm/config/parallel.py` 的 `All2AllBackend`）：`deepep_high_throughput` / `deepep_low_latency` / `deepep_v2`，默认 `allgather_reducescatter` | `DeepEPHTAll2AllManager`、`DeepEPLLAll2AllManager`、`DeepEPV2All2AllManager`（`device_communicators/all2all.py`） |
| SGLang | `DeepEPMode`：`normal` / `low_latency` / `auto`；`auto` 时 batch 中有 extend（prefill）就用 normal，否则用 low_latency（`third_party/sglang/python/sglang/srt/layers/moe/utils.py` 的 `DeepEPMode.resolve`） | `layers/moe/token_dispatcher/deepep.py` |
| 训练 | Hybrid-EP 面向训练（支持 `dispatch_with_permute`、反向的 `BACKWARD_COMBINE`） | 待确认：Megatron-LM 接入点 |

PD 分离部署下，prefill 实例用高吞吐 kernel、decode 实例用 LL，两边可以各自选最合适的 kernel，这是 PD 分离在通信层面的一个好处。

## 选型决策

```text
每 rank token 数小（decode，≲ 256）、要 CUDA Graph、要和 attention 重叠？
   └─ 是 → LL（V1 LL / NCCL EP LL / V2.5 + do_cpu_sync=False）
            显存够吗？ 接收缓冲 ≈ 2 × num_experts × num_max × msg_bytes
否（prefill / 训练，数千 token）
   ├─ 整个 EP 组在一个 NVLink 域内（单机 8 卡、NVL72）？ → 节点内路径，带宽看 NVLink，关注 SM 占用（Hybrid-EP 单节点 / V2.5）
   └─ 跨节点 → 分层（Hybrid-EP / NCCL EP HT / V2.5 hybrid）
                rail-optimized 组网 + 同号 GPU 转发；关注每 token 选中的远端节点数 m̄
```

## 对自研通信库的启示（面试可展开）

1. **落点预计算是性能和复杂度的分水岭**：能预先知道落点，就能用 WRITE + 固定布局 + 同 QP 标志，不需要队列；做不到就要队列和流控。Hybrid-EP 用一次很小的 AllGather 换来了这个前提。
2. **"同 QP、数据在前、标志在后"依赖 RC 按序**：如果自研网卡或协议支持多路径喷洒、乱序放置（比如前面讨论过的选择性重传、UET 的 RUD 语义），这个假设就不成立，需要在协议层提供"带完成语义的 fence"或"WRITE_WITH_IMM 计数完成"。这正是网络团队和通信库团队需要共同设计的地方。
3. **GPU 直驱网卡的接口要能抽象**：IBGDA、DOCA GPUNetIO、NCCL GIN、NIXL GDA 做的是同一件事（GPU 上组 WQE、敲门铃）。NCCL GIN 同时提供 GDAKI 和 Proxy 两个后端，自研网卡也可以先接 Proxy 后端跑通，再做 GPU 直驱。
4. **SM 是稀缺资源**：TMA + warp 专业化让通信只占 4~16 个 SM；LL 的 hook 让等待期间不占 SM。评价 EP 库时，除了带宽和延迟，还要报告"用了多少 SM"。

## 面试自测

1. **问**：DeepEP V2.5 去掉了 V1 LL 的哪种能力？可能的原因是什么？
   **要点**：README 明确不再支持"零 SM 的 RDMA EP"，也就是 V1 LL 的 hook 式"只发出、稍后接收、中间不占 SM"。V2.5 统一建在 NCCL GIN 上，dispatch/combine 都需要 SM；可能的原因是简化接口、统一后端、借助 NCCL 的通信器与容错能力（推测，待确认：读 V2.5 的设计说明或 issue）。
2. **问**：NCCL EP 和 DeepEP 是什么关系？
   **要点**：NCCL EP 的 LL 与 HT 模式分别改编自 DeepEP LL 和 Hybrid-EP（源码文件头注明），把 NVSHMEM IBGDA / DOCA 换成 NCCL 的 GIN put/signal 与 LSA；接口变成 NCCL 风格的 `ncclEpCreateGroup`、`ncclEpDispatch` 等，LL 模式支持 `send_only` + `ncclEpComplete` 的分阶段执行，相当于 recv hook。
3. **问**：同样是分层，V1 normal 和 Hybrid-EP 的根本区别？
   **要点**：V1 normal 在 kernel 里用队列（head/tail）在 RDMA sender、forwarder、NVL receiver 之间流转，接收计数要先 notify 并让 CPU 等待；Hybrid-EP 先 AllGather 路由，所有落点在 kernel 启动前就确定了，kernel 只是按表搬运，配合 TMA 和 chunk 标志流水执行。

## 待深入

- 待确认：DeepEP V2.5 `hybrid_dispatch.cuh` 中 scale-out 与 scale-up 两级之间的 slot 分配与 channel linked list 机制；确认方式：精读 `third_party/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh` 和 `deep_ep/buffers/ep.py` 中 `token_metadata_at_forward`、`channel_linked_list` 的说明。
- 待确认：NCCL EP 的 SM 占用与 V1/Hybrid-EP 的同平台对比；确认方式：在 GPU 环境运行 `third_party/nccl/contrib/nccl_ep/ep_bench.cu` 与 DeepEP 自带测试。
