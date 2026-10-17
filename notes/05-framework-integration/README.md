# 05 上层：框架如何调用通信库

> 层次：上层 · 来源：third_party/vllm@02653fa、third_party/sglang@d2a4364、third_party/DeepEP@93eb6eb、third_party/nccl@3e46c08、third_party/ai-infra-book@3bdcb4fc（第 6、7、9、10 章）

## 一句话

通信库（NCCL/DeepEP）的“用户”不是模型作者，而是框架的并行层：它决定什么时候、在哪个进程组、以多大消息、在哪条 stream 上发起哪种集合通信；做 CCL 优化必须先看懂这条调用链。

## 本章笔记

| 文件 | 主题 | 面向的问题 |
|---|---|---|
| [01-parallelism-and-comm-patterns.md](01-parallelism-and-comm-patterns.md) | DP/FSDP/ZeRO、TP、SP、PP、EP、CP 产生的集合通信；训练 vs prefill vs decode；PD 分离 KV 传输 | “这个并行策略会打出什么通信、多大、卡不卡关键路径” |
| [02-vllm-communication.md](02-vllm-communication.md) | vLLM `GroupCoordinator`、pynccl、custom allreduce、symm mem、all2all 后端、KV connector | “vLLM 里一次 TP AllReduce 到底走了哪条路径” |
| [03-sglang-communication.md](03-sglang-communication.md) | SGLang 通信层、DP attention、layer_boundary、DeepEP dispatcher、TBO、PD disaggregation | “SGLang 与 vLLM 的通信层哪里同源、哪里分叉” |
| [04-deepep.md](04-deepep.md) | DeepEP V1 normal / low-latency 概念，与本地 V2.5（NCCL Gin）代码对照 | “MoE All2All 为什么要专用库，它和 NCCL/NVSHMEM 什么关系”（kernel 细节见 [09-ep-kernels](../09-ep-kernels/README.md)） |
| [05-torch-distributed.md](05-torch-distributed.md) | ProcessGroup/ProcessGroupNCCL、comm 创建时机、stream/event、Work、watchdog、`TORCH_NCCL_*` | “hang 了/超时了，PyTorch 这一层发生了什么” |
| [06-pd-af-end-to-end.md](06-pd-af-end-to-end.md) | 一次请求走完 PD 分离 + AF 分离：每层 QKV/KV cache、prefill 与 decode 的 token 用量、KV 传输、attention ⇄ FFN 的 M2N 往返 | “一个 token 从输入到输出经过哪些计算和通信，哪些要算、哪些复用” |

## 上层调用链总图

```text
┌──────────────────────────────── 模型层 ────────────────────────────────┐
│ Attention / MLP / MoE / LM head                                       │
│  vLLM: layers/linear.py RowParallelLinear.forward → all_reduce        │
│        layers/fused_moe/modular_kernel.py → prepare/finalize          │
│  SGLang: layers/layer_boundary/ops.py, layers/dp_attention.py,        │
│          layers/moe/token_dispatcher/*.py                             │
└──────────────┬──────────────────────────────────────┬────────────────┘
               │ 普通集合通信                          │ MoE dispatch/combine
┌──────────────▼──────────── 并行层（进程组） ─────────▼────────────────┐
│ TP / PP / DP / EP / (D)CP / PCP / attn-TP / moe-EP ...                 │
│ vLLM:   parallel_state.py  get_tp_group() / get_ep_group() ...         │
│ SGLang: parallel_state.py  get_tp_group() / get_moe_ep_group() ...     │
│ 训练侧（概念）：DDP bucket AllReduce、FSDP AG/RS、PP send/recv           │
└──────────────┬────────────────────────────────────────────────────────┘
┌──────────────▼──────────── 通信抽象 ──────────────────────────────────┐
│ GroupCoordinator（cpu_group=gloo, device_group=nccl,                   │
│                   device_communicator, mq_broadcaster）                │
│ vLLM:   CudaCommunicator.all_reduce 按条件选后端；all2all_manager       │
│ SGLang: GroupCoordinator.all_reduce 内部选 "ca"/"pymscclpp"/... 后端    │
└───┬───────────────┬──────────────────┬──────────────────┬─────────────┘
    │               │                  │                  │
┌───▼─────────┐ ┌───▼──────────────┐ ┌─▼────────────────┐ ┌▼──────────────────────┐
│torch.        │ │pynccl            │ │custom allreduce   │ │EP all2all 专用库        │
│distributed   │ │（ctypes 直接调    │ │/ quick AR /        │ │DeepEP / Mori / NIXL-EP │
│ProcessGroup- │ │ libnccl.so）      │ │ symm_mem / FlashI- │ │/ FlashInfer / MoonEP  │
│NCCL（兜底）   │ │                  │ │ nfer（IPC/NVLink） │ │                       │
└───┬─────────┘ └───┬──────────────┘ └─┬────────────────┘ └┬──────────────────────┘
    │               │                  │                  │
┌───▼───────────────▼──────────────────▼──────────────────▼─────────────┐
│ NCCL（host API：ncclAllReduce/ncclSend/...；device API：LSA / GIN）     │
│ CUDA IPC / NVLink P2P / multimem（NVLS）；DeepEP V1 时代：NVSHMEM+IBGDA │
└──────────────┬────────────────────────────────────────────────────────┘
               ▼
       NVLink/NVSwitch（scale-up）   RDMA IB/RoCE（scale-out）

旁路：PD 分离 KV 传输 —— vLLM KVConnector（Nixl/Mooncake/LMCache...）、
      SGLang disaggregation/（mooncake/nixl/mori/ascend）→ RDMA 引擎，一般不经过 NCCL
```

## 读这一章的顺序

1. 先读 01，建立“并行方式 → 通信原语 → 消息大小 → 是否关键路径”的表。
2. 再读 02、03，把表里的每一行落到具体框架代码。
3. 读 04 理解 MoE All2All 为何脱离通用 collective。
4. 最后读 05，作为训练侧和排障时的 PyTorch 背景。
5. 06 可以最先读：它从模型计算的角度把一次请求从头走到尾，帮你定位 01~04 讲的每种通信发生在哪一步。

从 06 的一次请求出发，各步通信该去哪篇看细节：

```text
 06 一次请求：prefill → KV 传输 → decode（可选 AF 分离）
   ├─ 每层的 TP AllReduce、EP dispatch/combine：是什么、多大、是否关键路径 ──► 01
   ├─ 这些 AllReduce / dispatch 在 vLLM 里走哪条代码路径 ────────────────────► 02
   ├─ SGLang 的同一件事（DP-Attention gather/scatter、TBO 重叠）─────────────► 03
   ├─ dispatch/combine 的专用 kernel（normal / low-latency）─────────────────► 04 → 09-ep-kernels
   ├─ P → D 的 KV 传输（KVConnector / KVSender，走 RDMA 引擎）───────────────► 02 §7、03 §6
   └─ 兜底的 ProcessGroupNCCL、hang 与超时排障 ──────────────────────────────► 05
```

## 面试自测

1. 问：从 `RowParallelLinear` 到 NVLink 上的字节，中间经过哪几层？
   要点：模型层调用 `tensor_model_parallel_all_reduce` → TP 组 `GroupCoordinator.all_reduce` → 注册的 `torch.ops.vllm.all_reduce` 自定义算子 → `CudaCommunicator.all_reduce` 依据大小/拓扑选 custom AR、symm mem、pynccl 或 `torch.distributed` → NCCL kernel 或自研 IPC kernel → NVLink。
2. 问：为什么推理框架不直接全部用 `torch.distributed.all_reduce`？
   要点：decode 消息小（KiB 级）、延迟主导，NCCL 的启动/协议开销占比高；需要 CUDA Graph 友好；自研 one-shot/two-shot IPC kernel 或 multimem 在小消息上更快。`torch.distributed` 留作兜底和 CPU/gloo 控制面。
3. 问：PD 分离的 KV 传输为什么一般不走 NCCL？
   要点：它是点对点、按 block 寻址、跨独立实例（不同 world）的传输，需要动态建连和按需 READ/WRITE，更适合 NIXL/Mooncake 这类 RDMA 传输引擎；NCCL communicator 是固定成员的静态组。

## 待深入

- 训练框架（Megatron-LM、PyTorch FSDP2 C++ 部分）未克隆，训练侧调用链只在 01、05 中写概念。待确认：Megatron 中 TP/SP/CP 通信实现；确认方式：克隆 NVIDIA/Megatron-LM 后查 `megatron/core/tensor_parallel/` 与 `megatron/core/parallel_state.py`。
