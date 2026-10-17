# 领域上下文

本文统一仓库中反复出现的术语口径。展开讲解在 `notes/` 对应章节，这里只给能判断边界的定义。

## 术语

### 集合通信（Collective Communication）

- 定义：一组 rank 共同参与、语义由整组定义的通信操作，如 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll；区别于点对点 Send/Recv。
- 避免：把 "CCL" 只等同于 NCCL；CCL 泛指集合通信库（NCCL、RCCL、HCCL、自研 CCL 等）。
- 相关代码或契约：`third_party/nccl/src/nccl.h.in`、`labs/ccl_sim/algorithms.py`

### rank / world size

- 定义：rank 是参与通信组的单个进程（通常绑定一张 GPU）的编号；world size（本仓库记为 N）是组内 rank 总数。
- 避免：把 rank 当作物理机器编号；把 local rank（机内编号）与全局 rank 混用。
- 相关代码或契约：`labs/ccl_sim/algorithms.py`（以 list 下标模拟 rank）

### α-β 代价模型

- 定义：单条消息耗时 = α（启动延迟）+ n·β（每字节传输时间）；归约计算可另加 γ 项。用来比较算法的延迟项与带宽项。
- 避免：把 β 写成带宽本身（β 是带宽的倒数）。
- 相关代码或契约：`labs/ccl_sim/cost_model.py`、`notes/03-ccl-algorithms/02-alpha-beta-model.md`

### algbw / busbw

- 定义：algbw = 数据量 / 耗时；busbw = algbw × 与算法无关的修正因子，使不同 rank 数下的结果可以与硬件链路带宽直接比较（AllReduce 的因子为 2(N-1)/N）。
- 避免：拿 algbw 直接和网卡/NVLink 标称带宽比较。
- 相关代码或契约：`third_party/nccl-tests/doc/PERFORMANCE.md`

### scale-up / scale-out

- 定义：scale-up 指单个高速互联域内（NVLink/NVSwitch，机内或超节点）的 GPU 互联；scale-out 指跨域、经网卡与交换网络（RDMA：InfiniBand/RoCEv2）的互联。
- 避免：等同于"机内/机间"——NVL72 这类超节点的 scale-up 域可以跨多台机箱。
- 相关代码或契约：`notes/02-scale-up/README.md`、`notes/01-rdma/README.md`

### PFC / ECN / CNP / ETS（RoCE QoS）

- 定义：PFC 是按优先级（8 个之一）逐跳暂停的链路流控，暂停帧是 MAC 控制帧；ECN 是交换机在队列超阈值时给报文打的拥塞标记；CNP 是接收网卡看到 ECN 标记后回给发送方的拥塞通知包，触发 DCQCN 降速；ETS 是在各优先级之间按权重分配带宽的调度方式（与严格优先 strict 相对）。
- 避免：把 PFC 当成一种"更高优先级的流量"——它是作用在某个优先级上的暂停开关；把 RoCE 设成严格优先（会饿死 TCP）。
- 相关代码或契约：`notes/01-rdma/03-roce-ib-congestion.md`、`docs/references/2026-10-17-roce-qos-config.md`

### QP（Queue Pair）

- 定义：RDMA Verbs 的通信端点，由发送队列与接收队列组成，需经 RESET→INIT→RTR→RTS 状态转换才能收发。
- 避免：与 TCP socket 一一类比时忽略其需要显式注册内存（MR）和轮询完成队列（CQ）。
- 相关代码或契约：`third_party/rdma-core/libibverbs/verbs.h`

### channel（NCCL）

- 定义：NCCL 中一条独立的通信路径（如一个 ring 或一棵 tree），对应 GPU 上的一个 CTA；多个 channel 并行以填满带宽。
- 避免：与 CUDA stream 混淆。
- 相关代码或契约：`notes/04-nccl-internals/03-channels-protocols-kernels.md`

### 协议 LL / LL128 / Simple（NCCL）

- 定义：NCCL 的三种数据传输协议，在延迟与带宽之间取舍：LL 以 8 字节为单位带标志位、延迟最低；LL128 以 128 字节为单位；Simple 用大块数据加显式同步、带宽最高。
- 避免：与"算法"（Ring/Tree 等）混用——算法决定数据走向，协议决定每段数据如何搬运和同步。
- 相关代码或契约：`notes/03-ccl-algorithms/03-nccl-algorithms.md`

### 通算融合（Communication-Computation Fusion）

- 定义：把通信与计算在同一 kernel 或细粒度流水中交织执行，以隐藏通信延迟；比 stream 级 overlap 粒度更细。
- 避免：把任何"两个 stream 并行"都称为融合。
- 相关代码或契约：`notes/06-comm-compute-fusion/README.md`、`third_party/Triton-distributed/`

### NCCL device API / GIN

- 定义：NCCL 提供的 device 侧发起通信接口：GPU kernel 内直接对对称内存窗口（`ncclCommWindowRegister`）做 put/signal。GIN（GPU-Initiated Networking）是其跨节点部分，有 GDAKI（GPU 直接驱动网卡，类似 IBGDA）与 PROXY（经 CPU 代理）两种后端。
- 避免：认为 device-initiated 通信只能用 NVSHMEM——本地 DeepEP V2.5 已改用 NCCL GIN。
- 相关代码或契约：`third_party/nccl/docs/userguide/source/usage/deviceapi.rst`、`third_party/nccl/docs/contrib/GIN/`、`notes/06-comm-compute-fusion/03-nvshmem-and-device-initiated.md`

### Full-mesh / one-shot / two-shot

- 定义：Full-mesh 指任意两 rank 直连（机内 NVSwitch），一步就与所有对端交换。one-shot AllReduce 让每个 rank 读所有对端的完整输入后本地归约（1 步、(N−1)·M 字节）；two-shot 是 Full-mesh ReduceScatter + AllGather（2 步、与 Ring 相同的 2(N−1)/N·M 字节）。
- 避免：在 PCIe 或部分互联拓扑上套用 Full-mesh 的代价公式——N−1 条并发流会挤在同一条链路上。
- 相关代码或契约：`notes/03-ccl-algorithms/05-handwrite-ring-fullmesh.md`、`labs/ccl_sim/algorithms.py`、`third_party/vllm/csrc/custom_all_reduce.cuh`

### dispatch / combine（EP）

- 定义：MoE 专家并行中的两次 All-to-All-v。dispatch 按 top-k 路由把 token 发到专家所在 rank 并按专家排布；combine 把专家输出送回 token 原 rank 并按门控权重求和。
- 避免：把它当成普通 `ncclAlltoAll`——数据量由每步路由动态决定，且与 token 重排、加权求和融合在同一 kernel。
- 相关代码或契约：`notes/09-ep-kernels/README.md`、`third_party/DeepEP-hybrid-ep/csrc/kernels/internode_ll.cu`

### LL / HT（low-latency / high-throughput）

- 定义：EP 通信的两类 kernel。LL 面向 decode：直连、固定容量缓冲、无 CPU 同步、可 CUDA Graph；HT（DeepEP 叫 normal）面向训练与 prefill：跨节点 RDMA + 节点内 NVLink 分层，追求带宽。
- 避免：认为 LL 一定"纯 RDMA"——DeepEP LL 在对端可经 NVLink 访问时会直接 store。
- 相关代码或契约：`notes/09-ep-kernels/02-deepep-low-latency.md`、`third_party/nccl/contrib/nccl_ep/include/ep_enums.h`

### Hybrid-EP

- 定义：NVIDIA 在 DeepEP `hybrid-ep` 分支实现的高吞吐 EP：先 AllGather 路由并预计算每个 token 的落点，跨节点只发给同 local rank 的 peer，再经 NVLink 扇出，用 TMA warp 专业化流水线降低 SM 占用。
- 避免：与 DeepEP V2.5 的 `allow_hybrid_mode`（RDMA+NVLink 分层模式）和 NCCL EP 的 `hybrid_ep.cuh`（Hybrid-EP 的 NCCL 移植）混为一谈；三者都指"分层"，但实现不同。
- 相关代码或契约：`third_party/DeepEP-hybrid-ep/csrc/hybrid_ep/`、`notes/09-ep-kernels/03-hybrid-ep.md`、`notes/09-ep-kernels/04-lineage-and-selection.md`

### IBGDA

- 定义：InfiniBand GPUDirect Async，GPU 线程直接在网卡 SQ 中写 WQE、更新 doorbell record 并写 BlueFlame 寄存器发起 RDMA，不经过 CPU。
- 避免：认为 IBGDA 改变了网卡侧的 RC 协议——网卡看到的 WQE 和门铃与 CPU 发起时相同，变的只是提交者。
- 相关代码或契约：`third_party/DeepEP-hybrid-ep/csrc/kernels/ibgda_device.cuh`

## 术语关系

| 关系 | 说明 |
|---|---|
| AllReduce = ReduceScatter + AllGather | Ring 与 Halving-Doubling 的带宽最优实现都基于此分解 |
| 算法 × 协议 | NCCL 对每次调用分别选择算法和协议，二者正交 |
| scale-up 与 scale-out 的分层 | 分层集合通信先在 scale-up 域内归约，再跨 scale-out 网络 |
| 上层 → 中层 → 下层 | 框架（vLLM/SGLang/PyTorch）→ CCL（NCCL/DeepEP）→ transport（P2P/SHM/NVLS · IB Verbs） |

## 待澄清歧义

| 词语 | 可能含义 | 影响范围 | 确认方式 |
|---|---|---|---|
| "scale up 等互联技术"（JD 原文） | NVLink/NVSwitch；或字节自研的 scale-up 互联方案 | `notes/02-scale-up/` 的深度取舍 | 面试或与团队沟通时确认 |
| "自研 RDMA 协议、拥塞算法"（JD 原文） | 对 RoCEv2 的改进；或全新传输协议 | `notes/01-rdma/03-roce-ib-congestion.md` | 查阅字节公开论文/演讲，或面试时确认 |
