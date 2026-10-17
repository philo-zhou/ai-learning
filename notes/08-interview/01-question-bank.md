# 集合通信库研发面试题库

> 层次：求职 · 来源：third_party/ai-infra-book@3bdcb4fc（第 4、5、6、7、9、10 章）、third_party/nccl@3e46c089、third_party/nccl-tests@afd59aba、third_party/perftest@05200c2f、third_party/rdma-core@bd3282a1、third_party/Triton-distributed@7908e4ea、third_party/DeepEP@93eb6eb2、third_party/vllm@02653fa2、third_party/sglang@d2a43644

## 一句话

覆盖 JD 全部条目的题库，每题给要点答案；与书或源码相关的给出本地路径（书路径前缀 `third_party/ai-infra-book/manuscripts/`）。

## 面试自测

难度：★ 基础必答 · ★★ 常见追问 · ★★★ 深挖/开放。

### 一、RDMA 与网络（R2、Q2）

**1. ★ RDMA 相比 TCP 为什么快？**
- kernel bypass：用户态直接写 doorbell 提交 WR，数据面不进内核；zero-copy：网卡 DMA 直接读写已注册内存；传输层（可靠、分段、重传、ACK）由网卡硬件完成，CPU 不参与。
- 代价：内存要预先注册（pin + 地址翻译表）、连接状态在网卡上（QP 上下文），扩展性受网卡缓存限制。
- 参考：书 `07-数据中心网络.md` §7.3.1~7.3.2。

**2. ★ 讲一下 Verbs 编程的基本对象和一次 RC Write 的完整流程。**
- 对象：device context → PD（保护域）→ MR（注册内存，得到 lkey/rkey）→ CQ（完成队列）→ QP（SQ+RQ）。
- 流程：创建 QP → 带外交换 QPN/LID 或 GID/PSN/rkey/addr → `ibv_modify_qp` 依次 INIT → RTR → RTS → `ibv_post_send`（opcode=RDMA_WRITE，带 remote addr + rkey）→ 轮询 CQ 拿 WC。
- 参考：`third_party/rdma-core/libibverbs/examples/rc_pingpong.c`（`ibv_reg_mr`、`ibv_create_qp`、`ibv_modify_qp`、`ibv_post_send`）。

**3. ★★ QP 状态机各状态需要设置什么关键属性？**
- INIT：端口号、pkey、访问权限；RTR：对端 QPN、对端地址（LID 或 GID/RoCE 下的 GRH）、path MTU、RQ PSN、max_dest_rd_atomic、min_rnr_timer；RTS：SQ PSN、timeout、retry_cnt、rnr_retry、max_rd_atomic。
- 常见坑：RoCE 下 GID index 选错（v1/v2、IPv4/IPv6）导致建链失败或不通；timeout 太小在拥塞时误重传。NCCL 对应参数 `NCCL_IB_GID_INDEX`、`NCCL_IB_TIMEOUT`（`third_party/nccl/src/transport/net_ib/connect.cc`，默认分别为 -1 和 20）。

状态迁移与每一步要设的属性：

```text
RESET ──► INIT ──────────────► RTR ──────────────────────────────► RTS
          port, pkey,          remote QPN,                         SQ PSN, timeout,
          access flags         remote addr (LID or GID/GRH),       retry_cnt, rnr_retry,
                               path MTU, RQ PSN,                   max_rd_atomic
                               max_dest_rd_atomic,
                               min_rnr_timer
```

**4. ★ 单边（Write/Read）与双边（Send/Recv）有什么区别？各适合什么？**
- 单边：目标端 CPU 不参与，需事先知道远端地址与 rkey；Write 延迟低，Read 需要往返且占用 outstanding read 资源。双边：接收端要预先 post recv，适合控制消息、不知道目标地址的场景，可能遇到 RNR。
- NCCL 的 IB transport 数据面以 RDMA Write 为主，配合通知机制告诉对端数据到达。待确认：具体使用 Write-with-Imm 还是独立 flag 写；确认方式：阅读 `third_party/nccl/src/transport/net_ib/p2p.cc`。

**5. ★ RC / UC / UD 区别？为什么集合通信多用 RC？**
- RC：可靠、有连接、支持全部操作；UC：不可靠有连接，不支持 Read/Atomic；UD：不可靠无连接，单包 ≤ MTU，可扩展性好。
- 集合通信要可靠、大消息、单边写，因此用 RC；代价是 N² 连接状态。DC（动态连接）、SRD 等是扩展性方案（待确认：本地源码中的对应实现，NCCL 有 `src/transport/net_efa_gda/` 目录）。

**6. ★ IB 与 RoCEv2 的区别？RoCEv2 为什么需要无损网络？**
- IB：专用链路层，基于 credit 的链路级流控，天然无损，子网管理器集中管理；RoCEv2：IB 传输层封装在 UDP/IP（目的端口 4791）之上，跑在以太网上，可路由、可用 ECMP。
- 传统 RoCE 网卡用 go-back-N 重传，丢包代价大，所以需要 PFC 做无损 + ECN/DCQCN 做拥塞控制。

**7. ★★ PFC 和 DCQCN 各自解决什么问题？为什么两者都要？**
- PFC：一跳反馈，交换机入口队列超阈值向上游发 PAUSE，按优先级暂停，防止缓冲溢出丢包；DCQCN：端到端，交换机打 ECN 标记 → 接收端回 CNP → 发送端降速，从源头降低速率。
- 书中的量化：PFC 一跳反馈约 0.33 μs，端到端反馈约 20 μs（书中取值），PFC 先在短时间内兜住缓冲，DCQCN 再把速率降下来解除暂停（书 `07-数据中心网络.md` §7.5.2）。
- PFC 副作用：队头阻塞、拥塞扩散、PFC 风暴、死锁（§7.5.4）。

两种反馈的路径对比（时延为书中取值）：

```text
PFC: hop-by-hop, ~0.33 us
  upstream ──data──► switch (ingress queue > threshold)
     ▲                         │
     └───────── PAUSE ─────────┘        按优先级暂停上游，兜住缓冲

DCQCN: end-to-end, ~20 us
  sender ──data──► switch (marks ECN) ──data+ECN──► receiver
     ▲                                                 │
     └──────────────────────── CNP ────────────────────┘    发送端降速
```

**8. ★★ 什么是 ECMP 哈希冲突？对集合通信有什么影响？怎么缓解？**
- 集合通信流少而大（大象流），五元组哈希可能把多条流压到同一上行链路，造成局部拥塞，整体带宽远低于对分带宽。
- 缓解：每连接多 QP 分散源端口（NCCL `NCCL_IB_QPS_PER_CONNECTION`，`src/transport/net_ib/connect.cc`，默认 1）、rail-optimized 拓扑让同号 rail 流量留在同一叶交换机、自适应路由/包喷洒、作业放置错峰（书 §7.2.5、§7.5.3；案例 `third_party/ai-infra-book/case-studies/network-planning-and-collectives.md`）。

**9. ★★ 什么是 GPUDirect RDMA？和 GPU 发起通信（IBGDA/GDAKI）有什么区别？**
- GPUDirect RDMA：网卡直接 DMA GPU 显存，载荷不经主机内存，但请求提交和完成处理仍由 CPU（NCCL 中是 proxy 线程）做。
- GPU 发起：GPU 自己构造 WQE、敲 doorbell、轮询 CQ，CPU 退出关键路径；书中比较了三种发起位置的 PCIe 穿越次数：CPU 代理 3 次、GPU SM 2 次、网卡处理器 0 次（书 `06-超节点.md` §6.4.4）。
- NCCL 中相关目录：`third_party/nccl/src/transport/net_ib/gdr.cc`、`net_ib/gdaki/`。

**10. ★★★ 为什么小消息时网卡带宽用不满？如何估算？**
- 两个限制：在途窗口（每个请求占用槽位时间 × 槽位数）和请求启动速率 $1/\delta$。有效带宽 ≤ min(链路带宽, 槽位数 × 消息大小 / 槽位占用时间, 消息大小 / δ)。
- 书中例题：RoCE RC δ≈18.6 ns 时，50 GB/s 链路需消息约 931 B 以上才能被带宽限制（书中取值，`07-数据中心网络.md` §7.3.4、§7.2.3）。MoE decode 的 dispatch 正处在这种区间。

**11. ★★★ 大规模下 RC 的连接扩展性问题怎么解决？**
- 每 QP 上下文数百字节，N 个端点全连接状态 ∝ N²，超过网卡片上缓存后频繁从主机内存取上下文，性能下降。书中算例：RoCE RC 每连接 512 B 上下文、256 KiB 缓存时 N≥23 溢出（书中取值，§7.4.2）。
- 方案：按需建连（NCCL 也是 lazy connect，待确认：具体位置，确认方式：搜索 `third_party/nccl/src/transport.cc` 中连接建立逻辑）、共享接收队列 SRQ、DC/UD 类传输、事务层与传输层分离（UB 的 Jetty 思路）。

### 二、集合通信算法与代价模型（Q3、R4）

**12. ★ 说出常见集合操作的语义，并说明 AllReduce 和 ReduceScatter/AllGather 的关系。**
- AllReduce = ReduceScatter + AllGather；Broadcast/Reduce 只涉及一个根；All2All 按目的地重排数据。
- MoE combine 不是 AllReduce：它包括回传、恢复 token 顺序、按路由权重求和（书 `06-超节点.md` §6.4.1）。

**13. ★ 推导 Ring AllReduce 的步数和每卡发送量，并写出 α-β 时间。**
- RS 阶段 n−1 步、AG 阶段 n−1 步，每步每卡发 M/n，总发送 $2(n-1)M/n$。
- $T_{ring}=2(n-1)\alpha+2(n-1)M/(nB)$（书式 6-9）。大 M 时趋近 $2M/B$，与 n 基本无关，带宽最优；α 项随 n 线性增长，小消息和大规模时吃亏。

以 n=4、数据切成 c0~c3 为例，每格是该 rank 在这一步发给下一个 rank 的块（环：R0→R1→R2→R3→R0）：

```text
            R0    R1    R2    R3
RS step 1   c0    c1    c2    c3      收到的块与本地同号块累加
RS step 2   c3    c0    c1    c2      发上一步刚累加好的块
RS step 3   c2    c3    c0    c1
            → RS 结束：R_r 持有完整归约的 c_(r+1)，例如 R3 持有 c0
AG step 1   c1    c2    c3    c0      先发自己完整的块
AG step 2   c0    c1    c2    c3      再转发收到的完整块（只覆盖，不累加）
AG step 3   c3    c0    c1    c2
            → 共 2(n-1) = 6 步，每步每卡发 M/n
```

**14. ★ Tree 算法的时间？环和树如何选择？**
- 未分段二项树：$T_{tree}=2\log_2 n(\alpha+M/B)$（书式 6-10）。轮数少，但每轮在关键路径上传完整张量。
- 书中八卡 H100（α=0.822 μs、B=450 GB/s，书中取值）交点约 680 KiB：小于它用树，大于它用环（书 §6.4.2）。
- NCCL 的 Tree 是 pipeline 分段的 double binary tree，大消息也能接近带宽最优（`third_party/nccl/src/graph/trees.cc` 的 `ncclGetDtree`/`ncclGetBtree`）。

**15. ★★ double binary tree 为什么要两棵树？**
- 单棵二叉树中叶子只收不发、内部节点发两份，带宽利用不对称；两棵树互补（一棵中的叶子是另一棵中的内部节点），各传一半数据，使每个 rank 收发负载均衡，同时保持 $O(\log n)$ 延迟。
- 源码：`third_party/nccl/src/graph/trees.cc` 注释 "Build a double binary tree"；kernel 侧 `src/device/all_reduce.h` 的 `runTreeUpDown`、`runTreeSplit`。

**16. ★★ 讲一下 Recursive Halving-Doubling（Rabenseifner）AllReduce。**
- RS 用 recursive halving：第 k 步与距离 $n/2^k$ 的伙伴交换一半数据并归约，$\log_2 n$ 步后每卡持有 1/n 结果；AG 用 recursive doubling 反向。
- 时间：$2\log_2 n \cdot \alpha + 2(n-1)M/(nB)$，带宽项与 ring 相同、延迟项为对数，中等消息有优势；要求 n 为 2 的幂（否则需要预处理），伙伴距离大时在物理拓扑上可能多跳、拥塞（书 §6.5.1 用"递归算法每轮对端距离翻倍"与 Swing 对比链路负载）。
- NCCL 里有没有？本地源码核实：算法编号只有 Tree/Ring/CollNetDirect/CollNetChain/NVLS/NVLSTree/PAT 七种（`third_party/nccl/src/include/plugin/nccl_tuner.h` 的 `NCCL_ALGO_*`，`src/init.cc` 的 `ncclAlgoStr`），`src/config/algorithm_registry.cc` 的 `algRegistry[]` 中也没有 halving/doubling 条目，`grep -riE 'halving|doubling' third_party/nccl/src` 无匹配，所以 NCCL 没有独立的 Halving-Doubling（Rabenseifner）AllReduce；最接近的是 PAT——仅 AllGather/ReduceScatter、仅 Simple，`src/include/collectives.h` 的 `PatRSAlgorithm`/`PatAGAlgorithm` 按 2 的幂维度（`nrPow2`、`firstBitSet`）选对端，是对数步算法。

**17. ★★ 分层 AllReduce 是怎么做的？为什么比平铺的大环好？**
- 机内 ReduceScatter（NVLink）→ 同号 rank 跨机 AllReduce（每个 rank 走自己的 NIC/rail）→ 机内 AllGather。
- 书中例题 7.2：两台 HGX 16 rank，连续环跨服务器只用 2 张网卡、720 MiB；交错环所有边跨机、5760 MiB；分层归约 384 MiB 且 8 张网卡并行（书 `07-数据中心网络.md` §7.2.1）。
- 前提：机内带宽显著高于网卡带宽。

以例题中两台 8 卡服务器为例，三个阶段如下：

```text
step 1  intra-node ReduceScatter (NVLink)
        server A: GPU0 GPU1 ... GPU7        server B: GPU0 GPU1 ... GPU7
        每卡得到 1/8 块的机内部分和

step 2  inter-node AllReduce between same-index ranks, one NIC / rail each
        A.GPU0 ◄═══ NIC0 / rail0 ═══► B.GPU0
        A.GPU1 ◄═══ NIC1 / rail1 ═══► B.GPU1
          ...                           ...
        A.GPU7 ◄═══ NIC7 / rail7 ═══► B.GPU7      8 张网卡并行

step 3  intra-node AllGather (NVLink)
        每卡拿回完整结果
```

**18. ★★ busbw 是什么？为什么 AllReduce 的系数是 2(n−1)/n，All2All 是 (n−1)/n？**
- algbw = 数据量 / 时间；busbw = algbw × 系数，系数是该操作在最优算法下每卡实际需收发的数据量与用户数据量之比，使 busbw 可与链路带宽直接比较、且不随 n 变。
- 源码：`third_party/nccl-tests/src/all_reduce.cu` 的 `AllReduceGetBw`（factor = 2(n−1)/n）；`src/alltoall.cu` 的 `AlltoAllGetBw`（数据量按 count×nranks，factor = (n−1)/n）。
- 注意：busbw 不等于每张网卡的实际吞吐，分层/在网归约下要按物理路径重算（书 §6.4.3）。

**19. ★★ 在网归约（SHARP）/ NVLS 为什么能更快？**
- 交换机内做归约，端点各发一次、收一次结果，字节数从 $2(S-1)/S$ 份降为约 1 份，轮数也减少；书中 32 台服务器时在网归约时间约为环的 49%（书中模型，§7.2.1）。
- 限制：交换机归约吞吐、支持的数据类型、组规模。NCCL 中对应 CollNet（`src/transport/coll_net.cc`）与 NVLS（`src/transport/nvls.cc`）。

**20. ★★★ All2All 在多机间怎么优化？**
- 问题：n² 对小消息，网卡报文速率和 incast 成为瓶颈；EP 场景还不等长。
- 手段：分层 All2All（先机内 NVLink 聚合同目的节点的数据再跨机，减少跨机消息数）、PXN 类"经同号 rail 的 GPU 转发"、多 QP、错峰/打散发送顺序避免 incast、GPU 发起减少 CPU 代理开销（DeepEP 低延迟模式）。
- NCCL 中不等长 All2All 用 group 内成组 Send/Recv 表达（书 §6.4.3）。本地 NCCL 中 PXN 由 `ncclPxnDisable(comm)` 控制（调用处见 `third_party/nccl/src/init.cc`、`src/enqueue/enqueue.cc`）；待确认：对应环境变量名与 PXN 的具体转发路径；确认方式：查找 `ncclPxnDisable` 的定义。

### 三、NCCL 原理（Q4、R1）

**21. ★ 讲一下 NCCL 初始化流程。**
- `ncclGetUniqueId` → 各 rank `ncclCommInitRank`（`third_party/nccl/src/init.cc`）→ bootstrap 建立带外 socket 环、AllGather 各 rank 信息（`src/bootstrap.cc`）→ 拓扑探测 `ncclTopoGetSystem`（`src/graph/topo.cc`）→ 路径计算 `ncclTopoComputePaths`（`src/graph/paths.cc`）→ 图搜索 `ncclTopoCompute`（`src/graph/search.cc`）得到 ring/tree 等图 → 建 channel 与连接（`src/graph/connect.cc`、`src/channel.cc`、`src/transport.cc`）→ tuning 模型 `ncclTopoTuneModel`（声明于 `src/include/graph.h`）。

**22. ★★ 什么是 channel？channel 数怎么影响性能？**
- channel 是一套独立的通信路径（一组 ring/tree 连接 + 缓冲），每个 channel 对应 kernel 中的一个 CTA，数据切块并行走多个 channel 以用满多条链路/多张网卡。
- channel 多：带宽高、但占 SM 多，和计算争用（书 §6.4.3）；channel 少：SM 省但带宽可能不足。可用 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` 调整（`NCCL_MIN_NCHANNELS` 读取于 `src/graph/connect.cc`）。

**23. ★★ LL、LL128、Simple 三种协议区别？**
- LL：每 8 字节中 4 字节数据 + 4 字节 flag（`src/include/device.h` 的 `union ncclLLFifoLine`：data1/flag1/data2/flag2），接收方轮询 flag 即知数据到达，无需内存栅栏，延迟最低、有效带宽约 50%。
- LL128：128 字节行（`NCCL_LL128_LINESIZE 128`）中 1 个 8 字节元素作 flag（`NCCL_LL128_DATAELEMS = LINEELEMS - 1`），带宽效率约 120/128，依赖硬件保证 128 字节写的原子可见性。
- Simple：大块数据 + 独立 flag，需要 fence，延迟高、带宽最好。
- 源码：`src/device/prims_ll.h`、`prims_ll128.h`、`prims_simple.h`；可用 `NCCL_PROTO` 强制。
- 待确认：LL128 默认启用的硬件/平台条件；确认方式：阅读 `third_party/nccl/src/tuning/` 中 LL128 相关判断。

两种协议的行格式（LL128 一行按数量示意，flag 在行内的实际位置见 `prims_ll128.h` 的 `NCCL_LL128_FLAGTHREAD`）：

```text
LL: one ncclLLFifoLine = 16 B
┌─────────┬─────────┬─────────┬─────────┐
│ data1   │ flag1   │ data2   │ flag2   │    每 8 B 里 4 B 数据 + 4 B flag，有效约 50%
└─────────┴─────────┴─────────┴─────────┘
   4 B       4 B       4 B       4 B

LL128: one line = 128 B = 16 x 8 B
┌──────────────────────────────┬───────┐
│ 15 x 8 B data                │ 8 B   │    数据 120 B / 128 B
│ (NCCL_LL128_DATAELEMS)       │ flag  │
└──────────────────────────────┴───────┘
```

**24. ★★ NCCL 如何选择算法和协议？**
- 对每个 (算法, 协议) 用代价模型估计时间 ≈ 基础延迟 + 硬件延迟 × 步数 + 数据量 / 带宽，选最小者；带宽来自图搜索得到的每 channel 带宽，延迟来自常量表（`src/tuning/` 中 `baseLatencies`、`hwLatencies`，见 `tuning/nvls.cc`、`tuning/collnet.cc`）。
- 可用 `NCCL_ALGO`、`NCCL_PROTO` 覆盖；也支持 tuner 插件（`src/plugin/`）。

**25. ★★ proxy 线程是做什么的？**
- 网络 transport 下，GPU kernel 把数据放到缓冲并更新 FIFO 中的 tail/head，CPU proxy 线程（`src/proxy.cc` 的 `ncclProxyProgress`）轮询这些状态，调用网络插件 isend/irecv/test 推进收发，再把完成状态写回给 GPU。
- 机内 P2P/SHM 通常不需要 proxy 参与数据面。proxy 是 CPU 侧瓶颈与抖动来源之一；GPU 发起（GDAKI）可以去掉它（`src/transport/net_ib/gdaki/`）。

网络 transport 下一次发送的推进过程：

```text
 GPU kernel                  CPU proxy thread               net plugin / NIC
     │                       (ncclProxyProgress)                 │
     │ 1. data -> buffer,           │                            │
     │    update tail/head ────────►│ 2. poll FIFO state         │
     │                              │ 3. isend / irecv ─────────►│
     │                              │    test until done ◄───────│
     │◄──────────────────────────── │ 4. write completion back   │
```

**26. ★★ NCCL 如何感知拓扑？如何判断两卡能否走 P2P？**
- 通过 sysfs/NVML 建拓扑图（GPU、PCI 交换、NIC、CPU 节点、NVLink），计算任意两点路径类型（如 NVLink、同 PCIe 交换、跨 CPU 等）。
- P2P 是否启用受 `NCCL_P2P_LEVEL`/`NCCL_P2P_DISABLE` 约束（`src/graph/paths.cc` 中 `ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL")`）。可用 `NCCL_TOPO_DUMP_FILE` 导出拓扑。

**27. ★★ ncclGroupStart/End 的作用？不用会怎样？**
- 把多个操作合并提交（一次 launch、一次连接/调度），也是在单进程多 GPU 下避免死锁、在 Send/Recv 对之间避免互相等待的手段。
- 书中提醒：需要一起推进的 Send/Recv 要放进同一组，否则会串行等待对方尚未发起的操作（书 §6.4.3）。源码：`src/group.cc`。

**28. ★★ NCCL 调用返回意味着什么？为什么集合通信 hang？**
- 返回只表示已入队到 stream，完成要靠 stream 顺序或 event；缓冲区在完成前不能复用（书 §6.4.3、`05-算子与运行时.md` §5.1.3）。
- hang 常见原因：各 rank 调用顺序/数量/count 不一致；某个 rank 没调用（条件分支、异常）；网络建链失败或 GID/接口错误；死锁的 group 用法；某卡上游计算卡住。

**29. ★★★ 如果让你设计一个新的 CCL，你会怎么分层？**
- 参考 NCCL 分层：API 与 group 语义 → 任务入队与调度（enqueue、scheduler）→ 拓扑/图/调优 → device kernel 与 primitives → transport 抽象（P2P/SHM/NVLS/NET/CollNet）→ 网络插件 → 观测（profiler、RAS）。
- 加分点：一开始就考虑 GPU 发起和对称内存（device API，NCCL 已有 `src/nccl_device/`、`src/gin/` 等目录，待确认：具体能力，确认方式：阅读对应目录与头文件）、拓扑感知算法可插拔、在线调优与故障检测、与框架的 stream/内存分配器协作（vLLM `pynccl_allocator.py`）。

### 四、scale-up（R2）

**30. ★ NVLink/NVSwitch 与 PCIe P2P 有什么区别？**
- NVLink 带宽远高于 PCIe（H100 NVLink 双向合计 900 GB/s，每方向 450 GB/s，书中引用规格，§4.5.3）；NVSwitch 把点对点直连变成交换结构，任意两卡全带宽。PCIe P2P 受 PCIe 交换/根复合体限制，跨 CPU socket 时通常很差。
- NVIDIA 从 DGX-1 混合立方体到 NVSwitch 再到 GB200 NVL72 机柜级 NVLink（书 `06-超节点.md` §6.5.3）。

**31. ★★ NCCL 机内的 P2P transport 和 SHM transport 分别在什么情况下使用？**
- P2P：GPU 可直接互访（NVLink 或允许的 PCIe 路径）时，用 CUDA IPC 映射对端缓冲，kernel 直接读写（`src/transport/p2p.cc`）。
- SHM：不能 P2P 时经主机共享内存中转（`src/transport/shm.cc`，可用 `NCCL_SHM_DISABLE` 关闭）。
- 待确认：P2P 模式下 read/write 方向与 CE（copy engine）使用条件；确认方式：阅读 `p2p.cc` 中相关参数。

**32. ★★ NVLS 是什么？对 AllReduce 有什么帮助？**
- 利用 NVSwitch 的 multicast 与交换机内归约，GPU 一次写入/一次读出即可完成归约，减少 SM 占用和 HBM 流量。NCCL 中算法 NVLS/NVLSTree，`NCCL_NVLS_ENABLE`（`src/transport/nvls.cc`，默认 2）。
- 待确认：默认值 2 的具体含义（推测为自动）；确认方式：阅读 `nvls.cc` 中 `ncclParamNvlsEnable()` 的使用处。

**33. ★★★ 超节点变大（如 72 卡 NVLink 域）后并行策略如何变化？**
- TP/EP 可以放在更大的 NVLink 域内，跨机 RDMA 只承载 DP/PP；书中比较 1024 卡下超节点 8/64/128/256 卡时的时间变化（`07-数据中心网络.md` §7.6.4），以及 TP16 跨两台 HGX vs 同一 NVLink 域的差异（`06-超节点.md` §6.5.1、§6.5.3）。
- 同时要考虑故障域扩大、功率与成本（§6.7.2）。

### 五、通信与计算融合（R3）

**34. ★ 通信与计算重叠有哪几种粒度？**
- 算子级：不同 stream 并发（DDP 分桶、FSDP 预取）；分块流水：把 GEMM 和通信按 chunk 切开交替；kernel 内融合：一个 kernel 里按 tile 计算、发送、等待信号（Triton-distributed、persistent kernel，书 `05-算子与运行时.md` §5.5.4）。

**35. ★★ 为什么重叠后加速比达不到预期？**
- 通信 kernel 与 GEMM 争用 SM、HBM 带宽、L2；书实验 6-5：64 MiB 时通信从 36.92 ms 变 45.21 ms、计算从 11.03 ms 变 12.89 ms，重叠只省约 2.7 ms（书 §6.4.3，CPU+Gloo 实验）。
- 评价重叠要测同时运行的时间，不能用单独时间相减；还要看数据依赖（计算出最后一块才能发）。

**36. ★★ 讲一下 AllGather + GEMM 融合的思路。**
- TP/SP 下，GEMM 需要完整的激活，传统做法先 AllGather 再 GEMM。融合：每块分片到达后立即计算对应行块的 GEMM，通信由 copy engine 或少量 SM 执行，计算 tile 按"已到达分片"顺序调度（swizzle），用 signal/wait 同步。
- 源码：`third_party/Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py`、`ag_gemm_threadblock_swizzle.py`；反向的 `gemm_reduce_scatter.py`。

**37. ★★ 对称内存/NVSHMEM 编程模型的关键原语？**
- 每个 rank 在对称堆上分配同样大小的缓冲，可直接用远端地址 put/get；signal（写远端标志）+ wait（等待本地标志）实现同步；fence 保证顺序、quiet 保证完成。
- 参考：`third_party/Triton-distributed/tutorials/01-distributed-notify-wait.py`；`third_party/Triton-distributed/shmem/nvshmem_bind/`；vLLM `device_communicators/symm_mem.py`、SGLang `torch_symm_mem.py`。书案例 `case-studies/remote-ordering-and-completion.md` 讨论顺序与完成语义。

**38. ★★ 融合时有哪些正确性约束？**
- 计算依赖归约后的完整值的算子不能在归约前做：RMSNorm([1,0]+[0,1]) = [1,1]，分别归一化再加得 [√2,√2]（书 §6.4.4）。
- 跨 SM 交接需要 fence + flag（书 `04-加速器架构.md` §4.4.4）；缓冲复用需要等对端消费完（背压，书 `07-数据中心网络.md` §7.4.4）。

**39. ★★★ MoE 的 dispatch / GEMM / combine 如何融合或流水？**
- micro-batch 流水：三阶段错开，理想时间 = 一个 micro-batch 三段之和 + 最长段；书中算例 0.827 ms → 0.581 ms（书 `09-分布式推理.md` §9.4.3，书中数值）。
- kernel 级：dispatch 到达即开始对应专家的 grouped GEMM，GEMM 结束即 combine 发送；参考 `third_party/Triton-distributed/python/triton_dist/kernels/nvidia/all_to_all_single_gemm.py`、`ep_all2all_fused.py`、`moe_reduce_rs.py`。
- 注意 grouped GEMM 的 padding 与 DMA-GEMM 争用 HBM。

### 六、框架与场景（R5）

**40. ★ 各种并行方式分别产生什么通信？**
- DP：梯度 AllReduce；ZeRO/FSDP：参数 AllGather + 梯度 ReduceScatter；TP：每层前向 2 次 AllReduce（或 SP 下 AG+RS）；PP：相邻 stage Send/Recv；CP：注意力 KV 的 ring P2P 或 All2All；EP：dispatch/combine All2All（书 `06-超节点.md` §6.2、`10-训练系统.md` §10.2.1）。

**41. ★★ DDP 为什么要做梯度分桶？桶大小怎么选？**
- 反向逐层产生梯度，分桶后桶满即发起 AllReduce 与后续反向计算重叠；桶太大要等最后一个梯度，桶太小启动次数多（α 项大）（书 §10.3.3）。
- 待确认：PyTorch 默认桶大小与 Reducer 实现；确认方式：本地未克隆 PyTorch，需克隆后查 `torch/csrc/distributed/c10d/reducer.cpp`。

**42. ★★ 推理框架为什么不直接用 NCCL AllReduce，而要自带 custom allreduce？**
- decode 阶段 TP AllReduce 消息小（如 10 KiB 量级），启动和轮次占主导；NCCL 通用路径的固定开销相对大。custom allreduce 用 CUDA IPC 映射缓冲，一个 kernel 内 one-shot/two-shot 直接读写对端，并可被 CUDA Graph 捕获（书 §6.4.4 末尾提到框架为小数据量选专门实现）。
- 源码：`third_party/vllm/vllm/distributed/device_communicators/custom_all_reduce.py`、`quick_all_reduce.py`、`symm_mem.py`、`flashinfer_all_reduce.py`；SGLang 同名目录下 `custom_all_reduce.py`、`pymscclpp.py`。

**43. ★★ vLLM 中 TP AllReduce 的调用链？**
- 模型层调用 → `parallel_state.py` 中的 `GroupCoordinator`（持有 `device_communicator`）→ `cuda_communicator.py` 按条件选择 custom allreduce / symm mem / PyNccl（`pynccl.py` 中 `PyNcclCommunicator`，通过 `pynccl_wrapper.py` ctypes 调 NCCL）。
- 待确认：`cuda_communicator.py` 中各实现的选择顺序与阈值；确认方式：阅读 `third_party/vllm/vllm/distributed/device_communicators/cuda_communicator.py` 的 `all_reduce`。

**44. ★★ DeepEP 的 normal 与 low-latency 模式有何区别？**
- normal：面向训练/prefill 的大批量，机内 NVLink + 机间 RDMA 分层转发，追求带宽；low-latency：面向 decode，GPU 直接发起 RDMA，纯 RDMA 路径、占用 SM 少、可与计算重叠（书 §6.4.4 提到 DeepEP 低延迟 kernel 走 GPU SM 发起路径）。
- 待确认：本地 DeepEP 版本中的接口名（本地 `deep_ep/buffers/ep.py` 中有 `dispatch`、`combine`）、模式切换方式；确认方式：阅读 `third_party/DeepEP/README.md` 与 `deep_ep/buffers/ep.py`。

**45. ★★ PD 分离中 KV 传输有什么通信问题？**
- 数据量大（与序列长度、层数成正比），需要逐层/分块传输以重叠 prefill 计算；两端内存驻留时间；连接与拓扑选择（同机 NVLink 还是跨机 RDMA）（书 `09-分布式推理.md` §9.2.2）。
- 参考 SGLang `device_communicators/mooncake_transfer_engine.py`。

### 七、性能与排查（Q5）

**46. ★ nccl-tests 怎么用？输出看哪些列？**
- `all_reduce_perf -b 8 -e 8G -f 2 -g <每进程 GPU 数>`，多机用 MPI 启动；参数定义见 `third_party/nccl-tests/src/common.cu`（`minbytes`→`-b`、`stepfactor`→`-f`、`ngpus`→`-g`）。
- 看 time、algbw、busbw 与 #wrong；大消息 busbw 对比链路理论值（NVLink、网卡线速），小消息看 time（延迟）。

**47. ★★ 多机 AllReduce busbw 只有预期一半，怎么排查？**
- 分层：先 `ib_write_bw`（perftest，`third_party/perftest/src/write_bw.c`）验证每张网卡点对点线速和 GPU 内存（`--use_cuda`）；再单机 nccl-tests 验证机内；再两机。
- 看 `NCCL_DEBUG=INFO` 日志：是否选中全部 NIC（`NCCL_IB_HCA`）、是否走了 GDR（`NCCL_NET_GDR_LEVEL`）、是否误走 Socket、channel 数、ring 是否跨 NUMA。
- 网络侧：ECMP 冲突（多 QP）、PFC 暂停计数、ECN 标记、链路降速/误码；拓扑侧：NIC 与 GPU 不在同一 PCIe 交换。

**48. ★★ 训练任务 NCCL 超时/hang，怎么定位？**
- 先判断是"所有 rank 都在等"还是"某 rank 没到"：打印各 rank 最后一次集合调用序号与参数、py-spy/cuda-gdb 看卡在哪；NCCL RAS（`third_party/nccl/src/ras/`）与 flight recorder 类工具。
- 常见根因：调用不匹配、某 rank 异常退出、网络故障导致重传超时（`NCCL_IB_TIMEOUT`）、上游计算慢（straggler）。
- 书中 MegaScale 案例：网络带宽稳定，但各 rank 发起通信时间差增大，根因在前向期间的主机操作（书 `07-数据中心网络.md` §7.6.1）。

**49. ★★ 怎么区分"通信慢"还是"在等慢的计算"？**
- 集合通信的时间 = 等最晚 rank 到达 + 真正传输。profile 中看每个 rank 的发起时刻；对比 nccl-tests 同尺寸纯通信时间；看 straggler 分布（书 §10.4.5：N 张卡时步时间是 N 个样本的最大值）。
- 参考 `third_party/ai-infra-book/case-studies/collective-paths-and-diagnosis.md`。

**50. ★★ 列出你常用的 NCCL 环境变量及用途。**
- 观测：`NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS`、`NCCL_TOPO_DUMP_FILE`；选路：`NCCL_ALGO`、`NCCL_PROTO`、`NCCL_P2P_DISABLE`/`NCCL_P2P_LEVEL`、`NCCL_SHM_DISABLE`、`NCCL_NVLS_ENABLE`、`NCCL_CROSS_NIC`；网络：`NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、`NCCL_IB_GID_INDEX`、`NCCL_IB_TIMEOUT`、`NCCL_IB_QPS_PER_CONNECTION`、`NCCL_NET_GDR_LEVEL`；资源：`NCCL_MIN_NCHANNELS`、`NCCL_MAX_NCHANNELS`、`NCCL_BUFFSIZE`。
- 以上变量名均在 `third_party/nccl/src/` 中能找到读取或提示处（如 `NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 20)`、`NCCL_PARAM(CrossNic, "CROSS_NIC", 2)`、`NCCL_PARAM(BuffSize, "BUFFSIZE", -2)`）；各变量取值含义以对应源码为准。

**51. ★★★ 自动调优（如 AutoCCL）解决什么问题？代价是什么？**
- 单独 benchmark 选出的 channel/线程/分块配置在实际训练中因与计算争用而不再最优；AutoCCL 在训练中用真实通信时间做反馈，先选算法/协议/transport 再搜索线程、channel、分块（书 §6.4.3，论文数据：AllGather 18.26 → 32.44 GB/s）。
- 代价：搜索开销 S，需要 S/Δ 次调用才能收回（书 §6.4.3；案例 `case-studies/communication-tuning.md`）。

### 八、项目表述与开放题

**52. ★★ 介绍一个你做过的通信优化项目。（模板）**
- 背景：什么模型/规模/并行方式，瓶颈表现（步时间、通信占比、busbw）。
- 分析：用 α-β 模型或 profile 判断是延迟主导还是带宽主导、是否有等待偏差、拓扑是否被用满。
- 方案：算法/协议/拓扑/重叠/融合中的哪一个，为什么不选其他。
- 结果：量化收益 + 对照实验 + 适用边界。
- 本仓库可用素材：`labs/ccl_sim/`（算法模拟器 + α-β 代价模型，`python3 -m labs.ccl_sim`）、nccl-tests 扫参实验、Triton-distributed AG+GEMM 实验（见 [03-learning-path.md](../00-roadmap/03-learning-path.md)）。

**53. ★★★ 给你 1024 张 GPU 训练一个 MoE 模型，你如何规划通信？**
- 按带宽需求从高到低把 TP/EP 放入 NVLink 域，DP/PP 走 RDMA；估算每步各类通信量与可重叠窗口；选分层算法与 rail 对齐放置；EP All2All 的负载均衡与副本；监控 straggler 与网络拥塞。
- 参考：书 `07-数据中心网络.md` §7.6.4、`06-超节点.md` §6.7.3、`09-分布式推理.md` §9.4。

**54. ★★★ decode 阶段 TP8 AllReduce 怎么优化？**
- 消息小（10 KiB 量级），α 主导：书中算例八卡 14 轮、启动 11.51 μs 占绝大部分，带宽翻倍只省约 2.5 μs，α 减半省约 0.74 ms/128 次（书 §6.4.2，书中数值）。
- 手段：减少轮次（树/one-shot）、减少每轮开销（IPC 直接读写、CUDA Graph、NVLS）、与残差+RMSNorm 融合、量化通信数据。

## 待深入

- 补充 PyTorch `ProcessGroupNCCL`/DDP 相关题目的源码出处（本地未克隆 PyTorch）。
- 根据真实面试反馈增补题目与追问。
