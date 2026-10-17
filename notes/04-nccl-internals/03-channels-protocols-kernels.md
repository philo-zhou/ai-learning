# channel、协议（LL/LL128/Simple）、device primitives、enqueue 与 group

> 层次：中层 · 来源：third_party/nccl@3e46c08（NCCL 2.32.3）、ai-infra-book@3bdcb4fc 第 6 章 6.4.3~6.4.4 节

## 一句话

一个 NCCL 集合操作在主机侧被拆成"每个 channel 一段数据"的 work，打包进 kernel plan，用一次 `cuLaunchKernelEx` 发射，grid 的每个 block 对应一个 channel；block 内的线程用 primitives（`send/recv/recvReduceSend/...`）按 LL、LL128 或 Simple 协议在 peer 之间的环形缓冲上做"等对方 → 读/算/写 → 通知对方"的流水。

## 在调用链中的位置

- 上层：`ncclAllReduce`（`src/collectives.cc`）→ `ncclEnqueueCheck`（`src/enqueue/enqueue.cc:3478`）。
- 本模块：任务入队与调度（`src/enqueue/`、`src/group.cc`）、算法/协议选择（`src/tuning/`）、kernel 发射（`ncclLaunchKernel`）、设备代码（`src/device/`）。
- 下层：primitives 读写的缓冲由 transport 建立（P2P/SHM 见 [../02-scale-up/02-nccl-p2p-shm-nvls.md](../02-scale-up/02-nccl-p2p-shm-nvls.md)；NET 缓冲由 proxy 驱动，见 [04-proxy-and-net-plugin.md](04-proxy-and-net-plugin.md)）。Ring/Tree 每一步发什么见 [../03-ccl-algorithms/README.md](../03-ccl-algorithms/README.md)。

## 核心概念

### channel

- 主机侧 `struct ncclChannel`（`src/include/comm.h:169`）：`peers`（到每个 rank 的 send/recv `ncclConnector`）、`ring`、`tree`、`collnetChain`、`collnetDirect`、`nvls`、`id`、`workFifoProduced`。
- 设备侧 `struct ncclDevChannelPeer`（`src/include/device.h`）：`send[NCCL_MAX_CONNS]`、`recv[NCCL_MAX_CONNS]` 两组 `ncclConnInfo`。
- `struct ncclConnInfo`：`buffs[NCCL_NUM_PROTOCOLS]`（每种协议一块缓冲；注释 "Local for recv, remote for send"）、`tail`/`head`（生产者/消费者计数）、`stepSize`、`ptrExchange`（direct 模式交换指针）、`connFifo`（GPU 与 proxy 通信）、`step`。
- 上限：`MAXCHANNELS 64`（`device.h:93`）。每个 channel 是一条独立的 ring/tree 实例，多 channel 并行切分数据以用满多条 NVLink/多块网卡。
- `NCCL_STEPS 8`：每个连接的缓冲被切成 8 个 slot，形成生产者-消费者环形队列。

### 三种协议（`src/include/plugin/nccl_tuner.h`：`NCCL_PROTO_LL 0`、`LL128 1`、`SIMPLE 2`）

| 协议 | 同步方式 | 有效载荷比 | 适用 | 源码 |
|---|---|---|---|---|
| LL（Low Latency） | 每 16B 行 = `data1,flag1,data2,flag2`（`union ncclLLFifoLine`，`device.h:77`），接收方自旋直到两个 flag 等于期望值；用 `st.volatile.global.v4.u32` 一次写 16B | 50%（`ProtoLL::calcBytePerStep` 注释 "Half is data"） | 小消息、追求延迟 | `src/device/prims_ll.h`（`readLL`、`storeLL`） |
| LL128 | 128B 行中 120B 数据 + 8B flag（`NCCL_LL128_LINESIZE 128`、`NCCL_LL128_DATAELEMS = 15`），每 8 个线程一组，`flagThread = (tid % 8) == 7` 负责 flag；依赖 128B 写入的原子可见性 | 120/128 ≈ 94% | 中等消息，需要 NVLink 等保证 128B 顺序的路径 | `src/device/prims_ll128.h` |
| Simple | 大块数据 + 显式 `tail/head` 计数同步：写完数据后 fence 再更新计数 | ~100% | 大消息、追求带宽 | `src/device/prims_simple.h`（`waitPeer`、`postPeer`） |

三种协议在一个 slot 里的数据与同步信息布局如下（LL128 中每 8 个线程写一行，每线程 16 B，`tid % 8 == 7` 的线程把自己后 8 B 换成 flag，见 `prims_ll128.h` 中 `store128(..., flagThread ? flag : v[u + 1])`）：

```text
LL      one 16 B line (union ncclLLFifoLine), one 16 B volatile store
        byte 0        4        8        12       16
             ┌────────┬────────┬────────┬────────┐
             │ data1  │ flag1  │ data2  │ flag2  │      ← 8 B 数据 + 8 B flag：有效 50%
             └────────┴────────┴────────┴────────┘

LL128   one 128 B line = 16 x uint64, written by 8 threads x 16 B
        thread  t0       t1              t6       t7
             ┌────────┬────────┬─ ... ─┬────────┬────┬────┐
             │  16 B  │  16 B  │       │  16 B  │ 8B │flag│   ← 15 个 u64 数据 + 1 个 u64 flag
             └────────┴────────┴─ ... ─┴────────┴────┴────┘

Simple  one slot = stepSize bytes of pure data
             ┌─────────────────────────────────────────────┐
             │ data ...                                    │   ← 没有内嵌 flag，靠独立的
             └─────────────────────────────────────────────┘     tail/head 计数同步（第 7 节）
```

LL128 的启用条件见 `src/tuning/cost_model.cc::isLL128Enabled`：节点内路径需 `<= PATH_NVB`；网络侧 Hopper 及以上可到 `PATH_PXN`（受 `NCCL_LL128_C2C`），否则只到 `PATH_PXB`；cc < 70 禁用。这解释了"PCIe 机器上看不到 LL128"。

### 算法（`nccl_tuner.h`）

`TREE 0`、`RING 1`、`COLLNET_DIRECT 2`、`COLLNET_CHAIN 3`、`NVLS 4`、`NVLS_TREE 5`、`PAT 6`；字符串 `ncclAlgoStr = {"Tree","Ring","CollNetDirect","CollNetChain","NVLS","NVLSTree","PAT"}`、`ncclProtoStr = {"LL","LL128","Simple"}`（`src/init.cc:57-59`），即 `NCCL_ALGO`/`NCCL_PROTO` 接受的名字。

## 源码导读

下图先把第 1~6 节串成一条线：左边是调用线程，右边是 GPU；`ncclGroupEndInternal` 只有在 depth 归零时才往下走（第 2 节）。

```text
 host thread                                                     GPU
 ───────────                                                     ───
 ncclAllReduce (collectives.cc)
  └ ncclEnqueueCheck
     ├ ncclGroupStartInternal                  depth + 1
     ├ ArgsCheck → taskAppend ──► comm->planner
     └ ncclGroupEndInternal                    depth - 1, continue only if 0
        └ groupLaunch → groupLaunchLegacy
           ├ ncclPrepareTasks                  ncclGetAlgoInfo → algo / proto / channels
           └ doLaunches
              ├ ncclLaunchPrepare              scheduleCollTasksToPlan / scheduleP2pTasksToPlan
              ├ ncclLaunchKernelBefore_NoUncapturedCuda     uploadWork
              ├ ncclLaunchKernel ── cuLaunchKernelEx ══════► grid = nChannels blocks
              │                                               block b → channelId (channelMask)
              │                                               ncclKernelMain
              │                                                └ RunWorkColl<func, algo, proto>
              │                                                   └ runRing → Primitives
              └ ncclLaunchKernelAfter_NoCuda
                 └ hostStreamPlanTask ──► uploadProxyOps + ncclProxyStart   ← 只有需要 proxy 的 plan
```

### 1. API → 任务队列

- `src/collectives.cc::ncclAllReduce` → `ncclAllReduceConfigImpl`（打包 `ncclInfo`）→ `ncclEnqueueCheck`。
- `ncclEnqueueCheck`：`CommCheck`、revoked 检查 → `ncclGroupStartInternal` → `ncclCommEnsureReady` → `ArgsCheck` → `INFO(NCCL_COLL, "%s: opCount %lx sendbuff %p ...")`（`NCCL_DEBUG_SUBSYS=COLL` 时每次调用一行，可用来比对各 rank 调用序列是否一致）→ `taskAppend` → `comm->opCount++` → `ncclGroupEndInternal`。
- `taskAppend` 分派到 `collTaskAppend`（集合）、`p2pTaskAppend`（send/recv）、`ceCollTaskAppend`（copy engine 集合）、`rmaTaskAppend`（one-sided）等，把任务挂到 `comm->planner`，并把 comm 挂入线程局部的 `ncclGroupCommHead`。

### 2. group 语义（`src/group.cc`）

- `thread_local int ncclGroupDepth`：`ncclGroupStart` 加一，`ncclGroupEndInternal` 减一，只有减到 0 才真正执行（源码 `if ((--ncclGroupDepth) > 0) goto exit;`）。每个集合 API 内部都包了一层 Start/End，所以单独调用时立即执行，在用户 group 里则被攒起来。
- 用途（`docs/userguide/source/usage/groups.rst`）：单线程管理多 GPU 避免死锁、聚合多个操作减少 launch、把多个 send/recv 合并成一次（如实现 all-to-all），且在 group 内调用返回时操作未必已入 stream。
- 执行路径：`groupLaunch` → `groupLaunchLegacy`（默认）或 `groupLaunchEnqueueRearch`（`NCCL_ENQUEUE_REARCH_ENABLE=1`）→ 预处理（`ncclPrepareTasks`：按 (func, op, type) 聚合、调用 tuning 选算法/协议、决定是否需要按需建连 `ncclCollPreconnect`）→ `doLaunches`。
- `doLaunches`：把同一进程中属于同一"clique"（`intraComm0` 相同）的 comm 一起处理：`ncclLaunchPrepare` → 逐个 plan `ncclLaunchKernelBefore_NoUncapturedCuda` → `ncclLaunchKernel`（或 `ncclLaunchCeColl` / `ncclLaunchRma`）→ `ncclLaunchKernelAfter_NoCuda` → 最后 `ncclLaunchFinish`。`NCCL_LAUNCH_MODE=GROUP` 时多 comm 之间用 intra-process barrier 同步发射（`useBarrier`），默认 `PARALLEL`。

### 3. 选算法/协议/通道（`src/tuning/`）

- `ncclTuningInit`（`tuning.cc:44`）在 init 末尾建立模型；`cost_model.cc::ncclTuningCostModelInit` 读 `NCCL_PROTO`/`NCCL_ALGO`/`NCCL_SYM_KERNEL` 并用 `parseList` 解析（支持 `^` 取反和按函数前缀，如注释示例 `NCCL_PROTO="^LL128;allreduce:LL128"`）。
- 每个算法一个模型文件：`ring.cc`、`tree.cc`、`nvls.cc`、`collnet.cc`、`pat.cc`、`sym_model.cc`、`ce_model.cc`；每个任务由 `enqueue.cc::ncclGetAlgoInfo` 调 `ncclTuningCompute` 汇总所有候选、`ncclTuningSelectBestTuning` 选时间最小者。代价参数表 `ncclTunerConstantsDefaults`（`cost_model.cc`）包含 `baseLatencies` 与按 NVLINK/PCI/NET 分的 `hwLatencies`（单位 us），是 α-β 模型在 NCCL 中的落地。
- 所有候选都不可用时报 `No algorithm/protocol nor symKernelId available for function ...`，并在用户强制了 `NCCL_ALGO/PROTO` 时返回 `ncclInvalidUsage`。
- 外部 tuner 插件可覆盖结果（`src/plugin/tuner.cc`，`NCCL_TUNER_PLUGIN`）。
- 线程数：`tuning_general.cc` 中 `NCCL_NTHREADS`、`NCCL_LL128_NTHREADS`。

### 4. kernel plan 与发射（`src/enqueue/enqueue.cc`）

- `scheduleCollTasksToPlan` / `scheduleP2pTasksToPlan`：把任务按 channel 切块（`calcCollChunking` 决定 chunk 大小与 step），生成 `ncclDevWorkColl` / p2p work，组成 work batch；需要网络的部分调用 `ncclAddProxyOpIfNeeded` 生成 proxy op。
- `uploadWork`：把 work 写入 work FIFO（`NCCL_WORK_FIFO_BYTES`）或 kernel 参数；`uploadProxyOps` 提交 proxy op。
- `ncclLaunchPrepare`：处理用户 stream 与 NCCL 内部 deviceStream 的依赖；在 CUDA graph 捕获时，需要 proxy 的 plan 通过 host 任务（`hostStreamPlanTask`）把 proxy op 推给 proxy 线程（注释 "host tasks impose a high performance cost in CUDA"）。
- `ncclLaunchKernel`（`enqueue.cc:1886`）：`grid = {nChannels,1,1}`、`block = {threadPerBlock,1,1}`、动态共享内存 `ncclShmemDynamicSize`；设置 launch attribute（cluster 维度、`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`、launch completion event、programmatic stream serialization、`NVLINK_UTIL_CENTRIC_SCHEDULING`），用 `cuLaunchKernelEx` 或 `cuLaunchKernel` 发射。

### 5. 设备侧入口（`src/device/common.h`、`generate.py`）

- `generate.py` 枚举 `all_colls × all_redops × all_tys × all_algos × all_protos` 生成 `ncclDevFunc_*` 与少数特化的 `ncclDevKernel_*`；可用 `make ONLY_FUNCS="AllReduce Sum f32 * *"` 只编部分组合（文件内注释），缩短编译时间。
- `ncclKernelMain`：用 `channelMask` 的第 n 个置位把 `blockIdx.x` 映射到 `channelId`；warp 0 加载 `ncclKernelComm`，warp 1 加载 `ncclDevChannel`，其余 warp 加载 work batch 到共享内存 `ncclShmem`；然后循环执行：特化函数直接调用，否则查 `ncclDevFuncTable[funcId]`；`ncclShmem.aborted` 置位时退出（主机 abort 时 kernel 能退出的原因）。

### 6. 算法 → primitives（以 AllReduce Ring 为例）

`src/device/all_reduce.h::runRing`：构造 `Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(..., &ring->prev, &ring->next, ...)`，然后：

1. `directSend`：第一步发送自己的一块；
2. `directRecvReduceDirectSend` × (n-2)：收、归约、发（reduce-scatter 阶段）；
3. `directRecvReduceCopyDirectSend`：最后一次归约并写到输出，同时发出去（进入 all-gather）；
4. `directRecvCopyDirectSend` × (n-2)：收、拷贝、转发；
5. `directRecv`：收最后一块。

Tree 是 `runTreeUpDown`/`runTreeSplit`（`FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>` 向上 reduce，再反向 broadcast）；NVLS 是 scatter/gather 线程组加 `directRecvDirectSend` 走 multicast（见 [../02-scale-up/02-nccl-p2p-shm-nvls.md](../02-scale-up/02-nccl-p2p-shm-nvls.md)）。`RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>` 等特化把 (算法, 协议) 组合绑到具体实现。

### 7. Simple 协议的同步细节（`src/device/prims_simple.h`）

- `waitPeer`：发送方等 `connStepCache + NCCL_STEPS >= step + StepPerSlice`（对端 head 前进，说明有空 slot）；接收方等 tail 前进（有数据）。读计数用 `ld_volatile_global`（注释说明 volatile 比 acquire 快）。
- `postPeer`：数据写完后更新对端可见的计数（`st_relaxed_sys_global` 写 `tail/head`），需要时先 fence。
- 自旋超过 `NCCL_SPINS_BEFORE_CHECK_ABORT 10000` 次检查一次 abort 标志（`primitives.h`）。这就是 hang 时 GPU 100% 占用但不前进的原因：kernel 在 `waitPeer` 里自旋。

下图是一条 Simple 连接上 `NCCL_STEPS = 8` 个 slot 的生产者-消费者关系。按 `struct ncclConnInfo` 的注释，`buffs` 和 `tail` 在接收端本地（发送端远程写），`head` 在发送端本地（接收端远程写）。NET 连接中对端的角色由 proxy 承担，见 [04-proxy-and-net-plugin.md](04-proxy-and-net-plugin.md)：

```text
   sender (send conn)                          receiver (recv conn)
  ┌────────────────┐                          ┌──────────────────────────────────────┐
  │                │ ── (2) write slice ────► │ buffs[p], slot = step % NCCL_STEPS   │
  │                │                          │ ┌───┬───┬───┬───┬───┬───┬───┬───┐    │
  │                │                          │ │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │    │
  │                │                          │ └───┴───┴───┴───┴───┴───┴───┴───┘    │
  │                │ ── (3) postPeer: tail ─► │ tail                                 │
  │ head           │ ◄─ (5) postPeer: head ── │                                      │
  └────────────────┘                          └──────────────────────────────────────┘
  (1) sender   waitPeer: connStepCache(head) + NCCL_STEPS >= step + StepPerSlice   ← 有空 slot 才写
  (4) receiver waitPeer: connStepCache(tail) >= step + StepPerSlice                ← 有数据才读
```

## 性能要点与常见坑

- 小消息延迟主要由 kernel launch、LL 协议的 flag 轮询、算法步数决定；大消息带宽由 Simple 协议、通道数、链路决定。`NCCL_PROTO`/`NCCL_ALGO` 只用于定位问题，不建议生产中硬编码。
- 通道数 = block 数 = 占用 SM 数；与计算 kernel 重叠时会抢 SM（ai-infra-book 6.4.3 提到 AutoCCL 在实际并发干扰下调线程/通道/分块）。`NCCL_MIN_CTAS`/`NCCL_MAX_CTAS`、`NCCL_CTA_POLICY` 是控制入口。
- group 内的 send/recv 必须两端配对，否则 GPU 在 `waitPeer` 中永久等待。
- CUDA graph 捕获后，带网络的操作每次 replay 还要 host 回调推 proxy op，CPU 繁忙会拖慢。
- LL 协议在接收端是 GPU 轮询内存，使用 host 内存作缓冲时（如走 SHM 或无 GDR 的网络），PCIe 读延迟会放大。

## 面试自测

1. 问：LL、LL128、Simple 的本质区别？为什么 LL128 有平台限制？
   答：区别在"数据有效性如何判断"。LL 把 4B flag 和 4B 数据放在同一 8B 原子写里，接收方看到 flag 就能信任数据，无需 fence，但浪费一半带宽；LL128 以 128B 为单位，1/16 是 flag，依赖 128B 写入不被拆分重排，只有 NVLink 等路径能保证；Simple 用大块数据 + fence + 计数器，带宽最高但每步同步开销大。
2. 问：NCCL kernel 的 grid 大小怎么来的？一个 block 内做什么？
   答：`grid.x = nChannels`（`ncclLaunchKernel`），每个 block 负责一个 channel；block 内用前两个 warp 加载 comm/channel，其余加载 work，然后按 (func, algo, proto) 调用 `RunWorkBatch`/`RunWorkColl`，内部用 primitives 完成收发和归约。
3. 问：`ncclGroupStart/End` 的语义是什么？举两个必须用 group 的场景。
   答：延迟执行并合并；`ncclGroupDepth` 归零时统一调度发射。必须用的场景：一个线程管理多 GPU（否则第一个调用阻塞等其他 GPU）；多个 send/recv 组成 all-to-all（否则互相等待死锁）。
4. 问：hang 时 GPU 利用率 100%，kernel 在做什么？
   答：在 primitives 的 `waitPeer` 中自旋等对端 `tail/head` 前进，每 10000 次检查一次 abort 标志；说明对端没有发数据（没调用、调用参数不一致、网络/proxy 卡住）。
5. 问：`NCCL_ALGO=Ring NCCL_PROTO=Simple` 跑出来比默认快，说明什么？
   答：tuning 的代价模型对当前平台估计不准（或消息大小落在切换点附近）。应收集 `NCCL_DEBUG_SUBSYS=TUNING` 日志和 nccl-tests 全尺寸曲线，考虑 tuner 插件，而不是全局硬编码。

## 待深入

- `calcCollChunking` 中 chunkSize/sliceSteps 与 `NCCL_BUFFSIZE`、`NCCL_STEPS` 的关系。
- PAT 算法的设备实现与适用场景：`src/tuning/pat.cc`、`src/device/` 中 `NCCL_ALGO_PAT` 相关特化。
- 对称内存 kernel（`src/sym_kernels.cc`、`src/device/symmetric/`）与 nccl-tests `-R 2`、`-D` 参数的关系。
- 待确认：`NCCL_CTA_POLICY` 各取值（DEFAULT/EFFICIENCY/ZERO）对通道数的实际影响；确认方式：读 `src/init.cc:85-140` 与 `src/tuning/tuning_general.cc::ncclTuningGetChannels`。
