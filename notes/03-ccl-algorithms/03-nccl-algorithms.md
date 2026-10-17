# NCCL 中的算法与协议：Ring / Tree / CollNet / NVLS / PAT × LL / LL128 / Simple

> 层次：中层 · 来源：third_party/nccl@3e46c08（`src/tuning/`、`src/enqueue/`、`src/device/`、`src/graph/`、`src/config/algorithm_registry.cc`、`src/include/device.h`）、ai-infra-book@3bdcb4fc 第 6 章 6.4.2～6.4.4 节

## 一句话

NCCL 对每次集合调用，在"算法（怎么排通信拓扑）× 协议（每次传多大、怎么同步）"组成的候选表上，用一套带经验常数的 α-β 模型估算耗时，选最小者（可被环境变量、per-call 配置或 tuner 插件覆盖），再交给对应的 device kernel 模板执行。

## 在调用链中的位置

```
ncclAllReduce 等（src/collectives.cc）
  → taskAppend / collTaskAppend（src/enqueue/enqueue.cc）        任务入队
  → ncclGroupEnd → src/group.cc → ncclPrepareTasks（enqueue.cc）
       → ncclGetAlgoInfo（enqueue.cc）                           组装 ncclTuningInput_t
            → ncclTuningCompute（src/tuning/tuning.cc）
                 → ncclTuningComputeAllTunings：遍历每个 tuning id（= algo×proto，以及对称 kernel、CE 方法）
                      → 各模型的 *ModelSim（src/tuning/ring.cc、tree.cc、collnet.cc、nvls.cc、pat.cc …）
                 → tuner 插件 getCollInfo（可改写耗时表）
                 → ncclTuningSelectBestTuning：取最小耗时
                 → ncclTuningGetChannels：定 channel 数
  → 计算 chunk、生成 kernel plan、启动 kernel
  → device：RunWorkColl<func, T, RedOp, ALGO, PROTO>（src/device/all_reduce.h 等）
       → Primitives<..., Proto>（src/device/prims_simple.h / prims_ll.h / prims_ll128.h）
       → transport（P2P/SHM/NET，见 notes/04-nccl-internals/）
```

另外，本地版本还有一条新路径：`src/enqueue/task_prep/task_prep.cc` 的 `ncclTaskPrepare` 也对集合任务调用 `ncclTuningCompute`（跳过 AlltoAll、Scatter、Gather、AllGatherV），之后经 `task_classify.cc`、`task_posttuning.cc` 继续处理；`task_posttuning.cc` 中也调用了 `ncclGetAlgoInfo`。两条路径在什么条件下分别生效，见文末待确认。

## 核心概念

### 算法

算法与协议的编号定义在 `src/include/plugin/nccl_tuner.h`：`NCCL_ALGO_TREE 0`、`RING 1`、`COLLNET_DIRECT 2`、`COLLNET_CHAIN 3`、`NVLS 4`、`NVLS_TREE 5`、`PAT 6`；`NCCL_PROTO_LL 0`、`LL128 1`、`SIMPLE 2`。哪些组合存在、支持哪些原语，`src/config/algorithm_registry.cc` 的 `algRegistry[]` 表写得最清楚（该表注释要求与 `src/tuning/cost_model.cc` 的 `modelMap[]` 保持一致）：

| 算法 | 协议 | 支持的原语（algRegistry 的 collMask） | 思路 |
|---|---|---|---|
| RING | LL / LL128 / Simple | AllReduce、AllGather、ReduceScatter、Broadcast、Reduce | 环形，见 [02](02-alpha-beta-model.md)；带宽最优，α 项 O(n) |
| TREE | LL / LL128 / Simple | 仅 AllReduce | 节点内链、节点间 Double Binary Tree（`src/graph/trees.cc` 的 `ncclGetDtree`）；α 项 O(log nNodes) |
| COLLNET_DIRECT | Simple | AllReduce、AllGather、ReduceScatter | 借助 CollNet 网络插件做交换机内归约（如 SHARP 类能力）；节点内 rank 直接与网络端交互 |
| COLLNET_CHAIN | Simple | 仅 AllReduce | 节点内成链、链头与 CollNet 交互 |
| NVLS | Simple | AllReduce、AllGather、ReduceScatter | NVSwitch 内归约/组播（NVLink SHARP）：device 侧用 `multimem.ld_reduce` 等 PTX 指令（`src/device/reduce_kernel.h`） |
| NVLS_TREE | Simple | 仅 AllReduce | 节点内 NVLS + 节点间树；`src/tuning/nvls.cc` 中 `nNodes == 1` 时禁用 |
| PAT | Simple | AllGather、ReduceScatter | `src/tuning/pat.cc` 的 `ncclPatEnable`：要求 SM60+；非"每节点 1 rank"时要求多节点且启用 NVLS；不支持网卡设备卸载（`netDeviceType` 须为 HOST）；延迟模型含 `log2i(nNodes)` 项 |

device 侧实现与之对应：`src/device/broadcast.h`、`reduce.h` 只有 RING；`all_gather.h`、`reduce_scatter.h` 有 RING、PAT、NVLS、COLLNET_DIRECT；`all_reduce.h` 有 RING、TREE（三种协议）以及 COLLNET_DIRECT、NVLS、NVLS_TREE、COLLNET_CHAIN（仅 Simple）。

几个实现细节：

- **Ring AllReduce 的 kernel**（`src/device/all_reduce.h` 的 `runRing`）：把 channel 内数据按 `nranks * chunkCount` 为一轮循环；一轮里依次是 `directSend`（第 0 步）、`nranks-2` 次 `directRecvReduceDirectSend`（ReduceScatter）、1 次 `directRecvReduceCopyDirectSend`（得到最终结果并开始 AllGather）、`nranks-2` 次 `directRecvCopyDirectSend`、最后 `directRecv`。这正是 RS + AG 的 2(n-1) 步，且把"最后一次归约"和"第一次转发"融合成了一个原语。

  按 `runRing` 中的 chunk 下标公式展开 nranks = 4，位于环上第 `ringIx` 个位置的 rank 依次执行（chunk 号均 mod 4；每一步收到的块正是环上前一个 rank 上一步发出的块）：

  ```text
  t  primitive                         chunk         recv  reduce  send
  0  directSend                        ringIx-1       -      -      ✓     RS：发出自己那份
  1  directRecvReduceDirectSend        ringIx-2       ✓      ✓      ✓     RS：收 + 累加 + 转发
  2  directRecvReduceDirectSend        ringIx-3       ✓      ✓      ✓
  3  directRecvReduceCopyDirectSend    ringIx         ✓      ✓      ✓     RS 收尾：第 ringIx 块全和写入结果，同时作为 AG 第一份发出
  4  directRecvCopyDirectSend          ringIx-1       ✓      -      ✓     AG：收 + 写结果 + 转发
  5  directRecvCopyDirectSend          ringIx-2       ✓      -      ✓
  6  directRecv                        ringIx+1       ✓      -      -     AG：只收不发
  ```

  发送 6 次、接收 6 次，即 2(n-1)；下标与 [05](05-handwrite-ring-fullmesh.md) 题 1 伪代码（RS 第 s 步发 `r-s-1`，AG 第 s 步发 `r-s`）一致。
- **Tree AllReduce 的 kernel**：`runTreeSplit` 把线程分成两组，一组沿树向上 Reduce、一组向下 Broadcast，两阶段同时推进；`runTreeUpDown` 先上后下，只在特定 CUDA 版本（11.2～11.4、SM80+）的 Simple 协议下使用。
- **两棵树**：`ncclGetBtree` 构造"叶子与内部节点交替"的二叉树（注释中的 0-8-4-12… 图），`ncclGetDtree` 在 nranks 为偶数时取镜像、奇数时平移一位，得到第二棵树。树连接在 `src/graph/connect.cc` 的 `connectTrees` 中建立，环在 `connectRings`，另有 `connectCollNet`、`connectNvls`。

### 协议

协议决定"数据和同步标志怎么一起写到对端缓冲区"：

| 协议 | 线格式（`src/include/device.h`） | 同步方式 | 有效带宽（`src/tuning/` 中的系数） | 适用 |
|---|---|---|---|---|
| LL（Low Latency） | `union ncclLLFifoLine`：16 B = `data1, flag1, data2, flag2`，每 8 B 中 4 B 数据 + 4 B flag | 接收方轮询 flag，数据和 flag 由同一次 8 B 原子写落地，无需额外 fence | 约 50%：`ring.cc` 中 `busBw * .5`（另有 `llMaxBws` 上限）；`tree.cc` 中 `busBw * 1.0 / 3.8` | 小消息，延迟最低 |
| LL128 | `NCCL_LL128_LINESIZE 128`，`NCCL_LL128_DATAELEMS = LINEELEMS - 1`：每 128 B 中 120 B 数据 + 8 B flag | 以 128 B 为单位、依赖路径对 128 B 写入的顺序保证 | 理论 payload 占比 120/128 ≈ 94%；NCCL 的 Ring 代价模型取 92%（`src/tuning/ring.cc`：`busBw * (0.92 /*120.0/128.0*/)`），并受 `perChMaxRingLL128Bws` 限制 | 中等消息；只在满足条件的路径上启用 |
| Simple | 普通大块缓冲 | 写完一个 chunk 后用内存 fence + 标志通知 | 接近 100% | 大消息，延迟最高 |

两种低延迟协议的线格式（`src/include/device.h`）画出来就是下面的比例；LL 的 flag 紧跟在同一个 8 B 里的数据之后（`ncclLLFifoLine` 注释要求 flag 在数据之后），LL128 只示意"120 B 数据 + 8 B flag"的比例，flag 在行内的具体位置以 `src/device/prims_ll128.h` 为准：

```text
LL: one ncclLLFifoLine = 16 B
  byte 0        4        8        12       16
       ┌────────┬────────┬────────┬────────┐
       │ data1  │ flag1  │ data2  │ flag2  │
       └────────┴────────┴────────┴────────┘
       ◄─ 8 B atomic ───►◄─ 8 B atomic ───►        payload 8/16 = 50%

LL128: one line = NCCL_LL128_LINESIZE = 128 B = 16 x uint64
       ┌────────────────────────────────────────────────┬──────┐
       │ data: NCCL_LL128_DATAELEMS = 15 x uint64       │ flag │
       └────────────────────────────────────────────────┴──────┘
       ◄──────────────────── 120 B ────────────────────►◄ 8 B ►   payload 120/128 ≈ 94%

Simple: large chunk of plain data, then fence + flag per chunk     payload ≈ 100%
```

接收方只要轮询到 flag 等于本轮期望值，就知道同一行里的数据已经可用，不必再等一次单独的同步，这就是 LL/LL128 延迟低的原因；代价是每行都要让出一部分字节给 flag。

LL128 何时可用：`src/tuning/cost_model.cc` 的 `isLL128Enabled` 要求节点内路径 `intraType <= PATH_NVB`；节点间路径默认 `interType <= PATH_PXB`，在 SM90+ 且 `NCCL_LL128_C2C`（默认 1）开启时放宽到 `PATH_PXN`；还要求 SM70+、各 GPU 算力一致或都在 SM90+。原因（我的理解）：LL128 依赖"128 B 内的数据先于 flag 可见"，只有某些路径能保证。

### 代价模型怎么写的

每个模型在 Init 时为每个 (func, algo, proto) 预先算好 `generalLatencies` 和 `generalBandwidths`，Sim 时代入 `src/tuning/tuning_general.cc` 的 `ncclTuningGetTime`：

time(μs) = lat × latCount + nBytes / (1000 × bw)

其中 Ring 的 `latCount = numPipeOps`，其他算法按 `NCCL_MAX_DEV_WORK_BATCH_COLLS` 合批。这就是 α-β 模型：lat 是 α 的累加，bw 是"等效 algbw"。

- **Ring**（`ring.cc`）：`busBw = bw × nChannels`（bw 取 `bwIntra` 或 `bwInter`），`bandwidth = busBw × nRanks / nSteps`，`nSteps` 由 `ncclTuningGetNsteps` 给出（AllReduce 2(nRanks-1)，RS/AG nRanks-1）——对 AllReduce 恰好是 busBw × n / (2(n-1))，即 busbw 定义的反变换。延迟 = 基础延迟 + 节点内步数 × intraLat + 节点间步数 × interLat，多节点时还计入网络 post 开销（`NCCL_NET_OVERHEAD`）。多节点 AllReduce 的 Ring/Simple 还有"plateau"修正（`lat *= plateauFactor`）。
- **Tree**（`tree.cc`）：仅 AllReduce；`bandwidth = busBw × 0.5`；延迟 `2 * ((nRanks/nNodes - 1) * intraLat + log2i(nNodes) * interLat)`——节点内是链（线性）、节点间是树（对数）。中等消息再乘 `treeCorrectionFactor`（`cost_model.cc`，注释说树在中等大小上偏离模型，用静态系数修正）。
- **PAT**（`pat.cc`）：仅 RS/AG、仅 Simple；`bandwidth = busBw × 0.75`；延迟 `log2i(nNodes) * (interLat / 3.5) + nRanks * 2.8`（注释说线性部分希望以后去掉）。
- **NVLS / CollNet**（`nvls.cc`、`collnet.cc`）：AllReduce 的带宽按 `nRanks / nSteps` 换算，AG/RS 直接用 busBw；多节点 NVLS 的 AG/RS 需要 CollNet 插件支持 `iallgather`/`ireducescatter`。
- 常数表 `ncclTunerConstantsDefaults`（`cost_model.cc`）：`baseLatencies`（例如 Ring 的 LL/LL128/Simple 为 6.6/14.0/8.4 μs）和按 NVLINK/PCI/NET 分类的 `hwLatencies`（例如 NVLINK 上 Ring 为 0.6/1.9/3.4 μs，NET 上 Ring 为 2.7/4.0/14.0 μs）。这些常数可被 tuner 插件在 `init` 时改写（`tuning.cc` 的 `ncclTuningInit` 把 `&comm->tuningContext.tuningConstants` 传给 `comm->tuner->init`）。
- FP8 小细节：`ncclTuningGetTime` 对 fp8 且 `nRanks > 8` 的 Ring 乘一个很大的 `precision_ratio`，避免长链归约的精度损失。

可以看到 NCCL 的选择逻辑和 [02](02-alpha-beta-model.md) 的结论一致：LL 的 α 小、带宽打折，适合小消息；Simple 反之；Tree 的 α 项是 log(nNodes)，大规模时胜过 Ring。

### 人为干预

| 手段 | 位置 | 说明 |
|---|---|---|
| `NCCL_ALGO`、`NCCL_PROTO` | `cost_model.cc` 的 `ncclTuningCostModelInit` 用 `ncclGetEnv` 读取，`parseList` 解析 | 语法见 `parseList` 上方注释：`"ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"`，`^` 表示排除，如 `NCCL_PROTO="^LL128"`；env 强制的函数会覆盖 per-call 的 algSelection（`enqueue.cc` 中 `comm->tuningContext.forced[info->func]`） |
| `NCCL_SYM_KERNEL` | 同上 | 对称内存 kernel 的选择 |
| per-call algSelection | `ncclGetAlgoInfo` 中的 `info->algMask` | 选不到时按 `forceAlgSelection` 决定报错还是回退自动选择 |
| tuner 插件 | `src/include/plugin/nccl_tuner.h`；`ncclTuningCompute` 中调用 `comm->tuner->getCollInfo` | 插件拿到整张 algo×proto 耗时表，可改写某项（例如设为极小以强制选中） |
| `NCCL_PAT_ENABLE`、`NCCL_NVLS_ENABLE`、`NCCL_COLLNET_ENABLE` | `pat.cc`、`src/transport/nvls.cc`、`src/init.cc` | 开关对应算法 |
| `NCCL_NTHREADS`、`NCCL_LL128_NTHREADS`、`NCCL_MIN_NCHANNELS`、`NCCL_MAX_NCHANNELS`、`NCCL_BUFFSIZE`、`NCCL_LL_BUFFSIZE`、`NCCL_LL128_BUFFSIZE` | `tuning_general.cc`、`graph/connect.cc`、`init.cc` | 线程数、channel 数、各协议缓冲区大小 |
| 观察选择结果 | `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` | `ncclTuningInit` 在 rank 0 用 `INFO(NCCL_TUNING, ...)` 打印各算法/协议的延迟与带宽表；`ncclGetAlgoInfo` 打印 `Best tuning, algorithm, %s, protocol, %s` |

## 源码导读

| 路径 | 函数 / 符号 | 作用 |
|---|---|---|
| `src/enqueue/enqueue.cc` | `ncclPrepareTasks`、`ncclGetAlgoInfo`、`calcCollChunking`、`taskAppend` | 组装调优输入、调用 `ncclTuningCompute`、确定 chunk；AlltoAll/Gather/Scatter 拆成 P2P |
| `src/enqueue/task_prep/task_prep.cc` | `ncclTaskPrepare` | 新的任务准备路径，对集合任务调用 `ncclTuningCompute` |
| `src/tuning/tuning.cc` | `ncclTuningInit`、`ncclTuningCompute`、`ncclTuningComputeAllTunings`、`ncclTuningSelectBestTuning` | 调优总入口；遍历所有候选；接入 tuner 插件；取最小 |
| `src/tuning/cost_model.cc` | `parseList`、`isLL128Enabled`、`ncclTunerConstantsDefaults`、`treeCorrectionFactor`、`modelMap[]`、`ncclTuningCostModelInit` | 环境变量解析、LL128 条件、常数表、模型注册 |
| `src/tuning/cost_model.h` | 各 `ncclTuning*ModelInit/Sim` 声明 | 模型接口：Init 预计算、Sim 估算单次调用 |
| `src/tuning/ring.cc`、`tree.cc`、`pat.cc`、`nvls.cc`、`collnet.cc` | `ncclTuning<Algo>ModelInit/Sim` | 各算法的 α-β 模型 |
| `src/tuning/tuning_general.cc` | `ncclTuningGetNsteps`、`ncclTuningGetTime`、`ncclTuningGetChannels`、`ncclTuningSetThreadThresholds` | 公共工具：步数、耗时公式、channel 与线程数 |
| `src/include/tuning.h` | `ncclTuningInput_t`、`ncclTuningResult_t`、`ncclTuningContext_t` | 调优输入输出与 per-comm 状态（`generalLatencies`、`generalBandwidths`） |
| `src/config/algorithm_registry.cc` | `algRegistry[]` | 名字（如 `RING_LL128`）↔ algo/proto ↔ 支持的原语 |
| `src/include/plugin/nccl_tuner.h` | `NCCL_ALGO_*`、`NCCL_PROTO_*` | 编号定义；tuner 插件接口 |
| `src/device/all_reduce.h` | `runRing`、`runTreeUpDown`、`runTreeSplit`、`RunWorkColl<ncclFuncAllReduce, ...>` 各特化 | AllReduce 各算法 kernel |
| `src/device/all_gather.h`、`reduce_scatter.h`、`broadcast.h`、`reduce.h` | `RunWorkColl<...>` 特化，PAT 用 `PatAGAlgorithm`/`PatRSAlgorithm`（定义在 `src/include/collectives.h`） | 其他原语的 kernel |
| `src/device/primitives.h` | `ProtoSimple`、`ProtoLL`、`ProtoLL128` | 协议的编译期描述 |
| `src/device/prims_simple.h`、`prims_ll.h`、`prims_ll128.h` | `Primitives<..., Proto...>` 特化 | 各协议的收发/归约原语实现 |
| `src/include/device.h` | `ncclLLFifoLine`、`NCCL_LL128_LINESIZE`、`NCCL_STEPS`（=8） | 协议线格式与 FIFO 深度 |
| `src/graph/trees.cc`、`rings.cc`、`connect.cc`、`search.cc` | `ncclGetBtree`、`ncclGetDtree`、`connectRings`、`connectTrees`、`connectNvls`、`connectCollNet` | 拓扑搜索与环/树连接（细节在 `notes/04-nccl-internals/`） |

## 性能要点与常见坑

- 不要凭经验硬设 `NCCL_ALGO`/`NCCL_PROTO`：它会覆盖所有消息大小的自动选择，常见后果是小消息变慢。先用 `NCCL_DEBUG_SUBSYS=TUNING` 看模型选了什么。
- Tree 只用于 AllReduce；AllGather/ReduceScatter 的低延迟选项是 PAT（满足条件时）或 NVLS。
- NVLS、CollNet 需要硬件与插件支持，且只有 Simple 协议；不满足条件时模型直接把对应项置为无效。
- LL128 在不满足路径条件时被禁用，这就是同样的代码在不同机型上中等消息性能差异很大的原因之一。
- 模型里很多常数（`hwLatencies`、`treeCorrectionFactor`、plateau 系数）是经验值，新硬件上可能不准；大规模集群常用 tuner 插件按实测改写。
- busbw 低时先分清：是选错了算法/协议（看 TUNING 日志），还是 channel 太少、拓扑搜索结果差（看 GRAPH 日志）。

## 面试自测

1. **NCCL 怎么选算法和协议？**
   要点：每个 (algo, proto) 一个 α-β 模型，Init 时按拓扑预计算延迟与带宽，调用时 `time = lat × latCount + nBytes / bw`，遍历候选取最小；可被 `NCCL_ALGO`/`NCCL_PROTO`、per-call 配置、tuner 插件覆盖。
2. **LL、LL128、Simple 的区别？**
   要点：LL 每 8 B 中 4 B 数据 4 B flag，靠 8 B 原子写同时落地，免 fence，延迟最低、带宽约一半；LL128 每 128 B 中 120 B 数据，理论带宽约 94%（120/128，NCCL Ring 代价模型按 0.92 估），依赖路径对 128 B 写入的顺序保证；Simple 大块 + fence + flag，带宽最高、延迟最高。
3. **NCCL 的 Tree 是什么样的树？为什么节点内是链？**
   要点：节点间用 Double Binary Tree（镜像/平移得到第二棵）；节点内 GPU 由 NVLink/NVSwitch 高带宽连接，链的线性延迟项很小，且链便于流水；模型中延迟为 `2*((ppn-1)*intraLat + log2(nNodes)*interLat)`。
4. **NVLS 为什么能让 AllReduce 的 busbw 很高？**
   要点：归约在 NVSwitch 内完成（multimem 指令），每个 GPU 只需发送一次、接收一次，绕开了"端点归约"下 2(n-1)/n·M 的流量。
5. **PAT 用于什么场景？**
   要点：AllGather/ReduceScatter，延迟含 log(nNodes) 项，主要面向每节点 1 rank 的大规模场景，弥补 Ring 在 AG/RS 上 O(n) 延迟的问题；需满足 `ncclPatEnable` 的条件。

## 待深入

- 待确认：PAT 的全称（常见说法是 Parallel Aggregated Trees）；确认方式：本地源码未出现全称，需查 NCCL release notes 或 `third_party/nccl` 下的文档。
- 待确认：`ncclPrepareTasks`（`enqueue.cc`）与 `ncclTaskPrepare`（`task_prep/task_prep.cc`）两条路径在当前 commit 中的启用条件。已读到：二者都在 `src/group.cc` 中被调用——前者在 `ncclPrepareTasksAndCollPreconnectFunc` 和同步分支中，后者在 `ncclTaskPrepareJobFunc` 中；确认方式：读 `src/group.cc` 中创建这两种 async job 的分支条件。
- 待确认：`ncclTuningGetChannels` 如何根据消息大小和 `threadThresholds` 减少 channel 数；确认方式：读 `third_party/nccl/src/tuning/tuning_general.cc` 第 141 行起。
- 对称内存 kernel（`src/sym_kernels.cc`、`src/tuning/sym_model/`）与 CE（copy engine）集合（`src/ce_coll.cc`、`src/tuning/ce_model.cc`）也参与同一张候选表，留到 `notes/04-nccl-internals/`。
