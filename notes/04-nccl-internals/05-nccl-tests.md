# nccl-tests：结构、参数与 algbw/busbw

> 层次：中层（实践工具） · 来源：third_party/nccl-tests@afd59ab（版本 2.21.1）、third_party/nccl@3e46c08、ai-infra-book@3bdcb4fc 第 6 章 6.4.3 节、第 7 章

## 一句话

nccl-tests 是 NVIDIA 官方的集合通信基准：一份公共框架（`src/common.cu`）负责参数解析、多进程/多线程/多 GPU 组织、计时与校验，每个集合一个小文件（`src/all_reduce.cu` 等）只提供"缓冲大小怎么算、数据怎么初始化、怎么调用 NCCL、带宽怎么换算"四个回调；输出里的 busbw 是可以直接和硬件链路带宽对比的归一化带宽。

## 在调用链中的位置

- 上层：使用者（性能验收、回归测试、故障定位）。
- 本模块：`third_party/nccl-tests/src/`，链接 `libnccl`；多机需 `MPI=1` 编译后用 `mpirun` 启动。
- 下层：直接调用 `ncclCommInitRank`、`ncclAllReduce`、`ncclGroupStart/End` 等公开 API，内部路径见 [03-channels-protocols-kernels.md](03-channels-protocols-kernels.md)。
- 使用方法论见 [../07-perf-troubleshooting/02-perf-methodology.md](../07-perf-troubleshooting/02-perf-methodology.md)。

## 核心概念

### 目录与构建

- `src/Makefile`：`BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall alltoallv scatter gather sendrecv hypercube`，每个生成 `build/<name>_perf`；另有 `comm_ops_perf`（init/split/shrink/grow 延迟，需 `MPI=1`）。
- 构建：`make CUDA_HOME=... NCCL_HOME=...`；多机：`make MPI=1 MPI_HOME=...`（README "Build" 节）。
- rank 总数 = 进程数 × `-t` 线程数 × `-g` 每线程 GPU 数（README "Usage" 节）。

### 框架抽象（`src/common.h`）

```c
struct testColl {
  const char name[20];
  void (*getCollByteCount)(...);   // 每 rank 发送/接收元素数、in-place 偏移
  testResult_t (*initData)(...);   // 填输入、算期望结果
  void (*getBw)(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(...);    // 调用 NCCL
};
```

每个 `*.cu` 定义自己的 `struct testEngine ncclTestEngine`（如 `all_reduce.cu:580`），`common.cu::main` → `run()` → `threadRunTests` → `TimeTest` → `BenchTime`。

展开这条调用链（`threadRunTests` 与 `TimeTest` 之间隔着各集合自己的 `runTest`，如 `AllReduceRunTest` 先设置 `args->collTest = &allReduceTest`），可以看到 4 个回调各在哪一步被调用：

```text
 common.cu::main → run()
  └ threads[t].func = threadRunTests            one per -t thread
     └ ncclTestEngine.runTest                   e.g. AllReduceRunTest (all_reduce.cu)
        ├ args->collTest = &allReduceTest
        └ TimeTest                              per (type, op)
           └ for size in -b .. -e  (x -f, or + -i)
              ├ setupArgs ──► collTest->getCollByteCount     ← 算每 rank 元素数
              ├ BenchTime(in_place = 0)                      ← out-of-place 列
              └ BenchTime(in_place = 1)                      ← in-place 列

 BenchTime
  ├ warmup : startColl x -w ──► collTest->runColl, then completeColl
  ├ timed  : -n iters x -m agg_iters   (ncclGroupStart/End when -m > 1;
  │                                     -G: capture once, cudaGraphLaunch x -G)
  ├ deltaSec = elapsed / (iters x agg_iters) ──► Allreduce over ranks (-a)
  ├ collTest->getBw(count, typesize, deltaSec, &algBw, &busBw, nranks)
  └ -c checks: collTest->initData → startColl → completeColl → CheckData ──► #wrong
```

### algbw / busbw 计算位置

| 集合 | 函数（文件） | algbw | busbw 因子 |
|---|---|---|---|
| AllReduce | `AllReduceGetBw`（`src/all_reduce.cu:59`） | `count*typesize / 1e9 / sec` | `2(n-1)/n` |
| AllGather | `AllGatherGetBw`（`src/all_gather.cu:38`） | `count*typesize*nranks / 1e9 / sec` | `(n-1)/n` |
| ReduceScatter | `ReduceScatterGetBw`（`src/reduce_scatter.cu:37`） | `count*typesize*nranks / 1e9 / sec` | `(n-1)/n` |
| AlltoAll | `AlltoAllGetBw`（`src/alltoall.cu:47`） | `count*nranks*typesize / 1e9 / sec` | `(n-1)/n` |
| Broadcast | `BroadcastGetBw`（`src/broadcast.cu:34`） | `count*typesize / 1e9 / sec` | `1` |
| SendRecv | `SendRecvGetBw`（`src/sendrecv.cu:38`） | `count*typesize / 1e9 / sec` | `1` |

`count` 的含义：`BenchTime` 中 `count = args->nbytes / wordSize(type)`，`nbytes` 来自 `setupArgs` 里各集合 `*GetCollByteCount` 返回的 `paramcount`。AllGather/ReduceScatter/AlltoAll 的 `paramcount` 都是 `(size/eltSize/nranks)` 向 16B 对齐后的每 rank 份额，所以它们的 algbw 分子乘 `nranks` 后等于命令行 size（总数据量）；AllReduce 的 `paramcount` 就是 size 本身。也就是说，nccl-tests 的 `-b/-e` 对这些集合都表示"总缓冲大小"。

单位：GB/s，按 1e9 计（不是 GiB）。调用点：`common.cu::BenchTime` 中 `args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus)`。

推导（`doc/PERFORMANCE.md`）：AllReduce 每个元素要做 n-1 次加法和 n 次赋值，最少 `2(n-1)` 次传输，分摊到 n 条每条带宽 B 的链路，`t = (S/B) * 2(n-1)/n`，所以 `B = algbw * 2(n-1)/n`。文档强调这与具体用 ring 还是 tree 无关（只要是点对点 send/recv 实现）。ai-infra-book 6.4.3 给出实例：两台 HGX H100 共 16 rank、64 MiB AllReduce 466.1 us，algbw 143.99 GB/s，busbw 为其 1.875 倍（= 2×15/16）即 269.98 GB/s，并提醒 busbw 是归一化指标，不能直接当作每张网卡的实际吞吐（分层、交换机内归约会改变实际流量）。

把上面的推导和书中实例摆成一条换算线（数字均来自上文，未在本机运行）：

```text
 user view                              per-link view (n ranks, n links of bandwidth B)
 ─────────                              ───────────────────────────────────────────────
 S bytes all-reduced in time t          each link carries >= S x 2(n-1)/n
 algbw = S / t           ──── x 2(n-1)/n ────►   busbw = algbw x 2(n-1)/n

 example: n = 16, S = 64 MiB = 67,108,864 B, t = 466.1 us
   algbw = 67,108,864 / 1e9 / 466.1e-6   ≈ 144.0 GB/s    ← 书中给出 143.99
   busbw = algbw x (2 x 15 / 16)         = algbw x 1.875 ≈ 270.0 GB/s   ← 书中给出 269.98
```

### 计时方式（`common.cu::BenchTime`）

1. warmup `-w` 次（不计时）；
2. 计时 `-n` 次外层迭代，每次内含 `-m` 个聚合操作（`-m > 1` 时多个操作放进一个 group）；
3. `deltaSec = elapsed / (iters * agg_iters)`，CUDA graph 模式（`-G`）再除以 launch 次数；
4. `Allreduce(args, &deltaSec, average)` 在 rank 间按 `-a` 取平均/最小/最大；
5. 计算 algbw/busbw；`-c` 次正确性检查，`#wrong` 列为错误元素数；
6. 末尾打印 `# Avg bus bandwidth : X`，若设 `NCCL_TESTS_MIN_BW` 且低于 90% 则标 `FAILED`（`common.cu:1931`、`util.cu:902`）。

输出列（`util.cu`）：`size count type redop root | time algbw busbw #wrong`（out-of-place）`| time algbw busbw #wrong`（in-place）；`-I 1` 时增加 `i_min i_max i_p99 i_cv%` 每次迭代统计列。

## 源码导读

| 路径 | 看什么 |
|---|---|
| `src/common.cu::main`、`run` | 参数解析、MPI/线程启动、`NCCL_TESTS_SPLIT`/`NCCL_TESTS_SPLIT_MASK` 分组（`common.cu:1588`） |
| `src/common.cu::initComms` | 如何调用 `ncclCommInitRank`（多 GPU 单线程时包 group） |
| `src/common.cu::startColl` | `nGpus > 1` 时 `ncclGroupStart`，循环调用 `runColl` |
| `src/common.cu::BenchTime` | warmup、graph 捕获、计时、getBw、校验 |
| `src/all_reduce.cu` | `AllReduceGetCollByteCount`、`AllReduceInitData`、`AllReduceGetBw`、device API 实现（`-D`，NCCL ≥ 2.28 的 LSA/multimem kernel） |
| `src/util.cu` | 表头、结果行与 JSON 输出 |
| `doc/PERFORMANCE.md` | algbw/busbw 定义与各集合的推导 |

## 常用参数（README "Arguments" 节）

| 参数 | 含义 | 默认 |
|---|---|---|
| `-t` / `-g` | 每进程线程数 / 每线程 GPU 数 | 1 / 1 |
| `-b` / `-e` | 起始 / 结束字节数 | 32M / 32M |
| `-f` / `-i` | 倍增因子 / 固定步长（二选一） | 关 / 1M |
| `-o` / `-d` / `-r` | 归约操作 / 数据类型 / root | sum / float / 0 |
| `-n` / `-w` / `-m` | 计时迭代 / warmup / 每次聚合操作数 | 20 / 1 / 1 |
| `-a` | 多进程结果聚合：0=Rank0, 1=Avg, 2=Min, 3=Max | 1 |
| `-c` | 正确性检查次数 | 1 |
| `-z` | 阻塞模式 0~3 | 0 |
| `-G` | CUDA graph 捕获后 replay 次数 | 0 |
| `-C` | 报告 CPU 时间而非延迟 | 0 |
| `-R` | 本地注册(1)/对称注册(2) | 0 |
| `-D` / `-V` | device API 实现编号 / CTA 数 | 0 / 16 |
| `-I` / `-K` | 每迭代计时统计 / 跳过前 K 个样本 | 0 / 0 |
| `-T` | 每个测试超时秒数 | 关 |
| `-J` | JSON 输出文件 | 无 |
| `-U` | 打印 NCCL tuning 信息（NCCL ≥ 2.28） | 0 |
| `-M` | 显存使用报告 | 0 |

典型命令（README "Quick examples"）：

```shell
# 单机 8 卡，8B 到 128MiB，每次翻倍
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
# 8 节点 64 卡，每进程 1 卡
mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

`NCCL_TESTS_SPLIT="AND 0x7"`（或 `MOD 8`）在 8 卡机器上切成 8 个各含"每节点 1 卡"的组，纯测跨节点网络；`"OR 0x7"`（或 `DIV 8`）每节点一组，纯测节点内（README "Running multiple operations in parallel"）。注意 README 提醒报告的是每组带宽。

## 性能要点与常见坑

- 看小消息看 `time`（延迟），看大消息看 `busbw`；中间段是协议/算法切换区（结合 `NCCL_DEBUG_SUBSYS=TUNING` 或 `-U 1` 看选了什么）。
- 默认 `-b/-e` 都是 32M，只测一个尺寸；做曲线必须显式给 `-b 8 -e 8G -f 2`。
- `-n 20 -w 1` 对大规模或首次建连不够：默认懒连接（见 [01-init-bootstrap.md](01-init-bootstrap.md)），首个尺寸可能偏慢；加大 `-w`。
- `-a 1` 报的是 rank 平均；定位慢节点时用 `-a 2`/`-a 3` 或 `-I 1` 看抖动。
- 带宽单位是 1e9 字节/秒；与 `ib_write_bw` 的 Gb/s（比特）或 MiB/s 对比时注意换算。
- busbw 对 Broadcast/SendRecv 因子是 1，不能和 AllReduce 的 busbw 直接横比算法优劣。
- 进程绑核、GPU 可见性（`CUDA_VISIBLE_DEVICES`）、MPI 的网卡选择都会影响结果，测基线前固定这些条件。

## 面试自测

1. 问：algbw 和 busbw 的区别？AllReduce 的 busbw 因子为什么是 2(n-1)/n？
   答：algbw = 用户数据量 / 时间；busbw 把它换算成"每个 rank 链路上需要的最小带宽"，便于与硬件峰值比较。AllReduce 每个元素需要 2(n-1) 次传输分摊在 n 条链路上，所以乘 2(n-1)/n。
2. 问：8 卡 H100 单机 AllReduce 大消息 busbw 远低于预期，怎么排查？
   答：确认 `-b/-e` 覆盖大尺寸；`NCCL_DEBUG=INFO` 看是否走 `P2P/CUMEM` 或 `P2P/IPC`（而不是 SHM）、通道数、是否用 NVLS；`nvidia-smi topo -m` 和 NVLink 状态确认链路；对比 `NCCL_ALGO`/`NCCL_PROTO` 强制结果；检查其他进程占用 GPU。详见 [../07-perf-troubleshooting/02-perf-methodology.md](../07-perf-troubleshooting/02-perf-methodology.md)。
3. 问：如何用 nccl-tests 分别测节点内和节点间带宽？
   答：`NCCL_TESTS_SPLIT="DIV 8"`（每节点一组，节点内）与 `NCCL_TESTS_SPLIT="MOD 8"`（每组每节点 1 卡，节点间）；或单机 `-g 8` 跑节点内基线。
4. 问：`-m` 和 `-G` 分别在模拟什么？
   答：`-m` 把多个操作放进一个 group（聚合 launch，模拟框架的 bucket/coalescing）；`-G` 用 CUDA graph 捕获后 replay，去掉 CPU launch 开销，模拟 graph 化的推理/训练。
5. 问：`#wrong` 非 0 说明什么？
   答：数据校验失败：可能是硬件/链路错误（NVLink、PCIe、网卡 bit flip）、P2P 在 ACS/IOMMU 配置下不可用、驱动或插件 bug；应停止性能测试，先用 `NCCL_RUN_DIAGNOSTICS=1`、单链路测试定位。

## 待深入

- `-D` device API 实现（`src/all_reduce.cu` 中 `allReduceLsaKernel`、`allReduceMultimemKernel`）与 NCCL device API（`src/include/nccl_device/`）的对应关系。
- `verifiable/` 目录中数据生成与校验库的实现。
