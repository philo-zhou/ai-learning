# torch.distributed 与 ProcessGroupNCCL（概念）

> 层次：上层 · 来源：PyTorch 源码**未克隆**（C++ 部分全部待确认）；Python 前端引用 ai-infra-book@3bdcb4fc 内附的 torch 2.14.0 快照 `third_party/ai-infra-book/experiments/ch07/07-08/framework-sources/distributed_c10d.py` 与 `distributed.py`（来源与 torch git commit 见同目录 `framework.json`）；third_party/vllm@02653fa、third_party/DeepEP@b5a33ee 作调用佐证

## 一句话

`torch.distributed` 用 ProcessGroup 抽象一组 rank，NCCL 后端（ProcessGroupNCCL）在内部为每个设备/组懒创建 `ncclComm_t`、在独立的 NCCL stream 上发起通信、用 CUDA event 与计算 stream 互相等待，并用 watchdog 线程检测超时与异步错误——训练框架的几乎所有通信、推理框架的兜底通信都走这里。

## 在调用链中的位置

```text
DDP / FSDP / Megatron / vLLM 兜底 / DeepEP（复用 comm）
  └─ torch.distributed.all_reduce(tensor, group, async_op)      distributed_c10d.py
       └─ ProcessGroup.allreduce([tensor], opts) → Work
            └─ ProcessGroupNCCL（C++，待确认路径）
                 ├─ 取/建 ncclComm_t（按 device + group）
                 ├─ NCCL stream 等待计算 stream 上的 event
                 ├─ ncclAllReduce(..., ncclStream)
                 ├─ 记录完成 event → WorkNCCL
                 └─ watchdog 线程轮询 Work 完成/超时/ncclCommGetAsyncError
```

源码位置（全部待确认；确认方式：克隆 pytorch/pytorch 后查 `torch/csrc/distributed/c10d/`，预计在 `ProcessGroupNCCL.hpp/.cpp`、`NCCLUtils.hpp/.cpp`、`Work.hpp`、`FlightRecorder*` 等文件中）。

## 核心概念

### 1. ProcessGroup 与后端

- `init_process_group(backend, init_method/store, world_size, rank, timeout, device_id, pg_options)` 建默认组（WORLD）；`new_group(ranks)` / `split_group(parent_pg, split_ranks)` 建子组（快照中 `def init_process_group` L2290、`def new_group` L6811、`def split_group` L6550）。
- 一个 ProcessGroup 可以按设备类型挂多个 backend（例如 CPU→gloo、CUDA→nccl）；`_new_process_group_helper` 负责创建，`pg.bound_device_id` 记录绑定设备。
- rendezvous 用 Store（TCPStore 等）交换信息；NCCL 的 `ncclUniqueId` 就是通过 Store 由 rank0 广播给其他 rank（概念；具体 key 命名待确认）。
- vLLM 的做法相同：设备组 `backend="nccl"`、CPU 组 `backend="gloo"`（`third_party/vllm/vllm/distributed/parallel_state.py` `GroupCoordinator.__init__`）。

### 2. NCCL comm 创建时机

- **默认懒初始化**：第一次在某设备上发起集合通信时才 `ncclCommInitRank`——所以“第一次 AllReduce 特别慢 / 在第一次 collective 处 hang”很常见，本质是建连（bootstrap、拓扑探测、channel/transport 建立）发生在那时。
- **eager 初始化**：`init_process_group(device_id=...)` 时，快照文档写明在 NCCL 下有两个效果：communicator 立即建立（立即调用 `ncclCommInit*`），子组尽量用 `ncclCommSplit` 创建以降低开销；也可借此提前暴露初始化错误。代码上体现为 `eager_backend.eager_connect_single_device(device_id)`（快照 `_new_process_group_helper` 内，`supports_splitting` 时）。
- 子组与 split：每个 `new_group` 本质上是一个新的 NCCL communicator，资源（channel buffer、proxy 线程、QP）随组数增长；`ncclCommSplit`（`third_party/nccl/src/nccl.h.in` 有声明）可共享父 comm 的部分资源。
- 非成员 rank 调用 `new_group` 也必须参与（集合语义），这是“新建组时 hang”的常见原因。

### 3. stream 与 event 同步

- ProcessGroupNCCL 为每个设备维护**独立的 NCCL stream**（非用户当前 stream），因此通信天然可以与计算重叠。
- 发起时：在当前（计算）stream 上记录 event，NCCL stream 等待它 → 保证输入已经算好。
- 完成时：在 NCCL stream 上记录完成 event，存入 Work。
- `work.wait()`（NCCL 后端）语义：让**当前 stream 等待完成 event**，而不是阻塞 CPU（除非设置 `TORCH_NCCL_BLOCKING_WAIT`）；之后在当前 stream 上使用结果是安全的。
- 张量生命周期：输入/输出张量在通信 stream 上被使用，CUDA caching allocator 需要知道这一点——默认通过 `recordStream` 标记；`TORCH_NCCL_AVOID_RECORD_STREAMS=1` 改为在 Work 中持有张量引用直到完成，减少显存碎片/滞留。Llama 3 论文在 PP 异步 P2P 中开启了它以降低显存（`third_party/ai-infra-book/references/text/llama3.txt` 第 619 行附近）。
- `ProcessGroupNCCL.Options` 的 `is_high_priority_stream`：让 NCCL stream 用高优先级，在有计算 kernel 排队时优先调度通信（快照 `init_process_group` 的 `pg_options` 文档）。
- 与 vLLM pynccl 对比：pynccl 在**当前 stream** 上发起通信（`third_party/vllm/vllm/distributed/device_communicators/pynccl.py` `all_reduce(..., stream=None)` 默认 `current_stream()`），没有 event 交接，顺序简单、CUDA Graph 友好，但不能自动重叠。

把上面几条串成一次 `all_reduce(t, async_op=True)` 的时序（概念图，C++ 实现待确认）：

```text
 CPU thread              compute stream (current)         NCCL stream (per device)
 ──────────              ────────────────────────         ────────────────────────
 y = matmul(...)    ──►  [ matmul kernel ]
 work = all_reduce  ──►  record ev_in ──────────────────► wait(ev_in)
   (async_op=True)                                        [ ncclAllReduce kernel ]
   立即返回 WorkNCCL                                      record ev_done（Work 持有它）
 other_compute()    ──►  [ other kernels ]   ◄── 与通信并行（重叠）
 work.wait()        ──►  wait(ev_done) ◄───────────────── ev_done
 use(t)             ──►  [ kernel reading t ]             安全：已排在 ev_done 之后
 t.cpu() / torch.cuda.synchronize()                       ← 到这里 CPU 才真正等到结果
```

对比 pynccl：没有右边那条 stream，`ncclAllReduce` 直接排在 compute stream 上，因而没有 event 交接，也就不能和后面的计算重叠。

### 4. async_op 与 Work

- `all_reduce(tensor, op, group, async_op=False)`：内部构造 `AllreduceOptions`，`opts.asyncOp = async_op`，调用 `group.allreduce([tensor], opts)` 得到 `work`；`async_op=True` 直接返回 Work，否则在需要时 `work.wait()` 后返回 None（快照 `def all_reduce` L3812 起）。
- `async_op=False` 对 NCCL 也**不代表 CPU 等通信完成**，只代表当前 stream 已排上等待；CPU 侧计时必须 `torch.cuda.synchronize()`。
- `batch_isend_irecv(p2p_op_list)` 把多个 `P2POp` 用 NCCL group 语义（`ncclGroupStart/End`）一起发起，PP 与 CP 的 P2P 常用（快照 L3624）。
- coalescing：`_coalescing_manager(async_ops=...)` 把多次集合通信合并（快照 L3471）。
- 其他常用：`all_gather_into_tensor`、`reduce_scatter_tensor`、`all_to_all_single`、`barrier`（快照 L5181、L5934、L5987、L6280）。

### 5. watchdog、超时与错误处理

- 默认超时：快照中 `_set_pg_timeout` / `init_process_group` 文档写明 NCCL 默认 **10 分钟**，其他后端 30 分钟；`_get_default_timeout` 对 NCCL 返回 `default_pg_nccl_timeout`。
- 超时语义（文档原意）：超过 timeout 后 collective 被**异步 abort 且进程崩溃**，因为 CUDA 是异步执行的，失败的 NCCL 操作之后继续跑用户代码可能在损坏数据上运行；设置 `TORCH_NCCL_BLOCKING_WAIT` 时改为阻塞等待到超时再抛错。
- watchdog（概念）：后台线程周期性检查未完成的 WorkNCCL——完成 event 是否触发、是否超时、`ncclCommGetAsyncError` 是否报错；出错时 abort communicator（`ncclCommAbort`）并按 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 的策略处理（快照 `_abort_process_group` 注释：手动 abort 应在关闭该选项时使用，否则 watchdog 可能已自动处理错误/超时并 abort）。
- 还有一个“心跳监控”线程监视 watchdog 自身是否卡住（概念；变量名与默认值待确认）。

watchdog 对每个未完成 Work 的处理逻辑（概念图，函数名与具体分支待确认）：

```text
 watchdog 线程（周期性）
   for work in 未完成的 WorkNCCL:
     ├─ ev_done 已触发 ────────────────────────► 完成，移出列表
     ├─ 超过 timeout（NCCL 默认 10 min）
     │  或 ncclCommGetAsyncError 报错 ─────────► ncclCommAbort
     │                                            ├─► 按 TORCH_NCCL_ASYNC_ERROR_HANDLING 处理（终止进程等）
     │                                            └─► 超时 dump Flight Recorder（缓冲大小 > 0 时）
     └─ 其他 ──────────────────────────────────► 下一轮再查

 心跳监控线程 ──► 监视 watchdog 自己是否卡住
 TORCH_NCCL_BLOCKING_WAIT=1 ──► work.wait() 改为 CPU 阻塞等待，超时抛异常
```

- Flight Recorder：记录最近 N 次集合通信的元数据（序号、op、大小、状态、调用栈），超时时 dump，用来定位“哪个 rank 少调/多调了一次 collective”。快照里缓冲大小读 `TORCH_FR_BUFFER_SIZE`，回退到 `TORCH_NCCL_TRACE_BUFFER_SIZE`，默认 `"0"`（关闭）。
- NaN 检查：快照注释“ProcessGroupNCCL consumes TORCH_NCCL_NAN_CHECK itself”；后端无关版本为 `TORCH_DIST_NAN_CHECK`。
- 与 CUDA Graph 冲突：vLLM 的 skill 文档提到 ROCm 上在 HIP graph 中捕获 ProcessGroupNCCL collective 会触发 watchdog（不能查询正在捕获的 stream 上记录的 event），benchmark 时可用 `TORCH_NCCL_BLOCKING_WAIT=1` 关闭 watchdog（`third_party/vllm/.agents/skills/kernel-microbenchmark/SKILL.md`）。这也是推理框架偏爱 pynccl/custom AR 的原因之一。

### 6. TORCH_NCCL_* 常见配置

| 变量 | 作用（概念） | 本地可见证据 |
|---|---|---|
| `TORCH_NCCL_BLOCKING_WAIT` | `wait()` 阻塞 CPU 直到完成或超时，超时抛异常 | 快照文档；DDP `_dump_DDP_relevant_env_vars` 列表 |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | watchdog 发现错误/超时后的处理策略（abort comm、终止进程等） | 快照 `_abort_process_group` 注释；DDP env 列表 |
| `TORCH_NCCL_AVOID_RECORD_STREAMS` | 用持有引用代替 `recordStream`，降低显存滞留 | Llama 3 论文文本 |
| `TORCH_NCCL_TRACE_BUFFER_SIZE`（及 `TORCH_FR_BUFFER_SIZE`） | Flight Recorder 环形缓冲条目数，0 关闭 | 快照 L2831 附近 |
| `TORCH_NCCL_NAN_CHECK` | 集合通信前检查输入 NaN | 快照 L2939 注释 |
| `TORCH_NCCL_DUMP_ON_TIMEOUT`、`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`、`TORCH_NCCL_ENABLE_MONITORING`、`TORCH_NCCL_DESYNC_DEBUG`、`TORCH_NCCL_HIGH_PRIORITY`、`TORCH_NCCL_USE_COMM_NONBLOCKING` | 超时 dump、心跳监控、desync 诊断、高优先级 stream、非阻塞 comm 初始化 | 待确认：名称、默认值与语义；确认方式：克隆 pytorch/pytorch 后 `grep -rn "TORCH_NCCL_" torch/csrc/distributed/c10d/` |

DDP 打印的相关 NCCL 变量清单可作排障检查表：快照 `distributed.py` `_dump_DDP_relevant_env_vars`（`NCCL_SOCKET_IFNAME, NCCL_DEBUG, NCCL_IB_DISABLE, NCCL_P2P_DISABLE, NCCL_P2P_LEVEL, NCCL_SHM_DISABLE, NCCL_BUFFSIZE, NCCL_NTHREADS, NCCL_MAX_NCHANNELS, NCCL_MIN_NCHANNELS, NCCL_IB_HCA, NCCL_IB_TIMEOUT, NCCL_IB_RETRY_CNT, NCCL_IB_GID_INDEX, NCCL_IB_SL, NCCL_IB_TC, NCCL_NET_GDR_LEVEL, NCCL_ALGO, NCCL_PROTO, NCCL_TOPO_FILE, NCCL_COLLNET_ENABLE` 等）。这份列表里有些变量在 `third_party/nccl` 源码中已检索不到（如 `NCCL_RINGS`，`grep -rn NCCL_RINGS third_party/nccl/src` 无结果），使用前以实际 NCCL 版本为准。

### 7. DDP 分桶（训练侧的通信发起者）

- `distributed.py` `_DEFAULT_BUCKET_CAP_MB = 25`；第一个桶用 `dist._DEFAULT_FIRST_BUCKET_BYTES`（更小，尽早开始通信）；`bucket_cap_mb_list` 可指定每桶大小（`_BucketCapacityConfig`）。
- 反向中某个桶内梯度全部就绪即发起 AllReduce（异步），与后续反向计算重叠；`register_comm_hook` 可替换为压缩/自定义通信。
- ai-infra-book 实验 7-8 用这份 DDP + Gloo 记录了真实的通信 hook 时间线（`third_party/ai-infra-book/experiments/ch07/07-08/README.md`）：每 rank 每步 AllReduce bucket 载荷 6713344 bytes。

### 8. 与其他库共享 communicator

- DeepEP V2.5 默认复用 PyTorch 的 NCCL comm：`third_party/DeepEP/deep_ep/comm/handle.py` `get_nccl_comm_handle` 中 `backend._comm_ptr()`（`EP_REUSE_NCCL_COMM`，默认 1）——说明 ProcessGroupNCCL 暴露了底层 `ncclComm_t` 指针，第三方可在同一 communicator 上调用 NCCL device API。
- vLLM/SGLang pynccl 则另建 communicator，与 PyTorch 的互不影响。

## 性能要点与常见坑

- **hang 的三类根因**：(1) 各 rank 集合通信序列不一致（某 rank 少调一次、顺序不同、条件分支），(2) 某 rank 卡在计算/数据加载（掉队），(3) 网络/硬件故障。先用 Flight Recorder 对比各 rank 最后的序号，再看 `NCCL_DEBUG=INFO` 与网卡计数。
- 10 分钟默认超时对大模型初始化（加载权重、编译）可能太短，对线上推理又太长；按场景设置 `timeout`。
- 不要在 CPU 上用 `time.time()` 测 NCCL 通信，`async_op=False` 也不同步 CPU。
- 组太多会耗尽资源：每个 NCCL communicator 有自己的 buffer、proxy 线程与网络连接；优先 `device_id` eager + split。
- 通信 stream 与计算 stream 抢 SM：NCCL kernel 占 SM，计算重叠后两者都变慢；可以通过 `NCCL_MAX_NCHANNELS` 等限制（参见 `07-perf-troubleshooting/`）。
- CUDA Graph 捕获 ProcessGroupNCCL collective 需要特别小心（watchdog 查询 event、张量地址固定）。

## 面试自测

1. 问：`dist.all_reduce(t)` 返回后，CPU 能立即读 `t` 的结果吗？
   要点：不能保证。NCCL 后端只让当前 stream 等待通信完成 event；CPU 读之前要 `torch.cuda.synchronize()` 或 `.cpu()`（隐式同步当前 stream）。
2. 问：`async_op=True` 和 `False` 的区别？`work.wait()` 做什么？
   要点：True 返回 Work，调用方决定在哪里等待，可插入计算；`wait()` 让当前 stream 等待完成 event（非 CPU 阻塞，除非 BLOCKING_WAIT）；False 等价于立即 wait。
3. 问：为什么第一次 collective 特别慢？怎么提前暴露问题？
   要点：NCCL comm 懒创建，bootstrap、拓扑探测、建连都在第一次调用时发生；`init_process_group(device_id=...)` eager 初始化，子组用 `ncclCommSplit`。
4. 问：NCCL timeout 后进程为什么直接崩溃而不是抛异常继续？
   要点：CUDA 异步执行，失败的 NCCL 操作之后已排队的 kernel 可能读写损坏数据，继续运行不安全；watchdog 异步 abort comm 并终止；`TORCH_NCCL_BLOCKING_WAIT` 可改为阻塞等待并抛异常。
5. 问：训练 hang 了，你怎么定位是哪个 rank、哪个 collective？
   要点：开 Flight Recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE`）+ 超时 dump，比较各 rank 最后完成/发起的序号与 op；用 py-spy 看卡住 rank 的 Python 栈；`NCCL_DEBUG=INFO`/`NCCL_DEBUG_SUBSYS` 看建连与网络错误；检查是否有条件分支导致集合通信序列不一致。
6. 问：推理框架为什么还要自己 ctypes 调 NCCL（pynccl），而不全用 ProcessGroupNCCL？
   要点：需要在当前 stream 上发起以便 CUDA Graph 捕获、避免 watchdog/event 交互问题；可以加载不同版本 NCCL、使用 PyTorch 未暴露的新 API（窗口注册、suspend/resume）；控制更细。

## 待深入

- 待确认：ProcessGroupNCCL 中 comm 缓存的 key（device 字符串/组名）、`getNCCLComm` 等函数名、watchdog 与心跳线程实现、WorkNCCL 字段；确认方式：克隆 pytorch/pytorch 后查 `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`。
- 待确认：`TORCH_NCCL_*` 完整列表与默认值；确认方式：同上 `grep -rn "TORCH_NCCL_" torch/csrc/distributed/c10d/`。
- 待确认：torchcomms 与 ProcessGroupNCCL 的关系（快照中出现 `ProcessGroupNCCL.Options` 转 torchcomms hints 的函数）；确认方式：读快照 `distributed_c10d.py` L271 附近并克隆 pytorch 查 `torch/distributed/` 下 torchcomms 相关代码。
- 待确认：FSDP2 的 AllGather/ReduceScatter stream 安排；确认方式：克隆 pytorch/pytorch 后查 `torch/distributed/fsdp/_fully_shard/`。
