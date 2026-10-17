# NVSHMEM 与 device-initiated 通信

> 层次：跨层（偏中层） · 来源：third_party/Triton-distributed@7908e4ea、third_party/DeepEP@93eb6eb2、third_party/nccl@3e46c089、ai-infra-book 第 6.4.4、7.3.2 节
>
> 边界说明：NVSHMEM 本身的源码未克隆。"NVSHMEM 概念"一节里凡不是来自本地仓库文字的，都标为"通用知识"，需要时再用 NVSHMEM 官方文档核实。

## 一句话

device-initiated 通信 = 由 GPU kernel 里的线程直接发起 put/get/signal，数据与控制都不再回到主机；NVSHMEM 提供 PGAS 风格的对称内存与这套 device API，Triton-distributed 直接暴露它；DeepEP 在较新版本里已改走 NCCL 的 device API（GIN / LSA）；NCCL 本身也已包含 device API（本地 2.32.3），不再只有 host-initiated 一种模型。

## 在调用链中的位置

```text
Triton-distributed kernel / DeepEP kernel / NCCL device API 示例
        │ 调用 device 端 put / signal / wait / 直接指针访问
        ▼
NVSHMEM device API   或   NCCL device API（LSA 域内指针 + GIN 跨节点）
        │
        ├── 域内：NVLink/NVSwitch P2P 直接 load/store（对称内存映射）
        └── 跨节点：GPU 直接构造 WQE 敲 NIC doorbell（IBGDA / GDAKI），或经 CPU proxy（IBRC / GIN PROXY）
        ▼
RDMA NIC（见 ../01-rdma/README.md）
```

## 核心概念

### 1. NVSHMEM 概念（通用知识）

- **PGAS（Partitioned Global Address Space）**：每个 PE（processing element，通常对应一张 GPU / 一个 rank）有自己的一块内存，所有 PE 都能按"PE 编号 + 对称地址"访问别人的那块。
- **对称堆（symmetric heap）**：每个 PE 在相同偏移处分配相同大小的区域，所以一个本地指针可以换算成任一 PE 上的对应地址。代码里体现为：Triton-distributed 用 `nvshmem_create_tensor` 分配对称张量（`third_party/Triton-distributed/python/triton_dist/utils.py`），堆大小由 `NVSHMEM_SYMMETRIC_SIZE` 控制（`docs/testing.md` 的环境变量表）。

下图示意对称堆的布局：各 PE 在相同偏移处有同样大小的区域，所以本地指针可以换算成任一 PE 上的对应地址（PE 数与张量个数为示意）。

```text
              PE 0                  PE 1                  PE 2
offset 0   ┌───────────┐         ┌───────────┐         ┌───────────┐
           │ tensor x  │         │ tensor x  │         │ tensor x  │
offset k   ├───────────┤         ├───────────┤         ├───────────┤
           │ tensor y  │         │ tensor y  │         │ tensor y  │
           └───────────┘         └───────────┘         └───────────┘
堆大小由 NVSHMEM_SYMMETRIC_SIZE 控制
本地指针 p 指向 tensor x  ──►  dl.symm_at(p, rank) = 该 rank 上 tensor x 的同偏移地址
```

- **put / get**：单边的远端写 / 读；有 `nbi`（non-blocking implicit）变体，以及 thread / warp / block 粒度变体（见 Triton-distributed `docs/primitives.md` 中 `putmem_nbi_block`、`putmem_warp` 等清单）。
- **signal**：带信号的 put（`putmem_signal*`，数据先到、再原子更新对端一个信号字）与 `signal_op`、`signal_wait_until`。接收方轮询信号即可知道数据可用——这正是通算融合里"依赖信号"的基础。
- **fence / quiet**：`fence` 保证同一 PE 向同一目标的写入顺序；`quiet` 等待此前所有远端操作完成。两者名字都出现在 Triton-distributed 的 `libshmem_device` 清单中；语义细节为通用知识，待确认；确认方式：查 NVSHMEM 官方 API 文档。
- **IBRC vs IBGDA（通用知识，待核实细节）**：
  - IBRC：GPU 把请求交给 CPU 上的 proxy 线程，由其用 IB Verbs（RC 连接）发给 NIC。控制路径要经 CPU，请求率受 CPU 线程限制。
  - IBGDA：GPU 线程自己构造 WQE 并敲 NIC doorbell，CPU 完全退出控制路径。代价是占 SM 构造请求、轮询完成。
  - 本仓库内的间接证据：Triton-distributed 的 `scripts/build_nvshmem_from_src.sh` 默认设置 `NVSHMEM_IBGDA_SUPPORT=1`，`docs/testing.md` 环境变量表列出 `NVSHMEM_IBGDA_SUPPORT`。书中第 6.4.4 节的"CPU 代理线程 / GPU 的 SM / 网卡上的处理器"三种发起位置，对应 IBRC / IBGDA / DPU 类方案，各自穿越 PCIe 3 / 2 / 0 次控制路径；第 7.3.2 节讨论 GPUDirect RDMA 与"由产生数据的加速器直接发起通信"。

下图对比 IBRC（经 CPU proxy）与 IBGDA（GPU 线程直接发起）的控制路径：前者控制信号要经 CPU，书中第 6.4.4 节给出的 PCIe 穿越次数是 CPU 代理 3 次、GPU SM 发起 2 次。

```text
IBRC: GPU 经 CPU proxy
┌─────────────────┐   ┌────────────────────┐   ┌───────┐
│ GPU kernel      │   │ CPU proxy thread   │   │ NIC   │
│ (put request)   │──►│ (IB Verbs, RC)     │──►│       │
└─────────────────┘   └────────────────────┘   └───────┘

IBGDA / GDAKI: GPU 线程构造 WQE 并直接敲 NIC doorbell
┌──────────────────────┐   ┌──────────────┐
│ GPU kernel threads   │   │ NIC          │
│ build WQE            │──►│ (doorbell)   │
└──────────────────────┘   └──────────────┘
```

### 2. 与 NCCL 主机侧发起模型对比

| 维度 | NCCL 传统（host-initiated） | NVSHMEM / NCCL device API（device-initiated） |
|---|---|---|
| 谁发起 | 主机调用 `ncclAllReduce` 等，提交到 CUDA stream | kernel 内线程 |
| 通信 kernel | NCCL 自己的 kernel（占 SM），跨节点还需 proxy 线程（见 [04-nccl-internals](../04-nccl-internals/README.md)） | 用户自己的 kernel 内嵌通信，或轻量发起 |
| 同步 | stream 顺序 / event | 信号字、屏障、内存序（acquire/release） |
| 粒度 | 整个集合通信 | tile / token / 字节级 |
| 内存 | 普通 device 内存（用户 buffer 注册可选） | 必须是对称内存 / 注册窗口 |
| 通用性 | 集合语义完整，拓扑算法选择自动 | 算法、拓扑、重试要自己写 |
| 典型用途 | 通用训练/推理通信 | 通算融合、MoE all2all、低延迟小消息 |
| 主要风险 | 与计算争 SM、launch 开销、边界不可融合 | 死锁/内存序 bug、占 SM、调试难 |

要点：device-initiated 不是"替换 NCCL"，而是在 NCCL 难以与计算融合或不适合不等长交换的场景下，把通信放进用户 kernel。书中第 6.4.3 节也指出 MoE 的 dispatch 不等长，固定大小 All-to-All 测试不能当作实际 MoE 通信。

### 3. Triton-distributed 如何使用 NVSHMEM

- 初始化：`init_nvshmem_by_torch_process_group`、`nvshmem_barrier_all_on_stream`（`python/triton_dist/utils.py`）。
- 张量：`nvshmem_create_tensor(s)` 创建对称张量；`nvshmem_create_tensors(shape, dtype, rank, local_world_size)` 返回同节点所有 rank 的对应张量列表，AG 算子拿它们做节点内直接 P2P 写（见 `AllGatherGEMMTensorParallelContext.symm_workspaces`）。
- device API：`triton_dist.language.extra.libshmem_device`，CUDA 后端落在 `python/triton_dist/language/extra/cuda/libnvshmem_device.py`；同一接口有 ROCm（`librocshmem_device.py`、`libmori_shmem_device.py`）与沐曦（`libmxshmem_device.py`）后端，说明上层 kernel 对 SHMEM 库是抽象的。
- 典型 kernel：`low_latency_all_to_all.py` 的 `all_to_all_kernel` 用 `putmem_nbi_block`、`putmem_signal_nbi_block`、`fence`、`signal_wait_until` 组成"推数据 + 置信号 + 等对端信号"。
- 域内直接指针：`dl.symm_at(ptr, rank)` 对 NVLink 域内 rank 可用 `tl.store` 直接写对端，GEMM+RS 的 `FUSE_SCATTER` 分支就是如此（`gemm_reduce_scatter.py`）。
- 版本说明：README 写明 pip 方式依赖 `nvidia-nvshmem-cu12`、`nvshmem4py-cu12`，且不需要修改 NVSHMEM 代码。

### 4. DeepEP 如何做 device-initiated 通信（以本地 commit 为准）

**重要：这与旧版认知不同。** 本地克隆的 DeepEP@93eb6eb2（V2.5）README 明确写："Fully remove V1, including its APIs, NVSHMEM backend ... NVSHMEM is no longer a dependency"。因此不能再说"DeepEP 基于 NVSHMEM"。事实（以本地代码为准）：

- 通信后端是 NCCL 的 device API：`third_party/DeepEP/csrc/kernels/comm/context.cpp` 通过 `ncclCommQueryProperties` 查询能力，用 `ncclDevCommCreate` 创建 device communicator，`reqs.ginType = NCCL_GIN_TYPE_GDAKI`，并设置 GIN 上下文数（`ginContextCount`，等于分配的 QP 数）、队列深度、signal 数、连接类型（`NCCL_GIN_CONNECTION_FULL` / `RAIL` / `CUSTOM_STRIDE`）；用 `ncclCommWindowRegister(..., NCCL_WIN_STRICT_ORDERING)` 把对称内存注册为窗口；`EP_DISABLE_GIN=1` 可关闭 GIN。内存用 `ncclMemAlloc` 分配（`csrc/kernels/comm/symmetric.hpp`）。
- kernel：`csrc/kernels/ep/dispatch.hpp`、`combine.hpp` 在宿主侧封装；设备侧实现在 `deep_ep/include/deep_ep/impls/ep/`（`dispatch.cuh`、`combine.cuh`、`hybrid_dispatch.cuh`、`hybrid_combine.cuh`）。`hybrid_dispatch.cuh` 里节点内用 TMA store 写（`ptx::tma_store_1d`），跨节点用 `gin.put<ncclTeamTagRail>(...)`，源码注释写 "Issue IBGDA requests"，说明 GDAKI 这条路径在代码里仍被称作 IBGDA。
- "hybrid / direct 两种模式"：README 称 hybrid（节点内 NVLink、跨节点走 rail）与 direct 都支持；`allow_hybrid_mode` 决定连接类型。
- SM：README 写明 EP dispatch / combine 需要 GPU SM，不支持零 SM 的 RDMA EP；SM 与 QP 数解析估算（`get_theoretical_num_sms`）。bucket 的 all-gather 由 copy engine 驱动（`num_sms=0`）。
- 重叠接口：dispatch / combine 带 `async_with_compute_stream=True`，返回 `EventOverlap`，用 `.current_stream_wait()` 或 `.wait()` 取结果；`previous_event` 表达输入就绪事件；`defer_epilogue=True` 可延后 epilogue（README "Deferred epilogues"）。这是 A 类（stream 级）重叠，kernel 内部是 C 类。
- 旧版 DeepEP（V1，基于 NVSHMEM + IBGDA）的细节本地不可见；README 实验分支列表提到 Hybrid-EP、Zero-copy、LL-SBO（用信号让 Down GEMM 与 Combine 发送重叠）等外部分支，细节需进入对应 PR / 分支；待确认；确认方式：读 README "Experimental branches" 中链接。

### 5. NCCL 源码中的 device API / symmetric memory / window / GIN（grep 确认）

NCCL@3e46c089（版本为 2.32.3，见 `third_party/nccl/makefiles/version.mk`）已包含一整套 device API 与对称内存：

**host 侧接口**（`third_party/nccl/src/nccl.h.in`）
- `ncclMemAlloc`：分配可注册的内存。
- `ncclCommWindowRegister(comm, buff, size, &win, winFlags)`：把 buffer 注册为窗口。窗口标志：`NCCL_WIN_DEFAULT`、`NCCL_WIN_COLL_SYMMETRIC`、`NCCL_WIN_STRICT_ORDERING`、`NCCL_WIN_GIN_ONLY`、`NCCL_WIN_CFT_COUNTED`，要求 4096 对齐（`NCCL_WIN_REQUIRED_ALIGNMENT`）。
- 单边 RMA 风格的 host API：`ncclPutSignal`、`ncclSignal`、`ncclWaitSignal`（入队到 CUDA stream，带 `sigIdx`、`ctx`）。实现目录 `src/rma/`：`rma_ce.cc`（copy engine，节点内）、`rma_proxy*.cc`（CPU proxy）。具体选择规则待确认；确认方式：读 `src/rma/rma.cc`。
- `ncclCommQueryProperties` / `ncclDevCommCreate` 及 `ncclDevCommRequirements_t`（`src/include/nccl_device/host.h`）：声明需要多少 LSA barrier、GIN signal 等资源，创建 device communicator。

**device 侧接口**（`third_party/nccl/src/include/nccl_device/`，该目录 `README.md` 说明分层：公共声明 → `impl/*__types.h` → `impl/*__funcs.h`）
- `core.h`：`ncclTeamWorld/Lsa/Rail`、`ncclGetLsaPointer(window, offset, peer)`（拿到 NVLink 域内对端指针直接 load/store）、`ncclGetLsaMultimemPointer`、GIN 类型 `NCCL_GIN_TYPE_PROXY / GDAKI / GPI / EFA_GDA`。
- `gin.h`：`ncclGin` 会话，`put`、`putValue`、`flush`、`flushAsync`、`waitSignal`、`readSignal`、`waitCounter`；完成动作有"强/弱信号"之分：强信号表示可见时此前所有 put 均已完成，弱信号只保证捆绑的那次 put。

下图对比强信号与弱信号的可见性保证：强信号可见时此前所有 put 都已完成，弱信号只保证捆绑的那次 put。

```text
sender      put(d1) ──► put(d2) + signal
                                  │
                                  ▼
弱信号      receiver 看到 signal：d2（捆绑的那次 put）已到，d1 不保证已可见
强信号      receiver 看到 signal：d1 与 d2 都已到
```

- `lsa_barrier.h`（`ncclLsaBarrierSession`：`arrive`、`wait`、`sync`，带 memory_order 参数）、`gin_barrier.h`、`barrier.h`、`ll_a2a.h`（低延迟 all2all）、`cft*.h`、`reduce_copy.h`。
- GIN 后端实现：`src/gin/`（`gin_host.cc`、`gin_host_proxy.cc`、`proxy_gpucontext/`）、`src/include/nccl_device/gin/` 下 `gdaki/`、`proxy/`、`gpi/`、`efa_gda/`。
- 对称内存的 kernel：`src/device/symmetric/`（`all_gather.cuh`、`all_reduce.cuh`、`reduce_scatter.cuh`、`all_gather_gin.cuh`、`reduce_scatter_gin.cuh`）；`NCCL_SYM_GIN_KERNELS_ENABLE` 见 `docs/userguide/source/env.rst`。

**NCCL 自带的文档与示例（强烈建议先读）**
- `third_party/nccl/docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md`：贡献者文档，讲 Symmetric Memory、LSA、GIN。其中关键结论：Symmetric Memory 把对端内存抽象为统一虚拟地址空间；域内用普通内存语义，跨 RDMA 网络需要 GIN 发起消息语义传输；GIN 有两个后端——GDA-KI（GPU 直接构造 WQE 敲 doorbell，当前实现面向 Mellanox 系列）与 PROXY（GPU 写 128 B 的 GFD 到与 CPU 共享的环形缓冲，CPU proxy 线程经 IBRC verbs 发送）；文档对两者都有批评：GDA-KI 占 CUDA core 构造 WQE，PROXY 烧 CPU。这与上文 IBGDA / IBRC 的对照一致。
- `third_party/nccl/docs/examples/06_device_api/README.md`：`01_allreduce_lsa`（kernel 内 AllReduce）、`02_alltoall_gin`（纯 GIN）、`03_alltoall_hybrid`（LSA + GIN）、CFT 系列。
- `third_party/nccl/docs/examples/07_kernel_fusion/README.md`：RMSNorm 与 reduce-scatter / all-gather 融合，分 LSA / multimem / GIN / hybrid 四个示例；README 自述按 NCCL 2.30+ 编写，且不追求极致性能。
- `third_party/nccl/docs/examples/05_symmetric_memory/`：对称内存的 allreduce、allgather 示例。

**host 侧控制 SM 占用**：`NCCL_CTA_POLICY`（DEFAULT / EFFICIENCY / ZERO，`docs/userguide/source/env.rst`）；`src/ce_coll.cc` 中有 copy engine 集合实现的参数。ZERO policy 的具体含义与适用集合，待确认；确认方式：读 `docs/userguide/source` 中 `cta_policy_flags` 一节。

## 性能要点与常见坑

- 对称内存必须所有 rank 同步分配 / 注册；注册是集合操作（DeepEP `context.cpp` 注释说明 `ncclCommWindowRegister` 内部做 bootstrap barrier）。
- 版本匹配：DeepEP 要求 NCCL ≥ 2.32.3；PyTorch 与 DeepEP 必须加载同一份 NCCL 动态库；2.31 之前编译期与运行期版本必须一致（`third_party/DeepEP/README.md`）。
- 强弱信号语义不同：用弱信号时，后一个 put 的数据可能先于前一个 put 可见，不能把一个信号当"前面全部到齐"。
- GIN 通信占 CUDA core（GDAKI）或 CPU（PROXY），请求率、QP 数、队列深度都是调参项（DeepEP `ginContextCount`、`ginQueueDepth`）。
- 书中第 6.4.4 节提醒：让出 SM 能缩短计算，但共用网络出口的字节数不会减少，瓶颈在出口时没有帮助。

## 面试自测

1. NVSHMEM 的对称堆是什么？为什么能用"PE + 本地指针"访问远端？
   - 要点：各 PE 在相同偏移分配同样大小的区域，本地指针可换算为任意 PE 上的对应地址；建立在地址映射与注册之上，域内走 NVLink load/store，域外走 RDMA。
2. IBRC 与 IBGDA 的差别？各自瓶颈？
   - 要点：IBRC 经 CPU proxy 用 Verbs 发送，瓶颈在 CPU 线程与 PCIe 往返；IBGDA 由 GPU 线程构造 WQE 并敲 doorbell，CPU 退出，瓶颈在占用 SM 与 WQE/CQ 处理指令数。NCCL GIN 中对应 PROXY 与 GDAKI 两个后端。
3. 设备发起通信相对 host-initiated 为什么能降低端到端延迟？还剩哪些成本？
   - 要点：省掉 GPU→CPU→NIC 的控制路径往返与 stream 边界；数据可按 tile 算完即发。成本是占 SM、对称内存管理、内存序与信号设计、调试困难。
4. 为什么 DeepEP 现在不依赖 NVSHMEM 了？它的通信从哪里来？
   - 要点：V2.5 重构后以 NCCL device API 为后端（GIN GDAKI + LSA + 窗口注册），可复用已有 NCCL communicator，README 明确 NVSHMEM 已不是依赖。（证据：`context.cpp`、README News。）
5. NCCL 的 LSA 与 GIN 分别是什么？
   - 要点：LSA 指 NVLink 可直连的 load/store 可达域，对端内存可用指针直接访问（`ncclGetLsaPointer`）；GIN 是跨节点 GPU 发起的网络通信（put / signal / flush）。hybrid 模式：域内走 LSA，跨节点走 GIN。
6. 强信号和弱信号的区别？什么时候必须用强信号？
   - 要点：强信号可见时保证此前所有 put 已完成；弱信号只保证捆绑的那次 put。当一个信号要代表"一批数据全部到齐"时用强信号，或自行在发信号前 flush。
7. 为什么 `wait` 之后读数据要带 acquire 语义、写数据后置信号要带 release 语义？
   - 要点：否则消费者可能先看到信号、后看到数据（重排）；Triton-distributed `dl.wait(..., "gpu", "acquire")`、GEMM+RS 里 `atomic_add(sem="release")` 是配套的。

## 待深入

- 待确认：NVSHMEM `fence` / `quiet` 在 IBGDA 下的精确语义；确认方式：NVSHMEM 官方文档（源码未克隆）。
- 待确认：NCCL `src/rma/rma.cc` 中 host 侧 `ncclPutSignal` 何时走 CE、何时走 proxy；确认方式：读该文件及 `src/rma/rma_ce.cc`、`rma_proxy.cc`。
- 待确认：NCCL GDAKI 全称；文档写作 GDA-KI，仅从上述贡献者文档得知它是 GPU 直接驱动 NIC 的后端；确认方式：搜 `src/include/nccl_device/gin/gdaki`。
- 待深入：读 `docs/examples/06_device_api/03_alltoall_hybrid` 的源码，把 LSA+GIN 的 barrier 与 put/flush 顺序画成时序图。
