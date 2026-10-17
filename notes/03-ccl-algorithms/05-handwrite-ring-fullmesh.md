# 手写题：Ring 与 Full-mesh 集合通信

> 层次：中层 · 来源：third_party/vllm@02653fa2（`csrc/custom_all_reduce.cuh`、`csrc/custom_collective_common.cuh`、`vllm/distributed/device_communicators/custom_all_reduce.py`）、third_party/nccl@3e46c089、本仓库 `labs/ccl_sim`（可运行的参考实现与测试）

## 一句话

面试让你"写一个 Ring / Full-mesh AllReduce"，考的是三件事：**下标推得对**（每一步谁发哪一块给谁）、**不会死锁且同步正确**（MPI 版看收发顺序，GPU 版看 flag 屏障与内存序）、**说得清代价和适用场景**（Ring 省带宽但有 2(N−1) 个 α；Full-mesh 只要 1~2 个 α，但要求任意两卡直连）。本篇给出可以直接默写的伪代码，并且每段都和 `labs/ccl_sim` 里通过测试的实现对应。

## 在调用链中的位置

```text
推理框架 TP AllReduce（每层 2 次）
   ├─ 小消息、机内全互联：Full-mesh one-shot / two-shot   ← vLLM / SGLang custom allreduce（CUDA IPC + flag）
   ├─ 大消息或跨机：NCCL Ring / Tree / NVLS              ← ncclAllReduce
   └─ MoE EP：Full-mesh All2All（DeepEP LL 就是"每对 rank 直接写"）
```

## 约定

两套原语，面试时先说清楚用哪一套：

```python
# MPI 风格（CPU 视角，描述算法）
send(buf, to) / recv(buf, frm)                  # 阻塞
isend(buf, to) / irecv(buf, frm) -> req; waitall(reqs)
sendrecv(sendbuf, to, recvbuf, frm)             # 同时收发，不会因收发顺序死锁

# GPU kernel 风格（机内 NVLink，描述实现）
peer_ptr[p]          # 通过 CUDA IPC 映射的 rank p 的 buffer，可直接 load/store
signals[p]           # rank p 的信号区（也是 IPC 映射的），用来做跨卡屏障
```

记号：`N` 为 rank 数，`r` 为本 rank，`L` 为每个 rank 的元素数，`M` 为对应字节数。`split(buf, N)` 把 buffer 切成 N 块，不能整除时前 `L % N` 块各多 1 个元素（与 `labs/ccl_sim/algorithms.py` 的 `chunk_bounds` 一致）。

---

## 题 1：Ring AllReduce（最高频）

### 思路

AllReduce = ReduceScatter + AllGather，两段都沿环走 N−1 步：

- ReduceScatter 结束后，rank r 持有第 r 块的完整和；
- AllGather 再把这 N 块转一圈发给所有人。

每一步每个 rank 只向右邻居发 1/N 的数据，所以总字节数与 N 基本无关。

### 伪代码

```python
def ring_allreduce(buf, r, N):
    if N == 1:
        return buf
    chunk = split(buf, N)             # chunk[c] 是 buf 的视图
    right, left = (r + 1) % N, (r - 1) % N
    tmp = empty(max_chunk_len)

    # ReduceScatter：第 s 步发第 (r-s-1) 块，收第 (r-s-2) 块并累加
    for s in range(N - 1):
        send_c = (r - s - 1) % N
        recv_c = (r - s - 2) % N
        sendrecv(chunk[send_c], to=right, recvbuf=tmp[:len(chunk[recv_c])], frm=left)
        chunk[recv_c] += tmp[:len(chunk[recv_c])]
    # 此时 chunk[r] 是全局和

    # AllGather：第 s 步发第 (r-s) 块，收第 (r-s-1) 块并覆盖
    for s in range(N - 1):
        send_c = (r - s) % N
        recv_c = (r - s - 1) % N
        sendrecv(chunk[send_c], to=right, recvbuf=chunk[recv_c], frm=left)
    return buf
```

### 下标为什么对（面试时要能讲出来）

- **接收块号 = 左邻居的发送块号**：左邻居 `l = r−1` 在第 s 步发的是 `(l−s−1) = (r−s−2)`，所以 r 收的就是这一块。
- **ReduceScatter 的不变式**：第 c 块从 rank `c+1` 出发，每走一跳多累加一个 rank 的贡献，走 N−1 跳后正好停在 rank c，这时已累加全部 N 份。最后一步 `s = N−2` 时，r 收到的块号是 `(r−N) mod N = r`，验证了"结束时 rank r 持有第 r 块"。
- **AllGather**：第 0 步每个 rank 发出自己持有的完整块 `chunk[r]`，之后每步把上一步收到的块继续往右传。

把上面的公式代入 N = 4 逐步展开如下。每一行都是"所有 rank 同时发给右邻 (r+1)%4"；cK(abc) 表示"第 K 块，已沿路径累加 rank a、b、c 的贡献"，cK(*) 表示 4 份全齐：

```text
ReduceScatter: step s, rank r sends c[(r-s-1)%4], receives c[(r-s-2)%4] and adds
              rank 0        rank 1        rank 2        rank 3
s=0  send     c3            c0            c1            c2
     got      c2(30)        c3(01)        c0(12)        c1(23)
s=1  send     c2(30)        c3(01)        c0(12)        c1(23)
     got      c1(230)       c2(301)       c3(012)       c0(123)
s=2  send     c1(230)       c2(301)       c3(012)       c0(123)
     got      c0(*)         c1(*)         c2(*)         c3(*)       <- rank r holds c_r

AllGather: step s, rank r sends c[(r-s)%4], receives c[(r-s-1)%4] and overwrites
              rank 0        rank 1        rank 2        rank 3
s=0  send     c0(*)         c1(*)         c2(*)         c3(*)
     got      c3(*)         c0(*)         c1(*)         c2(*)
s=1  send     c3(*)         c0(*)         c1(*)         c2(*)
     got      c2(*)         c3(*)         c0(*)         c1(*)
s=2  send     c2(*)         c3(*)         c0(*)         c1(*)
     got      c1(*)         c2(*)         c3(*)         c0(*)       <- all 4 chunks everywhere
```

读图要点：每一步的 `send` 一行等于上一步的 `got` 一行（刚收到并累加好的块马上往右传）；c0 从 rank 1 出发，经 rank 2、3，在 s=2 停在 rank 0，正是上面说的"第 c 块从 rank c+1 出发、走 N−1 跳停在 rank c"。

对照实现：`labs/ccl_sim/algorithms.py` 的 `_ring_rs_phase`、`_ring_ag_phase`，测试 `test_all_algorithms_match_naive_sum`。

### 代价

`2(N−1)` 步，每步每 rank 发 `M/N`：

$$T_{ring} = 2(N-1)\alpha + \frac{2(N-1)}{N} M\beta + \frac{N-1}{N} M\gamma$$

### 必问追问

| 追问 | 回答要点 |
|---|---|
| 为什么用 `sendrecv`？换成先 `send` 再 `recv` 会怎样？ | 所有 rank 同时阻塞在 `send` 等对方 `recv`，形成环形等待，消息超过 eager 阈值时就会死锁。解法：`sendrecv`、`isend`+`irecv`+`waitall`，或者奇偶 rank 交错收发顺序。 |
| L 不能被 N 整除？ | 按 `chunk_bounds` 切出不等长的块，每步按收到那块的实际长度累加；或补齐到 N 的倍数。 |
| 能不能原地（in-place）？ | 可以，上面就是原地写在 `buf` 上。只需要一块 `tmp` 存接收数据，因为累加不能直接覆盖。 |
| 怎么把带宽跑满？ | (1) 把每块再切成多个 slice 做流水，发送第 k 个 slice 时累加第 k−1 个；(2) 开多个 channel（多个环、方向或起点错开）并行，用满多条 NVLink 或多张网卡。NCCL 就是"channel × 流水 slice"的组合（见 [03-nccl-algorithms](03-nccl-algorithms.md)）。 |
| 跨机怎么排环？ | 让环在每台机器内连续，每台只跨一次网卡；或者用分层：机内 ReduceScatter → 机间 AllReduce（每卡只负责 1/ppn）→ 机内 AllGather（`labs/ccl_sim/cost_model.py` 的 `hierarchical_allreduce`）。 |
| Ring 的弱点？ | α 项随 N 线性增长，小消息和大规模时延迟差，所以有 Tree、NVLS 和 Full-mesh。 |

---

## 题 2：单独写 Ring ReduceScatter / AllGather

就是题 1 的前半段和后半段。注意 AllGather 的输入是每个 rank 的一块，输出是拼起来的整体：

```python
def ring_allgather(my_piece, r, N):
    out = empty(N * len(my_piece)); blk = split(out, N)
    blk[r][:] = my_piece
    right, left = (r + 1) % N, (r - 1) % N
    for s in range(N - 1):
        sendrecv(blk[(r - s) % N], to=right, recvbuf=blk[(r - s - 1) % N], frm=left)
    return out
```

代价：`(N−1)α + (N−1)/N·Mβ`（M 为输出总大小）。

---

## 题 3：Full-mesh ReduceScatter / AllGather / All2All（MPI 风格）

### 思路

Full-mesh 指任意两个 rank 之间都有直连路径，比如机内 NVSwitch：每张卡都能同时以接近满带宽和其余 N−1 张卡通信。所以不再需要沿环一跳一跳地转发，**一步就和所有对端直接交换**。

### 伪代码

```python
def fullmesh_reduce_scatter(buf, r, N):
    chunk = split(buf, N)
    tmp = [empty_like(chunk[r]) for _ in range(N)]
    reqs = []
    for i in range(1, N):                       # 1) 先把所有接收挂上
        p = (r - i) % N
        reqs.append(irecv(tmp[p], frm=p))
    for i in range(1, N):                       # 2) 再发；起点错开，避免所有人先打 rank 0
        p = (r + i) % N
        reqs.append(isend(chunk[p], to=p))
    waitall(reqs)
    out = zeros_like(chunk[r])
    for p in range(N):                          # 3) 固定顺序累加 → 各 rank 结果按位一致
        out += chunk[r] if p == r else tmp[p]
    return out

def fullmesh_allgather(my_piece, r, N):
    out = empty(N * len(my_piece)); blk = split(out, N); blk[r][:] = my_piece
    reqs = [irecv(blk[(r - i) % N], frm=(r - i) % N) for i in range(1, N)]
    reqs += [isend(my_piece, to=(r + i) % N) for i in range(1, N)]
    waitall(reqs)
    return out

def fullmesh_all2all(sendbuf, r, N):           # sendbuf 第 j 段发给 rank j
    s = split(sendbuf, N); out = empty_like(sendbuf); o = split(out, N)
    o[r][:] = s[r]
    reqs = [irecv(o[(r - i) % N], frm=(r - i) % N) for i in range(1, N)]
    reqs += [isend(s[(r + i) % N], to=(r + i) % N) for i in range(1, N)]
    waitall(reqs)
    return out
```

要点：

- **先挂 irecv 再 isend**：接收缓冲提前就绪，发送方不需要等待或使用中转缓冲（rendezvous 协议下能直接写进去）。
- **发送起点错开** `(r+i) % N`：第 i 轮每个 rank 都发给不同的对端，任一时刻每个接收端只承接一条流，不会出现热点（这正是 pairwise All2All 的调度）。
- **固定顺序累加**：浮点加法不满足结合律，按 rank 0..N−1 的固定顺序累加，才能保证所有 rank 得到按位相同的结果。vLLM 的 one-shot kernel 注释也特意强调了这一点（`csrc/custom_all_reduce.cuh`）。

代价：都是 1 步，`α + (N−1)/N·Mβ`。字节数与 Ring 相同，α 从 N−1 个降到 1 个。对照实现：`fullmesh_reduce_scatter`、`fullmesh_allgather`、`fullmesh_all2all`（`labs/ccl_sim/algorithms.py`）。

---

## 题 4：Full-mesh AllReduce 的 GPU kernel（one-shot / two-shot）

这是推理通信库岗位最可能深挖的版本：vLLM、SGLang 的 custom allreduce 就是这样实现的。

### 前提

- 每个 rank 的输入放在对端可访问的 buffer 里（CUDA IPC 注册；vLLM 在输入未注册时会先把它拷进自己的 IPC buffer，见 `custom_all_reduce.py`），并且每个 rank 都拿到了所有对端 buffer 的指针 `peer_in[p]`。
- 每个 rank 有一块信号区 `Signal`，里面有 `start[block][rank]`、`end[block][rank]` 和本地递增计数 `_flag[block]`（`csrc/custom_collective_common.cuh` 的 `struct Signal`）。

### 跨卡屏障（先写这个，两种 kernel 都要用）

```cuda
// 每个 block 独立做屏障；flag 只增不减，省掉每轮清零（与 Hybrid-EP 的标志设计相同）
__device__ void barrier(Signal* self, Signal* peers[N], int rank, Phase ph, bool need_visibility) {
    if (ph != START) __syncthreads();            // 本 block 所有线程的读写都做完了，才能对外宣布"到了"
    uint32_t flag = self->_flag[blockIdx.x] + 1;
    if (threadIdx.x < N) {
        int p = threadIdx.x;
        // 1) 告诉 rank p："rank 这个 block 到了第 flag 轮"
        // 2) 等 rank p 也告诉我它到了
        if (need_visibility) {                   // 屏障之后要读对端在屏障之前写的数据
            store_release_sys(&peers[p]->slot(ph)[blockIdx.x][rank], flag);
            while (load_acquire_sys(&self->slot(ph)[blockIdx.x][p]) != flag) {}
        } else {
            store_volatile(&peers[p]->slot(ph)[blockIdx.x][rank], flag);
            while (load_volatile(&self->slot(ph)[blockIdx.x][p]) != flag) {}
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) self->_flag[blockIdx.x] = flag;
}
```

vLLM 中对应 `barrier_at_start` / `barrier_at_end`（`csrc/custom_collective_common.cuh`）：

- 开始屏障（`barrier_at_start`）和 one-shot 的最后一道屏障（`cross_device_reduce_1stage` 里的 `barrier_at_end<ngpus, true>`，即 `final_sync`）只需要 volatile 读写，因为它们不需要让对端看到本卡之前写的数据。
- two-shot 的中间屏障需要 release/acquire（`st_flag_release` / `ld_flag_acquire`），因为下一阶段要读对端在屏障之前写的 tmp。vLLM 的 `cross_device_reduce_2stage` 只有 `barrier_at_start` 和这道中间的 `barrier_at_end<ngpus>`（`final_sync = false`），stage 2 之后没有再加屏障。

### One-shot：每个 rank 读所有对端的完整输入，自己归约

```cuda
__global__ void oneshot_allreduce(T* peer_in[N], T* out, int n, int rank, Signal sigs) {
    barrier(START, /*need_visibility=*/false);   // 所有 rank 的输入都已就绪（输入在 kernel 之前写好）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float acc = 0;
        for (int p = 0; p < N; ++p)              // 固定顺序 → 按位一致
            acc += (float)peer_in[p][i];         // 经 NVLink 直接 load 对端显存
        out[i] = (T)acc;
    }
    barrier(END, /*need_visibility=*/false);     // 没有人还在读我的输入，之后才允许覆盖它
}
```

- 1 步、每 rank 收 (N−1)·M：`T = α + (N−1)Mβ`。字节最多，但只有 1 次同步。
- 结束屏障防的是 **WAR（读后写）冲突**：如果 rank A 已经开始下一轮、覆盖了自己的输入，rank B 却还在读 A 的上一轮数据，结果就错了。

对应 vLLM：`cross_device_reduce_1stage`（`csrc/custom_all_reduce.cuh`）。

### Two-shot：ReduceScatter + AllGather，各 1 步

```cuda
__global__ void twoshot_allreduce(T* peer_in[N], T* peer_tmp[N], T* out, int n, int rank, Signal sigs) {
    int part = n / N, beg = rank * part, end = (rank == N - 1) ? n : beg + part;
    barrier(START, false);
    // 第 1 步 ReduceScatter：我只负责 [beg, end)，读所有人的这一段，归约后写进我自己的 tmp
    for (int i = beg + tid; i < end; i += stride) {
        float acc = 0;
        for (int p = 0; p < N; ++p) acc += (float)peer_in[p][i];
        peer_tmp[rank][i - beg] = (T)acc;
    }
    barrier(MID, /*need_visibility=*/true);      // release/acquire：保证对端能看到我写的 tmp
    // 第 2 步 AllGather：从每个 rank 的 tmp 拉回它负责的那一段
    for (int i = tid; i < largest_part; i += stride)
        for (int p = 0; p < N; ++p)
            if (i < part || p == N - 1)
                out[p * part + i] = peer_tmp[p][i];
}
```

- 2 步、每 rank 收发 2(N−1)/N·M：`T = 2α + 2(N−1)/N·Mβ`。**带宽项与 Ring 相同，α 只有 2 个**。
- vLLM 源码里有一个细节值得在面试中提：两个阶段的线程下标 `tid` 必须对应，因为跨设备的可见性只在"屏障两侧相同 tid 的线程"之间得到保证（`cross_device_reduce_2stage` 的注释）。
- vLLM 的选择逻辑（`csrc/custom_all_reduce.cuh`）：2 卡永远用 one-shot；全互联时，≤4 卡且 < 512 KiB 或 ≤8 卡且 < 256 KiB 用 one-shot，否则用 two-shot；不是全互联（如部分 PCIe）且超过 2 卡时，Python 侧根本不启用 custom allreduce（`custom_all_reduce.py` 中 `fully_connected` 判断）。

两种 kernel 的访存区域对比如下（N = 4，从 rank 1 的视角；输入按 rank 切成 4 段 seg0..seg3，`X` 表示要读的区域）：

```text
            one-shot                  two-shot step 1 (RS)        two-shot step 2 (AG)
            seg0 seg1 seg2 seg3       seg0 seg1 seg2 seg3
in_0 (peer)  X    X    X    X          .    X    .    .           tmp_0 ──► out seg0
in_1 (self)  X    X    X    X          .    X    .    .           tmp_1 ──► out seg1
in_2 (peer)  X    X    X    X          .    X    .    .           tmp_2 ──► out seg2
in_3 (peer)  X    X    X    X          .    X    .    .           tmp_3 ──► out seg3
             │                              │
             ▼                              ▼            barrier(MID, release/acquire)
         out = Σ_p in_p             tmp_1 = Σ_p in_p[seg1]
     remote reads: (N-1)·M          remote reads: (N-1)/N·M   +   remote reads: (N-1)/N·M
     barriers: START, END           barriers: START, MID
```

one-shot 每个 rank 都把 N 份完整输入读一遍，字节多但只有一个阶段；two-shot 每个 rank 只归约自己负责的一段，再从其他 rank 的 tmp 拉回其余段，两个阶段之间必须隔一道带内存序的屏障。

对照实现：`fullmesh_allreduce_oneshot`、`fullmesh_allreduce_twoshot`（`labs/ccl_sim/algorithms.py`），测试类 `TestFullMesh`。

---

## 代价对比（8 卡 NVSwitch 量级，α = 2 µs，单卡 400 GB/s）

用 `python3 -m labs.ccl_sim --nranks 8 --alpha-us 2 --bw-gbps 400` 得到（单位 µs，模型值，不是实测）：

| 消息 | Ring | One-shot | Two-shot | 最优 |
|---|---|---|---|---|
| 16 KiB | 28.1 | 2.3 | 4.1 | one-shot |
| 64 KiB | 28.3 | 3.1 | 4.3 | one-shot |
| 256 KiB | 29.1 | 6.6 | 5.1 | two-shot |
| 4 MiB | 46.4 | 75.4 | 22.4 | two-shot |
| 64 MiB | 321.6 | 1176.4 | 297.6 | two-shot ≈ Ring |

| 算法 | 步数 | 每 rank 字节 | 拓扑要求 | 典型用途 |
|---|---|---|---|---|
| Ring | 2(N−1) | 2(N−1)/N·M | 只要求环上相邻可达 | 大消息、PCIe、跨机 |
| Full-mesh one-shot | 1 | (N−1)·M | 全互联 | decode 小消息 TP AllReduce |
| Full-mesh two-shot | 2 | 2(N−1)/N·M | 全互联 | 中等消息 |
| NVLS（NVSwitch 多播归约） | — | 归约在交换机里做 | NVSwitch + SHARP | NCCL 的机内大消息（见 [03-nccl-algorithms](03-nccl-algorithms.md)） |

模型告诉你的趋势是对的（小消息看 α，大消息看 β），但有两点不能照搬：

1. **模型假设一张卡可以同时以满端口带宽向 N−1 个对端发送**。真实系统中，N−1 条并发流会争用 NVLink 和 HBM，还要占用 SM 做 load/归约。所以大消息时 two-shot 并不一定比 NCCL 快，vLLM 也只在 `max_size`（默认 8 MiB）以下使用 custom allreduce。
2. **Full-mesh 依赖全互联**。在 PCIe 或部分 NVLink 拓扑上，N−1 条流会挤在同一条链路或同一个 PCIe switch 上，退化严重。这时就该用 Ring。

---

## 面试自测

1. **问**：手写 Ring AllReduce，并说明第 s 步 rank r 发哪块、收哪块。
   **答**：ReduceScatter 阶段发 `(r−s−1) mod N`、收 `(r−s−2) mod N` 并累加；AllGather 阶段发 `(r−s) mod N`、收 `(r−s−1) mod N` 并覆盖。用 `sendrecv` 避免死锁。
2. **问**：8 卡 H100 NVSwitch 上，decode 时 TP AllReduce 只有 64 KiB，你用什么算法？为什么？
   **答**：Full-mesh one-shot。消息落在 α 主导区间，Ring 需要 14 个 α，one-shot 只要 1 个；虽然字节数是 Ring 的约 4 倍，但 64 KiB 的传输时间远小于 α。配合 CUDA Graph 消除 launch 开销。
3. **问**：one-shot kernel 为什么需要结束屏障？去掉会怎样？
   **答**：防止对端还在读本卡输入时，本卡已经开始下一轮并覆盖输入（读后写冲突），去掉会偶发数值错误。two-shot 的中间屏障则保证 ReduceScatter 写入的 tmp 对其他卡可见，所以必须用 release/acquire；开始屏障和 one-shot 的结束屏障（`final_sync`）只用 volatile 即可。vLLM 的 two-shot 只有开始屏障和中间屏障两道，stage 2 之后没有结束屏障。
4. **问**：flag 为什么只增不减？
   **答**：不需要每轮清零。如果清零，必须保证所有对端都已看到上一轮的值再清，这又需要一次同步。递增的 flag 让"等于本轮值"天然区分新旧轮次，也方便被 CUDA Graph 重放。
5. **问**：Full-mesh All2All 和 Ring 能做 All2All 吗？哪个好？
   **答**：Ring 做 All2All 要把数据转发很多跳，总字节变成 O(N·M)，很差；Full-mesh（或 pairwise 错开对端）每块只走一跳，字节为 (N−1)/N·M。机内 NVSwitch 用 full-mesh 一步完成，跨机常用 pairwise 或分层（DeepEP normal / Hybrid-EP）。
6. **问**：为什么所有 rank 必须用同一个累加顺序？
   **答**：浮点加法不满足结合律，不同顺序会得到不同的低位结果。TP 的各个 rank 拿到不一致的激活，后续计算会逐层发散，也会破坏 batch invariance。

## 待深入

- 待确认：SGLang 的 custom allreduce（`third_party/sglang/python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh`）与 vLLM 版本的差异（是否有 push 模式、是否用 multimem）；确认方式：对比两个文件中 kernel 的命名与屏障实现。
- 待确认：NCCL 的 NVLS 与 one-shot/two-shot 在同一台 H100 上的实测分界点；确认方式：在 GPU 机器上运行 nccl-tests 的 `all_reduce_perf`（`NCCL_ALGO=NVLS` 与 `Ring`）和 vLLM 的 custom allreduce benchmark，记录 not-run → passed。
