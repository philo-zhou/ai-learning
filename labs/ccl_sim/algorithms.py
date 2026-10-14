"""集合通信算法的逐步（step）模拟。

模型约定（教学简化，不是 NCCL 的真实执行方式）：

- 每个 rank 的 buffer 是一个 Python list；``inputs[r]`` 是 rank r 的输入。
- 通信按 *同步 step* 推进：同一个 step 内的所有发送都基于该 step 开始时的 buffer 快照，
  全部发送完成后再统一写入接收方。这对应 α-β 模型里"每轮先发完再进入下一轮"的假设。
- 一次 ``Transfer`` 是一条 src -> dst 的消息，``nelems`` 是元素数；字节数 = 元素数 × ``elem_bytes``。
- 归约默认是逐元素求和（``op=operator.add``），可传入其他满足结合律、交换律的二元函数。
- 输入不会被原地修改；函数返回 ``CommResult``，其中 ``outputs[r]`` 是 rank r 的结果。

各算法的 step 数（N 为 rank 数）：

| 函数 | step 数 | 每 rank 发送总量（元素，L 为 buffer 长度且可整除时） |
|---|---|---|
| ring_allreduce | 2(N-1) | 2(N-1)/N * L |
| recursive_doubling_allreduce | log2 N（N 非 2 的幂时 +2） | log2 N * L |
| halving_doubling_allreduce | 2 log2 N（N 必须是 2 的幂） | 2(N-1)/N * L |
| binary_tree_allreduce | 2 floor(log2 N) | 非叶子更多，见函数说明 |
| pairwise_all2all | N-1 | (N-1)/N * L |
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

Number = float  # int 也可以；这里只作类型提示
ReduceOp = Callable[[Number, Number], Number]

DEFAULT_ELEM_BYTES = 4  # 默认按 fp32 / int32 计 4 字节


# ---------------------------------------------------------------------------
# 通信记录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transfer:
    """一条点对点消息：src 发给 dst，共 nelems 个元素。"""

    src: int
    dst: int
    nelems: int


@dataclass
class StepRecord:
    """一个同步 step 内的全部消息。"""

    index: int
    phase: str
    nranks: int
    transfers: List[Transfer] = field(default_factory=list)

    def sent_elems_by_rank(self) -> List[int]:
        """本 step 每个 rank 发出的元素数。"""
        sent = [0] * self.nranks
        for t in self.transfers:
            sent[t.src] += t.nelems
        return sent

    def recv_elems_by_rank(self) -> List[int]:
        """本 step 每个 rank 收到的元素数。"""
        recv = [0] * self.nranks
        for t in self.transfers:
            recv[t.dst] += t.nelems
        return recv

    @property
    def total_elems(self) -> int:
        """本 step 全网传输的元素总数。"""
        return sum(t.nelems for t in self.transfers)

    @property
    def max_port_elems(self) -> int:
        """本 step 最忙端口的元素数：max_r max(发送_r, 接收_r)。

        α-β 模型下一个 step 的耗时 ≈ α + β × (这个值 × elem_bytes)，
        即"全双工、单端口、该 step 被最忙的那个 rank 拖住"。
        """
        sent = self.sent_elems_by_rank()
        recv = self.recv_elems_by_rank()
        return max((max(s, r) for s, r in zip(sent, recv)), default=0)


@dataclass
class CommResult:
    """一次集合通信模拟的结果与通信记录。"""

    algorithm: str
    nranks: int
    outputs: list
    steps: List[StepRecord]
    elem_bytes: int = DEFAULT_ELEM_BYTES

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def sent_elems_per_rank(self) -> List[int]:
        """整个算法中每个 rank 发出的元素总数。"""
        total = [0] * self.nranks
        for st in self.steps:
            for r, v in enumerate(st.sent_elems_by_rank()):
                total[r] += v
        return total

    def sent_bytes_per_rank(self) -> List[int]:
        return [v * self.elem_bytes for v in self.sent_elems_per_rank()]

    def recv_elems_per_rank(self) -> List[int]:
        total = [0] * self.nranks
        for st in self.steps:
            for r, v in enumerate(st.recv_elems_by_rank()):
                total[r] += v
        return total

    def step_summary(self) -> List[Tuple[int, str, int, int]]:
        """每步摘要：(step 序号, 阶段, 本步全网字节数, 本步最忙端口字节数)。"""
        return [
            (st.index, st.phase, st.total_elems * self.elem_bytes, st.max_port_elems * self.elem_bytes)
            for st in self.steps
        ]


class _Recorder:
    """内部工具：执行同步 step 并记录通信。"""

    def __init__(self, nranks: int):
        self.nranks = nranks
        self.steps: List[StepRecord] = []

    def run_step(
        self,
        phase: str,
        bufs: List[list],
        msgs: Sequence[Tuple[int, int, int, int]],
        mode: str,
        op: ReduceOp = operator.add,
    ) -> None:
        """执行一个 step。

        msgs 中每项为 (src, dst, start, end)：把 src 的 bufs[src][start:end]
        发给 dst，写到 dst 的同一位置。mode 为 "reduce"（op 累加）或 "copy"（覆盖）。
        先对所有发送取快照，再统一写入，保证同一 step 内的消息互不影响。
        """
        payloads = [(src, dst, start, bufs[src][start:end]) for src, dst, start, end in msgs]
        record = StepRecord(index=len(self.steps), phase=phase, nranks=self.nranks)
        for src, dst, start, data in payloads:
            target = bufs[dst]
            if mode == "reduce":
                for i, v in enumerate(data):
                    target[start + i] = op(target[start + i], v)
            elif mode == "copy":
                target[start : start + len(data)] = data
            else:  # pragma: no cover - 内部调用错误
                raise ValueError(f"未知 mode: {mode}")
            record.transfers.append(Transfer(src, dst, len(data)))
        self.steps.append(record)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _validate_equal_buffers(inputs: Sequence[Sequence[Number]]) -> Tuple[int, int]:
    """检查输入：至少 1 个 rank，且每个 rank 的 buffer 等长。返回 (N, L)。"""
    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError("inputs 必须是非空的 rank buffer 列表")
    length = len(inputs[0])
    for r, buf in enumerate(inputs):
        if len(buf) != length:
            raise ValueError(f"rank {r} 的 buffer 长度 {len(buf)} 与 rank 0 的 {length} 不一致")
    return len(inputs), length


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _log2_int(n: int) -> int:
    """n 必须是 2 的幂。"""
    return n.bit_length() - 1


def chunk_bounds(length: int, nchunks: int) -> List[Tuple[int, int]]:
    """把长度为 length 的 buffer 尽量均匀切成 nchunks 段，返回 [(start, end), ...]。

    前 length % nchunks 段各多 1 个元素；length < nchunks 时允许出现空段。
    """
    base, extra = divmod(length, nchunks)
    bounds = []
    start = 0
    for c in range(nchunks):
        size = base + (1 if c < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


# ---------------------------------------------------------------------------
# Ring：ReduceScatter + AllGather
# ---------------------------------------------------------------------------


def _ring_rs_phase(bufs: List[list], bounds: List[Tuple[int, int]], rec: _Recorder, op: ReduceOp) -> None:
    """Ring ReduceScatter 阶段：N-1 步，结束后 rank r 持有第 r 块的完整归约结果。

    第 s 步（s = 0..N-2）rank r 把第 (r - s - 1) mod N 块发给 r+1，接收方累加。
    第 c 块从 rank c+1 出发，每步多累加一个 rank，N-1 步后停在 rank c。
    """
    n = rec.nranks
    for s in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r - s - 1) % n
            msgs.append((r, (r + 1) % n, bounds[c][0], bounds[c][1]))
        rec.run_step("reduce-scatter", bufs, msgs, "reduce", op)


def _ring_ag_phase(bufs: List[list], bounds: List[Tuple[int, int]], rec: _Recorder) -> None:
    """Ring AllGather 阶段：N-1 步，前提是 rank r 已持有第 r 块。

    第 s 步 rank r 把第 (r - s) mod N 块发给 r+1，接收方覆盖写入。
    """
    n = rec.nranks
    for s in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r - s) % n
            msgs.append((r, (r + 1) % n, bounds[c][0], bounds[c][1]))
        rec.run_step("allgather", bufs, msgs, "copy")


def ring_reduce_scatter(
    inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES, op: ReduceOp = operator.add
) -> CommResult:
    """Ring ReduceScatter。outputs[r] 是全体输入逐元素归约后的第 r 块（按 chunk_bounds 切分）。"""
    n, length = _validate_equal_buffers(inputs)
    bufs = [list(b) for b in inputs]
    bounds = chunk_bounds(length, n)
    rec = _Recorder(n)
    _ring_rs_phase(bufs, bounds, rec, op)
    outputs = [bufs[r][bounds[r][0] : bounds[r][1]] for r in range(n)]
    return CommResult("ring_reduce_scatter", n, outputs, rec.steps, elem_bytes)


def ring_allgather(inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES) -> CommResult:
    """Ring AllGather。inputs[r] 是 rank r 的分片（长度可以不同），outputs[r] 是所有分片按 rank 拼接。"""
    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError("inputs 必须是非空的 rank buffer 列表")
    n = len(inputs)
    bounds = []
    start = 0
    for piece in inputs:
        bounds.append((start, start + len(piece)))
        start += len(piece)
    bufs = []
    for r in range(n):
        buf = [0] * start
        buf[bounds[r][0] : bounds[r][1]] = list(inputs[r])
        bufs.append(buf)
    rec = _Recorder(n)
    _ring_ag_phase(bufs, bounds, rec)
    return CommResult("ring_allgather", n, bufs, rec.steps, elem_bytes)


def ring_allreduce(
    inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES, op: ReduceOp = operator.add
) -> CommResult:
    """Ring AllReduce = Ring ReduceScatter（N-1 步）+ Ring AllGather（N-1 步）。

    每步每个 rank 只发 L/N 个元素，因此每 rank 发送总量 2(N-1)/N × L，与 N 基本无关，
    这就是 Ring 在大消息下带宽最优的原因；代价是 2(N-1) 个 step 的延迟。
    """
    n, length = _validate_equal_buffers(inputs)
    bufs = [list(b) for b in inputs]
    bounds = chunk_bounds(length, n)
    rec = _Recorder(n)
    _ring_rs_phase(bufs, bounds, rec, op)
    _ring_ag_phase(bufs, bounds, rec)
    return CommResult("ring_allreduce", n, bufs, rec.steps, elem_bytes)


# ---------------------------------------------------------------------------
# Recursive Doubling
# ---------------------------------------------------------------------------


def recursive_doubling_allreduce(
    inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES, op: ReduceOp = operator.add
) -> CommResult:
    """Recursive Doubling AllReduce（蝶形交换）。

    N 为 2 的幂时：第 k 步 rank r 与 r XOR 2^k 交换 *整个* buffer 并累加，共 log2 N 步。
    每步都发完整 L 个元素，所以延迟项最小（log2 N 个 α），带宽项为 log2 N × L，适合小消息。

    N 不是 2 的幂时用常见的"折叠"处理：设 P 为不超过 N 的最大 2 的幂、rem = N - P，
    先让前 2*rem 个 rank 中的奇数 rank 把数据并入左邻偶数 rank（1 步），
    剩下 P 个活跃 rank 做标准 Recursive Doubling，最后把结果发回被折叠的奇数 rank（1 步），
    共 log2 P + 2 步。
    """
    n, _ = _validate_equal_buffers(inputs)
    bufs = [list(b) for b in inputs]
    rec = _Recorder(n)
    length = len(bufs[0])
    p = 1 << (n.bit_length() - 1)  # 不超过 n 的最大 2 的幂
    rem = n - p

    if rem:
        msgs = [(r, r - 1, 0, length) for r in range(1, 2 * rem, 2)]
        rec.run_step("fold-in", bufs, msgs, "reduce", op)

    # 活跃 rank 的新编号 -> 真实 rank
    active = [r for r in range(0, 2 * rem, 2)] + list(range(2 * rem, n))
    assert len(active) == p
    mask = 1
    while mask < p:
        msgs = []
        for new_id, real in enumerate(active):
            partner = active[new_id ^ mask]
            msgs.append((real, partner, 0, length))
        rec.run_step(f"exchange(mask={mask})", bufs, msgs, "reduce", op)
        mask <<= 1

    if rem:
        msgs = [(r - 1, r, 0, length) for r in range(1, 2 * rem, 2)]
        rec.run_step("fold-out", bufs, msgs, "copy")

    return CommResult("recursive_doubling_allreduce", n, bufs, rec.steps, elem_bytes)


# ---------------------------------------------------------------------------
# Halving-Doubling（Rabenseifner）
# ---------------------------------------------------------------------------


def halving_doubling_allreduce(
    inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES, op: ReduceOp = operator.add
) -> CommResult:
    """Rabenseifner AllReduce = Recursive Halving ReduceScatter + Recursive Doubling AllGather。

    要求 N 为 2 的幂，否则抛 ValueError（真实实现会像 recursive_doubling 那样先折叠，这里刻意不做，
    以便观察约束）。

    ReduceScatter：mask 从 N/2 逐步减半到 1。rank r 与 r XOR mask 配对，双方当前持有同一个块区间，
    把区间对半分：mask 位为 0 的保留下半、发出上半，为 1 的相反；接收方把收到的一半累加进保留的一半。
    log2 N 步后 rank r 恰好持有第 r 块的完整归约结果。
    AllGather：mask 从 1 翻倍到 N/2，与伙伴交换各自持有的区间（覆盖写入），区间每步翻倍。
    共 2 log2 N 步，每 rank 发送 2(N-1)/N × L：延迟像 Recursive Doubling，带宽像 Ring。
    """
    n, length = _validate_equal_buffers(inputs)
    if not _is_power_of_two(n):
        raise ValueError(f"halving_doubling_allreduce 要求 rank 数为 2 的幂，当前 N={n}")
    bufs = [list(b) for b in inputs]
    bounds = chunk_bounds(length, n)
    rec = _Recorder(n)
    # windows[r] = (块起点, 块终点)，左闭右开，单位是"块"
    windows = [(0, n) for _ in range(n)]

    def elem_range(blo: int, bhi: int) -> Tuple[int, int]:
        if blo >= bhi:
            return (0, 0)
        return (bounds[blo][0], bounds[bhi - 1][1])

    mask = n // 2
    while mask >= 1:
        msgs = []
        new_windows = list(windows)
        for r in range(n):
            partner = r ^ mask
            lo, hi = windows[r]
            mid = (lo + hi) // 2
            if r & mask == 0:
                keep, send = (lo, mid), (mid, hi)
            else:
                keep, send = (mid, hi), (lo, mid)
            s, e = elem_range(*send)
            msgs.append((r, partner, s, e))
            new_windows[r] = keep
        rec.run_step(f"halving(mask={mask})", bufs, msgs, "reduce", op)
        windows = new_windows
        mask //= 2

    mask = 1
    while mask < n:
        msgs = []
        new_windows = list(windows)
        for r in range(n):
            partner = r ^ mask
            s, e = elem_range(*windows[r])
            msgs.append((r, partner, s, e))
            plo, phi = windows[partner]
            lo, hi = windows[r]
            new_windows[r] = (min(lo, plo), max(hi, phi))
        rec.run_step(f"doubling(mask={mask})", bufs, msgs, "copy")
        windows = new_windows
        mask <<= 1

    return CommResult("halving_doubling_allreduce", n, bufs, rec.steps, elem_bytes)


# ---------------------------------------------------------------------------
# Binary Tree：Reduce + Broadcast
# ---------------------------------------------------------------------------


def tree_depth(n: int) -> int:
    """堆式二叉树（parent = (i-1)//2）的深度 = floor(log2 N)。"""
    return n.bit_length() - 1


def binary_tree_allreduce(
    inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES, op: ReduceOp = operator.add
) -> CommResult:
    """二叉树 AllReduce = 自底向上 Reduce 到 rank 0 + 自顶向下 Broadcast。

    树形：堆式完全二叉树，rank i 的父节点是 (i-1)//2，第 i 个节点所在层 = floor(log2(i+1))。
    Reduce：从最深层开始，每步让一整层的节点把完整 buffer 发给父节点累加；
    Broadcast：从第 1 层开始，每步让父节点把结果发给这一层的子节点。共 2 floor(log2 N) 步。

    这里 *不做流水线切块*，每步都发完整 L 个元素，且有两个孩子的父节点一个 step 要收/发 2L，
    所以带宽项远差于 Ring。NCCL 的 Tree 会把消息切成 chunk 流水，并用两棵互补的树
    （double binary tree）让每个 rank 在一棵树里当内部节点、在另一棵里当叶子，见笔记 03。
    """
    n, length = _validate_equal_buffers(inputs)
    bufs = [list(b) for b in inputs]
    rec = _Recorder(n)
    depth = tree_depth(n)

    def nodes_at_level(level: int) -> range:
        return range((1 << level) - 1, min((1 << (level + 1)) - 1, n))

    for level in range(depth, 0, -1):
        msgs = [(i, (i - 1) // 2, 0, length) for i in nodes_at_level(level)]
        rec.run_step(f"reduce(level={level})", bufs, msgs, "reduce", op)
    for level in range(1, depth + 1):
        msgs = [((i - 1) // 2, i, 0, length) for i in nodes_at_level(level)]
        rec.run_step(f"broadcast(level={level})", bufs, msgs, "copy")

    return CommResult("binary_tree_allreduce", n, bufs, rec.steps, elem_bytes)


# ---------------------------------------------------------------------------
# All2All
# ---------------------------------------------------------------------------


def pairwise_all2allv(
    send_blocks: Sequence[Sequence[Sequence[Number]]], elem_bytes: int = DEFAULT_ELEM_BYTES
) -> CommResult:
    """Pairwise All2Allv：send_blocks[r][j] 是 rank r 要发给 rank j 的数据（长度可以不同）。

    outputs[r][j] = send_blocks[j][r]，即 rank r 收到的来自 rank j 的数据块。
    调度：自己发给自己的块本地拷贝、不计通信；第 s 步（s = 1..N-1）rank r 发给 (r+s) mod N、
    同时从 (r-s) mod N 接收，共 N-1 步。每步每个 rank 恰好发一条、收一条，端口不冲突。
    块长度不均（MoE 路由不均衡）时，每步耗时由该步最大的那条消息决定，见 StepRecord.max_port_elems。
    """
    if not isinstance(send_blocks, (list, tuple)) or len(send_blocks) == 0:
        raise ValueError("send_blocks 必须是非空列表")
    n = len(send_blocks)
    for r, row in enumerate(send_blocks):
        if len(row) != n:
            raise ValueError(f"rank {r} 必须提供 {n} 个目的块，实际 {len(row)} 个")
    outputs: List[List[Optional[list]]] = [[None] * n for _ in range(n)]
    for r in range(n):
        outputs[r][r] = list(send_blocks[r][r])
    steps: List[StepRecord] = []
    for s in range(1, n):
        record = StepRecord(index=len(steps), phase=f"pairwise(shift={s})", nranks=n)
        # 每步的发送互不依赖（发的是原始输入），直接写入即可
        for r in range(n):
            dst = (r + s) % n
            data = list(send_blocks[r][dst])
            outputs[dst][r] = data
            record.transfers.append(Transfer(r, dst, len(data)))
        steps.append(record)
    return CommResult("pairwise_all2allv", n, outputs, steps, elem_bytes)


def pairwise_all2all(inputs: Sequence[Sequence[Number]], elem_bytes: int = DEFAULT_ELEM_BYTES) -> CommResult:
    """等长 Pairwise All2All：inputs[r] 长度为 N*k，第 j 段（k 个元素）发给 rank j。

    outputs[r] 的第 j 段 = inputs[j] 的第 r 段。共 N-1 步，每 rank 发送 (N-1)/N × L。
    长度不能被 N 整除时抛 ValueError。
    """
    n, length = _validate_equal_buffers(inputs)
    if length % n != 0:
        raise ValueError(f"all2all 要求 buffer 长度 {length} 能被 rank 数 {n} 整除")
    k = length // n
    blocks = [[list(inputs[r][j * k : (j + 1) * k]) for j in range(n)] for r in range(n)]
    res = pairwise_all2allv(blocks, elem_bytes)
    flat = [[x for blk in res.outputs[r] for x in blk] for r in range(n)]
    return CommResult("pairwise_all2all", n, flat, res.steps, elem_bytes)
