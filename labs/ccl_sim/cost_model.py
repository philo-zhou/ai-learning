"""α-β(-γ) 代价模型。

记号（与笔记 notes/03-ccl-algorithms/02-alpha-beta-model.md 一致）：

- n：rank 数；M：消息字节数（AllReduce/ReduceScatter 为每 rank 的输入大小，
  AllGather 为每 rank 的输出总大小，All2All 为每 rank 的发送总大小）。
- alpha：每个 step（一轮点对点通信）的固定启动开销，单位秒。
- beta：每字节传输时间，单位 秒/字节，= 1 / 单向链路带宽。
- gamma：每字节归约计算时间，单位 秒/字节，默认 0（GPU 上通常远小于 beta）。

所有函数返回秒。假设：全双工、每个 rank 单端口、同一 step 内的消息并行、step 之间严格同步、
无拥塞。它是"选算法的直觉工具"，不是 NCCL 的真实耗时；NCCL 自己的模型在
third_party/nccl/src/tuning/（见笔记 03）。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional


def gbps_to_beta(gbytes_per_s: float) -> float:
    """把单向带宽（GB/s，按 1e9 字节）换算成 beta（秒/字节）。"""
    if gbytes_per_s <= 0:
        raise ValueError("带宽必须为正")
    return 1.0 / (gbytes_per_s * 1e9)


def _check(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须是正整数，当前 {n!r}")
    if nbytes < 0 or alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("nbytes、alpha、beta、gamma 不能为负")


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _ceil_log2(n: int) -> int:
    return 0 if n <= 1 else (n - 1).bit_length()


# ---------------------------------------------------------------------------
# AllReduce
# ---------------------------------------------------------------------------


def ring_allreduce(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """Ring：2(n-1)α + 2(n-1)/n·Mβ + (n-1)/n·Mγ。"""
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    return 2 * (n - 1) * alpha + 2 * (n - 1) / n * nbytes * beta + (n - 1) / n * nbytes * gamma


def recursive_doubling_allreduce(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """Recursive Doubling：n 为 2 的幂时 log2 n·(α + Mβ + Mγ)。

    n 不是 2 的幂时按"折叠"实现：p = 不超过 n 的最大 2 的幂，多出 fold-in / fold-out 两步，
    共 (log2 p + 2) 步、每步发完整 M，归约 (log2 p + 1) 次。
    """
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    p = 1 << (n.bit_length() - 1)
    k = p.bit_length() - 1
    steps = k + (2 if n != p else 0)
    reductions = k + (1 if n != p else 0)
    return steps * (alpha + nbytes * beta) + reductions * nbytes * gamma


def halving_doubling_allreduce(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """Rabenseifner（Recursive Halving RS + Recursive Doubling AG）：

    2·log2 n·α + 2(n-1)/n·Mβ + (n-1)/n·Mγ。要求 n 为 2 的幂，否则 ValueError（与模拟器一致）。
    """
    _check(n, nbytes, alpha, beta, gamma)
    if not _is_power_of_two(n):
        raise ValueError(f"halving_doubling_allreduce 要求 n 为 2 的幂，当前 n={n}")
    if n == 1:
        return 0.0
    k = n.bit_length() - 1
    return 2 * k * alpha + 2 * (n - 1) / n * nbytes * beta + (n - 1) / n * nbytes * gamma


def binary_tree_allreduce(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """不切块的堆式二叉树 Reduce + Broadcast，与 algorithms.binary_tree_allreduce 的调度逐步对应。

    深度 d = floor(log2 n)。第 l 层（l = 1..d）若存在"有两个孩子的父节点"（n-1 >= 2^l），
    该层 step 的最忙端口要收（或发）2M，否则是 M。Reduce 与 Broadcast 各走一遍：

        T = Σ_l 2·(α + fan_l·Mβ) + Σ_l fan_l·Mγ，fan_l ∈ {1, 2}

    粗略上 ≈ 2·log2 n·(α + 2Mβ)：延迟项是对数级，但带宽项随 log n 增长，大消息很差。
    """
    _check(n, nbytes, alpha, beta, gamma)
    depth = n.bit_length() - 1
    total = 0.0
    for level in range(1, depth + 1):
        fan = 2 if n - 1 >= (1 << level) else 1
        total += 2 * (alpha + fan * nbytes * beta) + fan * nbytes * gamma
    return total


def double_binary_tree_allreduce(
    n: int,
    nbytes: float,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    nchunks: Optional[int] = None,
) -> float:
    """流水线化的 Double Binary Tree（NCCL Tree 的思路）的简化模型。

    两棵互补二叉树各负责 M/2；每个 rank 在一棵树里是内部节点（收 2 个孩子的 chunk），
    在另一棵里是叶子，于是流水稳定后每步每个 rank 的端口负载约为 M/k（k 为 chunk 数）。
    不考虑 Reduce 与 Broadcast 两个阶段的重叠：

        T(k) = 2·(D + k - 1)·(α + Mβ/k) + Mγ，D = ceil(log2 n)

    k 越大流水越满（带宽项 → 2Mβ，与 Ring 同阶），但 α 次数越多；
    nchunks 为 None 时取连续最优 k* = sqrt((D-1)·Mβ/α) 附近的整数。
    """
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    depth = _ceil_log2(n)

    def t(k: int) -> float:
        return 2 * (depth + k - 1) * (alpha + nbytes * beta / k) + nbytes * gamma

    if nchunks is not None:
        if nchunks < 1:
            raise ValueError("nchunks 必须 >= 1")
        return t(nchunks)
    max_k = max(1, int(nbytes))  # chunk 不小于 1 字节
    if alpha == 0:
        return t(max_k)
    k_star = math.sqrt(max(depth - 1, 0) * nbytes * beta / alpha)
    candidates = {1, max(1, min(max_k, math.floor(k_star))), max(1, min(max_k, math.ceil(k_star)))}
    return min(t(k) for k in candidates)


def hierarchical_allreduce(
    nnodes: int,
    ppn: int,
    nbytes: float,
    alpha_intra: float,
    beta_intra: float,
    alpha_inter: float,
    beta_inter: float,
    gamma: float = 0.0,
) -> float:
    """二维/分层 AllReduce：节点内 Ring RS → 节点间 Ring AllReduce（每 rank 只带 M/ppn）→ 节点内 Ring AG。

    假设每个 rank 有自己的节点间链路（多轨，rail-optimized），ppn 个分片的节点间 AllReduce 并行进行。

        T = 2(g-1)α_i + 2(g-1)/g·Mβ_i
          + 2(h-1)α_e + 2(h-1)/h·(M/g)·β_e
          + γ 项
    其中 g = ppn，h = nnodes。跨节点字节数被除以 g，这是分层算法的核心收益。
    """
    _check(nnodes, nbytes, alpha_inter, beta_inter, gamma)
    _check(ppn, nbytes, alpha_intra, beta_intra, gamma)
    g, h = ppn, nnodes
    t = 0.0
    if g > 1:
        t += 2 * (g - 1) * alpha_intra + 2 * (g - 1) / g * nbytes * beta_intra + (g - 1) / g * nbytes * gamma
    if h > 1:
        shard = nbytes / g
        t += 2 * (h - 1) * alpha_inter + 2 * (h - 1) / h * shard * beta_inter + (h - 1) / h * shard * gamma
    return t


def flat_ring_allreduce_two_level(
    nnodes: int,
    ppn: int,
    nbytes: float,
    alpha_intra: float,
    beta_intra: float,
    alpha_inter: float,
    beta_inter: float,
    gamma: float = 0.0,
) -> float:
    """对照组：把 n = nnodes·ppn 个 rank 排成一个大环，每步都被最慢的跨节点边拖住（教学上的最坏情况）。

    真实 NCCL 的环在每个节点只跨一次网卡，并用多个 channel 摊到多张网卡上，比这个对照好得多。
    """
    n = nnodes * ppn
    if nnodes == 1:
        return ring_allreduce(n, nbytes, alpha_intra, beta_intra, gamma)
    return ring_allreduce(n, nbytes, max(alpha_intra, alpha_inter), max(beta_intra, beta_inter), gamma)


# ---------------------------------------------------------------------------
# ReduceScatter / AllGather / All2All
# ---------------------------------------------------------------------------


def ring_reduce_scatter(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """Ring ReduceScatter：(n-1)α + (n-1)/n·Mβ + (n-1)/n·Mγ，M 为每 rank 输入大小。"""
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    return (n - 1) * alpha + (n - 1) / n * nbytes * (beta + gamma)


def ring_allgather(n: int, nbytes: float, alpha: float, beta: float) -> float:
    """Ring AllGather：(n-1)α + (n-1)/n·Mβ，M 为每 rank 输出总大小。"""
    _check(n, nbytes, alpha, beta)
    if n == 1:
        return 0.0
    return (n - 1) * alpha + (n - 1) / n * nbytes * beta


def bruck_allgather(n: int, nbytes: float, alpha: float, beta: float) -> float:
    """Bruck AllGather（任意 n）：ceil(log2 n)·α + (n-1)/n·Mβ。延迟对数级、带宽仍最优。"""
    _check(n, nbytes, alpha, beta)
    if n == 1:
        return 0.0
    return _ceil_log2(n) * alpha + (n - 1) / n * nbytes * beta


def pairwise_all2all(n: int, nbytes: float, alpha: float, beta: float) -> float:
    """Pairwise（或直接 P2P 逐对）All2All：(n-1)α + (n-1)/n·Mβ，M 为每 rank 发送总大小。"""
    _check(n, nbytes, alpha, beta)
    if n == 1:
        return 0.0
    return (n - 1) * alpha + (n - 1) / n * nbytes * beta


def bruck_all2all(n: int, nbytes: float, alpha: float, beta: float) -> float:
    """Bruck All2All：ceil(log2 n) 步，每步约发一半的块：ceil(log2 n)·(α + Mβ/2)。

    n 为 2 的幂时每步恰好发 n/2 个块即 M/2；否则为近似。另需本地旋转两次（未计入）。
    用更多字节换更少的 α，只在小消息、rank 多时划算。
    """
    _check(n, nbytes, alpha, beta)
    if n == 1:
        return 0.0
    return _ceil_log2(n) * (alpha + nbytes / 2 * beta)


# ---------------------------------------------------------------------------
# Full-mesh（机内 NVSwitch：一个 step 内同时与 N-1 个对端交换，共享本 rank 的端口带宽）
# ---------------------------------------------------------------------------


def fullmesh_allreduce_oneshot(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """One-shot：α + (n-1)·Mβ + (n-1)·Mγ。1 个 α，但字节是 Ring 的约 n/2 倍。"""
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    return alpha + (n - 1) * nbytes * (beta + gamma)


def fullmesh_allreduce_twoshot(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """Two-shot（RS + AG）：2α + 2(n-1)/n·Mβ + (n-1)/n·Mγ。β 项与 Ring 相同，α 只有 2 个。"""
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    return 2 * alpha + 2 * (n - 1) / n * nbytes * beta + (n - 1) / n * nbytes * gamma


def fullmesh_reduce_scatter(n: int, nbytes: float, alpha: float, beta: float, gamma: float = 0.0) -> float:
    """Full-mesh ReduceScatter：α + (n-1)/n·M(β+γ)，M 为每 rank 输入大小。"""
    _check(n, nbytes, alpha, beta, gamma)
    if n == 1:
        return 0.0
    return alpha + (n - 1) / n * nbytes * (beta + gamma)


def fullmesh_allgather(n: int, nbytes: float, alpha: float, beta: float) -> float:
    """Full-mesh AllGather：α + (n-1)/n·Mβ，M 为每 rank 输出总大小。"""
    _check(n, nbytes, alpha, beta)
    if n == 1:
        return 0.0
    return alpha + (n - 1) / n * nbytes * beta


def fullmesh_all2all(n: int, nbytes: float, alpha: float, beta: float) -> float:
    """Full-mesh All2All：α + (n-1)/n·Mβ，M 为每 rank 发送总大小。"""
    return fullmesh_allgather(n, nbytes, alpha, beta)


# ---------------------------------------------------------------------------
# algbw / busbw（口径对照 third_party/nccl-tests/doc/PERFORMANCE.md）
# ---------------------------------------------------------------------------

_BUSBW_FACTOR: Dict[str, Callable[[int], float]] = {
    "allreduce": lambda n: 2 * (n - 1) / n,
    "reducescatter": lambda n: (n - 1) / n,
    "allgather": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}


def busbw_factor(collective: str, n: int) -> float:
    """busbw = algbw × factor。collective 取 allreduce/reducescatter/allgather/alltoall/broadcast/reduce。"""
    key = collective.lower().replace("_", "").replace("-", "")
    if key not in _BUSBW_FACTOR:
        raise ValueError(f"未知集合操作: {collective}")
    if n < 1:
        raise ValueError("n 必须 >= 1")
    return _BUSBW_FACTOR[key](n)


def algbw(nbytes: float, seconds: float) -> float:
    """algbw = S / t（字节/秒）。S 的口径按 nccl-tests：AllReduce 为每 rank 的 buffer 大小。"""
    if seconds <= 0:
        raise ValueError("seconds 必须为正")
    return nbytes / seconds


def busbw(collective: str, n: int, nbytes: float, seconds: float) -> float:
    """busbw = algbw × busbw_factor，用来和硬件单向链路峰值比较。"""
    return algbw(nbytes, seconds) * busbw_factor(collective, n)


# ---------------------------------------------------------------------------
# 从模拟记录反推耗时
# ---------------------------------------------------------------------------


def cost_from_trace(result, alpha: float, beta: float) -> float:
    """按 α-β 模型给模拟结果计时：Σ_step (α + β × 最忙端口字节数)。

    result 是 algorithms.CommResult（这里按鸭子类型使用，避免循环依赖）。
    与上面的解析公式对比，可以验证"公式推导"与"逐步调度"是否一致。
    """
    total = 0.0
    for st in result.steps:
        total += alpha + beta * st.max_port_elems * result.elem_bytes
    return total


# __main__ 使用的 AllReduce 模型表：名字 -> f(n, nbytes, alpha, beta)
ALLREDUCE_MODELS: Dict[str, Callable[[int, float, float, float], float]] = {
    "ring": ring_allreduce,
    "rec_doubling": recursive_doubling_allreduce,
    "halving_doubling": halving_doubling_allreduce,
    "binary_tree": binary_tree_allreduce,
    "dbtree_pipe": double_binary_tree_allreduce,
    "fm_oneshot": fullmesh_allreduce_oneshot,
    "fm_twoshot": fullmesh_allreduce_twoshot,
}
