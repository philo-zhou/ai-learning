"""``python3 -m labs.ccl_sim``：打印各算法在不同消息大小下的 α-β 估算对比表。

示例：
    python3 -m labs.ccl_sim
    python3 -m labs.ccl_sim --nranks 16 --alpha-us 2 --bw-gbps 50
    python3 -m labs.ccl_sim --nranks 6      # 非 2 的幂：halving_doubling 显示 n/a

数值只用于比较算法的相对趋势（小消息看 α、大消息看 β），不代表 NCCL 实测。
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence, TextIO

from . import cost_model as cm

SIZES = [1 << s for s in range(10, 31, 2)]  # 1 KiB .. 1 GiB，按 4 倍递增


def human_bytes(n: int) -> str:
    for unit, shift in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if n >= (1 << shift):
            return f"{n / (1 << shift):g} {unit}"
    return f"{n} B"


def _fmt_us(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    us = seconds * 1e6
    return f"{us:.1f}" if us < 1e5 else f"{us:.3e}"


def _table(header: List[str], rows: List[List[str]], out: TextIO) -> None:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = " | ".join(h.rjust(w) for h, w in zip(header, widths))
    out.write(line + "\n")
    out.write("-+-".join("-" * w for w in widths) + "\n")
    for r in rows:
        out.write(" | ".join(c.rjust(w) for c, w in zip(r, widths)) + "\n")


def allreduce_table(n: int, alpha: float, beta: float, out: TextIO) -> None:
    names = list(cm.ALLREDUCE_MODELS)
    rows = []
    for size in SIZES:
        times = {}
        for name, fn in cm.ALLREDUCE_MODELS.items():
            try:
                times[name] = fn(n, size, alpha, beta)
            except ValueError:
                times[name] = None
        valid = {k: v for k, v in times.items() if v is not None}
        best = min(valid, key=valid.get)
        ring_t = times["ring"]
        ring_busbw = cm.busbw("allreduce", n, size, ring_t) / 1e9 if ring_t else 0.0
        rows.append([human_bytes(size)] + [_fmt_us(times[k]) for k in names] + [best, f"{ring_busbw:.1f}"])
    _table(["size"] + [f"{k}(us)" for k in names] + ["best", "ring busbw(GB/s)"], rows, out)


def hierarchical_table(
    nnodes: int, ppn: int, alpha_i: float, beta_i: float, alpha_e: float, beta_e: float, out: TextIO
) -> None:
    rows = []
    for size in SIZES:
        h = cm.hierarchical_allreduce(nnodes, ppn, size, alpha_i, beta_i, alpha_e, beta_e)
        f = cm.flat_ring_allreduce_two_level(nnodes, ppn, size, alpha_i, beta_i, alpha_e, beta_e)
        rows.append([human_bytes(size), _fmt_us(f), _fmt_us(h), f"{f / h:.2f}x" if h > 0 else "-"])
    _table(["size", "flat_ring(us)", "hierarchical(us)", "speedup"], rows, out)


def all2all_table(n: int, alpha: float, beta: float, out: TextIO) -> None:
    rows = []
    for size in SIZES:
        p = cm.pairwise_all2all(n, size, alpha, beta)
        b = cm.bruck_all2all(n, size, alpha, beta)
        rows.append([human_bytes(size), _fmt_us(p), _fmt_us(b), "bruck" if b < p else "pairwise"])
    _table(["size(per rank)", "pairwise(us)", "bruck(us)", "best"], rows, out)


def main(argv: Optional[Sequence[str]] = None, out: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m labs.ccl_sim", description="α-β 模型下集合通信算法耗时对比")
    parser.add_argument("--nranks", type=int, default=8, help="rank 数（默认 8）")
    parser.add_argument("--alpha-us", type=float, default=5.0, help="每步启动开销 α，微秒（默认 5）")
    parser.add_argument("--bw-gbps", type=float, default=25.0, help="单向链路带宽，GB/s（默认 25，约 200 Gb/s 网卡）")
    parser.add_argument("--nnodes", type=int, default=4, help="分层算法：节点数（默认 4）")
    parser.add_argument("--ppn", type=int, default=8, help="分层算法：每节点 rank 数（默认 8）")
    parser.add_argument("--intra-alpha-us", type=float, default=1.0, help="节点内 α，微秒（默认 1）")
    parser.add_argument("--intra-bw-gbps", type=float, default=200.0, help="节点内单向带宽 GB/s（默认 200）")
    args = parser.parse_args(argv)
    if args.nranks < 1:
        parser.error("--nranks 必须 >= 1")

    alpha = args.alpha_us * 1e-6
    beta = cm.gbps_to_beta(args.bw_gbps)
    n = args.nranks

    out.write(f"== AllReduce，n={n}，alpha={args.alpha_us} us，bw={args.bw_gbps} GB/s ==\n")
    out.write(f"   α·B 交叉点 ≈ {human_bytes(int(alpha / beta))}：远小于它看延迟项，远大于它看带宽项\n")
    allreduce_table(n, alpha, beta, out)

    out.write(
        f"\n== 分层 AllReduce，{args.nnodes} 节点 × {args.ppn} rank/节点；"
        f"节点内 α={args.intra_alpha_us} us、{args.intra_bw_gbps} GB/s，节点间 α={args.alpha_us} us、{args.bw_gbps} GB/s ==\n"
    )
    hierarchical_table(
        args.nnodes,
        args.ppn,
        args.intra_alpha_us * 1e-6,
        cm.gbps_to_beta(args.intra_bw_gbps),
        alpha,
        beta,
        out,
    )

    out.write(f"\n== All2All，n={n}（size 为每 rank 发送总量）==\n")
    all2all_table(n, alpha, beta, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
