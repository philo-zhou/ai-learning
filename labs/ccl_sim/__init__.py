"""ccl_sim：集合通信算法的教学模拟器（纯 Python 标准库）。

- ``algorithms``：用 Python list 模拟 N 个 rank 的 buffer，按 step 同步执行
  Ring / Recursive Doubling / Halving-Doubling / Binary Tree / Full-mesh（one-shot、two-shot）AllReduce、
  Full-mesh ReduceScatter / AllGather / All2All 与 Pairwise All2All，
  返回结果和逐步通信记录。
- ``cost_model``：α-β(-γ) 代价模型的解析公式，以及从模拟记录反推耗时的 ``cost_from_trace``。

运行 ``python3 -m labs.ccl_sim`` 打印各算法在不同消息大小下的估算对比表。
"""

from .algorithms import (
    CommResult,
    StepRecord,
    Transfer,
    binary_tree_allreduce,
    fullmesh_all2all,
    fullmesh_allgather,
    fullmesh_allreduce_oneshot,
    fullmesh_allreduce_twoshot,
    fullmesh_reduce_scatter,
    halving_doubling_allreduce,
    pairwise_all2all,
    pairwise_all2allv,
    recursive_doubling_allreduce,
    ring_allgather,
    ring_allreduce,
    ring_reduce_scatter,
)

__all__ = [
    "CommResult",
    "StepRecord",
    "Transfer",
    "binary_tree_allreduce",
    "fullmesh_all2all",
    "fullmesh_allgather",
    "fullmesh_allreduce_oneshot",
    "fullmesh_allreduce_twoshot",
    "fullmesh_reduce_scatter",
    "halving_doubling_allreduce",
    "pairwise_all2all",
    "pairwise_all2allv",
    "recursive_doubling_allreduce",
    "ring_allgather",
    "ring_allreduce",
    "ring_reduce_scatter",
]
