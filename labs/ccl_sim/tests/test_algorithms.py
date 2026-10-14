"""ccl_sim 的正确性与代价模型测试。

运行（仓库根目录）：
    python3 -m unittest discover -s labs -t .
"""

from __future__ import annotations

import io
import math
import random
import unittest

from labs.ccl_sim import algorithms as alg
from labs.ccl_sim import cost_model as cm
from labs.ccl_sim.__main__ import main as cli_main

ALLREDUCE_FUNCS = [
    alg.ring_allreduce,
    alg.recursive_doubling_allreduce,
    alg.halving_doubling_allreduce,
    alg.binary_tree_allreduce,
]


def make_inputs(rng: random.Random, n: int, length: int):
    """随机整数输入：用整数避免浮点求和顺序带来的误差。"""
    return [[rng.randint(-1000, 1000) for _ in range(length)] for _ in range(n)]


def naive_sum(inputs):
    return [sum(col) for col in zip(*inputs)] if inputs and inputs[0] else []


def naive_all2all(inputs):
    n = len(inputs)
    k = len(inputs[0]) // n
    return [[x for j in range(n) for x in inputs[j][r * k : (r + 1) * k]] for r in range(n)]


class TestAllReduceCorrectness(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20261014)

    def test_all_algorithms_match_naive_sum(self):
        for fn in ALLREDUCE_FUNCS:
            for n in [1, 2, 4, 8, 16]:
                for length in [0, 1, 7, 16, 33]:
                    with self.subTest(fn=fn.__name__, n=n, length=length):
                        inputs = make_inputs(self.rng, n, length)
                        expected = naive_sum(inputs)
                        res = fn(inputs)
                        self.assertEqual(len(res.outputs), n)
                        for r in range(n):
                            self.assertEqual(res.outputs[r], expected)

    def test_non_power_of_two_supported_by_ring_rd_tree(self):
        for fn in [alg.ring_allreduce, alg.recursive_doubling_allreduce, alg.binary_tree_allreduce]:
            for n in [3, 5, 6, 7, 9, 12]:
                with self.subTest(fn=fn.__name__, n=n):
                    inputs = make_inputs(self.rng, n, 23)
                    res = fn(inputs)
                    expected = naive_sum(inputs)
                    for out in res.outputs:
                        self.assertEqual(out, expected)

    def test_inputs_not_mutated(self):
        inputs = make_inputs(self.rng, 4, 8)
        snapshot = [list(b) for b in inputs]
        for fn in ALLREDUCE_FUNCS:
            fn(inputs)
        self.assertEqual(inputs, snapshot)

    def test_float_inputs_close(self):
        inputs = [[self.rng.random() for _ in range(32)] for _ in range(8)]
        expected = naive_sum(inputs)
        for fn in ALLREDUCE_FUNCS:
            for out in fn(inputs).outputs:
                for a, b in zip(out, expected):
                    self.assertTrue(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))

    def test_custom_reduce_op_max(self):
        inputs = make_inputs(self.rng, 8, 10)
        expected = [max(col) for col in zip(*inputs)]
        for fn in ALLREDUCE_FUNCS:
            with self.subTest(fn=fn.__name__):
                for out in fn(inputs, op=max).outputs:
                    self.assertEqual(out, expected)

    def test_allreduce_equals_reduce_scatter_plus_allgather(self):
        inputs = make_inputs(self.rng, 6, 20)
        rs = alg.ring_reduce_scatter(inputs)
        bounds = alg.chunk_bounds(20, 6)
        expected = naive_sum(inputs)
        for r in range(6):
            self.assertEqual(rs.outputs[r], expected[bounds[r][0] : bounds[r][1]])
        ag = alg.ring_allgather(rs.outputs)
        ar = alg.ring_allreduce(inputs)
        self.assertEqual(ag.outputs, ar.outputs)
        self.assertEqual(rs.num_steps + ag.num_steps, ar.num_steps)


class TestStepsAndVolume(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)

    def test_ring_steps_and_bytes(self):
        for n in [2, 3, 4, 5, 8]:
            length = n * 6  # 可整除，便于精确比较
            res = alg.ring_allreduce(make_inputs(self.rng, n, length), elem_bytes=4)
            self.assertEqual(res.num_steps, 2 * (n - 1))
            expected_bytes = 2 * (n - 1) * length * 4 // n
            self.assertEqual(res.sent_bytes_per_rank(), [expected_bytes] * n)
            # 每步每个 rank 只发 L/N
            for st in res.steps:
                self.assertEqual(st.sent_elems_by_rank(), [length // n] * n)

    def test_recursive_doubling_steps(self):
        for n in [2, 4, 8, 16, 32]:
            res = alg.recursive_doubling_allreduce(make_inputs(self.rng, n, 10))
            k = int(math.log2(n))
            self.assertEqual(res.num_steps, k)
            self.assertEqual(res.sent_elems_per_rank(), [k * 10] * n)

    def test_recursive_doubling_non_power_of_two_extra_steps(self):
        res = alg.recursive_doubling_allreduce(make_inputs(self.rng, 6, 5))
        self.assertEqual(res.num_steps, 2 + 2)  # log2(4) + fold-in + fold-out
        self.assertEqual(res.steps[0].phase, "fold-in")
        self.assertEqual(res.steps[-1].phase, "fold-out")

    def test_halving_doubling_steps_and_bytes(self):
        for n in [2, 4, 8, 16]:
            length = n * 4
            res = alg.halving_doubling_allreduce(make_inputs(self.rng, n, length))
            self.assertEqual(res.num_steps, 2 * int(math.log2(n)))
            self.assertEqual(res.sent_elems_per_rank(), [2 * (n - 1) * length // n] * n)
            # ReduceScatter 阶段每步发送量减半
            rs_sends = [st.sent_elems_by_rank()[0] for st in res.steps[: res.num_steps // 2]]
            self.assertEqual(rs_sends, [length >> (i + 1) for i in range(len(rs_sends))])

    def test_binary_tree_steps(self):
        for n in [1, 2, 3, 4, 7, 8, 15, 16]:
            res = alg.binary_tree_allreduce(make_inputs(self.rng, n, 4))
            self.assertEqual(res.num_steps, 2 * (n.bit_length() - 1))

    def test_pairwise_all2all_correct_steps_bytes(self):
        for n in [1, 2, 3, 4, 8]:
            k = 3
            inputs = make_inputs(self.rng, n, n * k)
            res = alg.pairwise_all2all(inputs, elem_bytes=2)
            self.assertEqual(res.outputs, naive_all2all(inputs))
            self.assertEqual(res.num_steps, n - 1)
            self.assertEqual(res.sent_bytes_per_rank(), [(n - 1) * k * 2] * n)
            for st in res.steps:  # 每步每 rank 恰好发一条、收一条
                self.assertEqual(st.sent_elems_by_rank(), [k] * n)
                self.assertEqual(st.recv_elems_by_rank(), [k] * n)

    def test_pairwise_all2allv_uneven_blocks(self):
        n = 4
        sizes = [[self.rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
        sizes[0][3] = 50  # 制造一个热点（MoE 中某个专家被大量 token 选中）
        blocks = [[[self.rng.randint(0, 9) for _ in range(sizes[r][j])] for j in range(n)] for r in range(n)]
        res = alg.pairwise_all2allv(blocks)
        for r in range(n):
            for j in range(n):
                self.assertEqual(res.outputs[r][j], blocks[j][r])
        # 热点所在的那一步被最大消息拖住
        self.assertEqual(max(st.max_port_elems for st in res.steps), 50)


class TestNegative(unittest.TestCase):
    def test_halving_doubling_rejects_non_power_of_two(self):
        for n in [3, 5, 6, 7, 12]:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    alg.halving_doubling_allreduce([[1, 2, 3, 4]] * n)
                with self.assertRaises(ValueError):
                    cm.halving_doubling_allreduce(n, 1024, 1e-6, 1e-9)

    def test_mismatched_buffer_lengths_rejected(self):
        for fn in ALLREDUCE_FUNCS + [alg.pairwise_all2all, alg.ring_reduce_scatter]:
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn([[1, 2, 3, 4], [1, 2, 3]])

    def test_empty_rank_list_rejected(self):
        for fn in ALLREDUCE_FUNCS + [alg.pairwise_all2all, alg.ring_allgather]:
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn([])

    def test_all2all_requires_divisible_length(self):
        with self.assertRaises(ValueError):
            alg.pairwise_all2all([[1, 2, 3]] * 2)
        with self.assertRaises(ValueError):
            alg.pairwise_all2allv([[[1], [2]], [[3]]])

    def test_cost_model_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            cm.ring_allreduce(0, 1024, 1e-6, 1e-9)
        with self.assertRaises(ValueError):
            cm.ring_allreduce(4, -1, 1e-6, 1e-9)
        with self.assertRaises(ValueError):
            cm.busbw_factor("allscatter", 8)
        with self.assertRaises(ValueError):
            cm.gbps_to_beta(0)


class TestCostModel(unittest.TestCase):
    ALPHA = 5e-6
    BETA = cm.gbps_to_beta(25)

    def _all_models(self, n):
        models = dict(cm.ALLREDUCE_MODELS)
        if n & (n - 1):
            models.pop("halving_doubling")
        return models

    def test_monotonic_in_size_alpha_beta(self):
        sizes = [0, 1 << 10, 1 << 16, 1 << 20, 1 << 26, 1 << 30]
        for n in [2, 6, 8, 64]:
            for name, fn in self._all_models(n).items():
                with self.subTest(model=name, n=n):
                    ts = [fn(n, s, self.ALPHA, self.BETA) for s in sizes]
                    self.assertEqual(ts, sorted(ts))
                    self.assertLessEqual(fn(n, 1 << 20, self.ALPHA, self.BETA), fn(n, 1 << 20, 2 * self.ALPHA, self.BETA))
                    self.assertLessEqual(fn(n, 1 << 20, self.ALPHA, self.BETA), fn(n, 1 << 20, self.ALPHA, 2 * self.BETA))

    def test_ring_monotonic_in_n(self):
        ts = [cm.ring_allreduce(n, 1 << 20, self.ALPHA, self.BETA) for n in range(1, 65)]
        self.assertEqual(ts, sorted(ts))

    def test_small_vs_large_message_regimes(self):
        n = 16
        small, large = 1 << 10, 1 << 30
        ring_s = cm.ring_allreduce(n, small, self.ALPHA, self.BETA)
        rd_s = cm.recursive_doubling_allreduce(n, small, self.ALPHA, self.BETA)
        ring_l = cm.ring_allreduce(n, large, self.ALPHA, self.BETA)
        rd_l = cm.recursive_doubling_allreduce(n, large, self.ALPHA, self.BETA)
        hd_l = cm.halving_doubling_allreduce(n, large, self.ALPHA, self.BETA)
        self.assertLess(rd_s, ring_s)  # 小消息：log 级延迟胜出
        self.assertLess(ring_l, rd_l)  # 大消息：带宽最优胜出
        self.assertLessEqual(hd_l, ring_l)  # Rabenseifner 带宽项与 Ring 相同、α 更少

    def test_ring_bandwidth_term_approaches_2M_over_B(self):
        size = 1 << 30
        for n in [2, 8, 64, 1024]:
            t = cm.ring_allreduce(n, size, 0.0, self.BETA)
            self.assertAlmostEqual(t, 2 * (n - 1) / n * size * self.BETA)
            # busbw 恰好等于链路带宽：这就是 nccl-tests busbw 的含义
            self.assertAlmostEqual(cm.busbw("allreduce", n, size, t) / 1e9, 25.0)

    def test_busbw_factors_match_nccl_tests_doc(self):
        n = 8
        self.assertAlmostEqual(cm.busbw_factor("allreduce", n), 2 * 7 / 8)
        self.assertAlmostEqual(cm.busbw_factor("reduce_scatter", n), 7 / 8)
        self.assertAlmostEqual(cm.busbw_factor("allgather", n), 7 / 8)
        self.assertAlmostEqual(cm.busbw_factor("alltoall", n), 7 / 8)
        self.assertEqual(cm.busbw_factor("broadcast", n), 1.0)
        self.assertEqual(cm.busbw_factor("reduce", n), 1.0)

    def test_analytic_formula_matches_simulated_trace(self):
        """解析公式 == 按模拟器每步最忙端口计时，验证推导与调度一致。"""
        rng = random.Random(3)
        elem_bytes = 4
        cases = [
            (alg.ring_allreduce, cm.ring_allreduce, [2, 3, 5, 8]),
            (alg.recursive_doubling_allreduce, cm.recursive_doubling_allreduce, [2, 4, 6, 7, 8]),
            (alg.halving_doubling_allreduce, cm.halving_doubling_allreduce, [2, 4, 8, 16]),
            (alg.binary_tree_allreduce, cm.binary_tree_allreduce, [2, 3, 4, 5, 7, 8, 9]),
        ]
        for sim_fn, model_fn, ns in cases:
            for n in ns:
                with self.subTest(algo=sim_fn.__name__, n=n):
                    length = n * 8
                    res = sim_fn(make_inputs(rng, n, length), elem_bytes=elem_bytes)
                    nbytes = length * elem_bytes
                    self.assertAlmostEqual(
                        cm.cost_from_trace(res, self.ALPHA, self.BETA),
                        model_fn(n, nbytes, self.ALPHA, self.BETA),
                        places=12,
                    )
        for n in [2, 3, 8]:
            res = alg.pairwise_all2all(make_inputs(rng, n, n * 5), elem_bytes=elem_bytes)
            self.assertAlmostEqual(
                cm.cost_from_trace(res, self.ALPHA, self.BETA),
                cm.pairwise_all2all(n, n * 5 * elem_bytes, self.ALPHA, self.BETA),
                places=12,
            )

    def test_hierarchical_beats_flat_ring_when_inter_link_is_slow(self):
        args = (1 << 26, 1e-6, cm.gbps_to_beta(200), 5e-6, cm.gbps_to_beta(25))
        self.assertLess(cm.hierarchical_allreduce(4, 8, *args), cm.flat_ring_allreduce_two_level(4, 8, *args))
        # 单节点时分层退化为节点内 Ring
        self.assertAlmostEqual(
            cm.hierarchical_allreduce(1, 8, *args), cm.ring_allreduce(8, args[0], args[1], args[2])
        )

    def test_double_binary_tree_chunking(self):
        n, size = 64, 1 << 26
        auto = cm.double_binary_tree_allreduce(n, size, self.ALPHA, self.BETA)
        one = cm.double_binary_tree_allreduce(n, size, self.ALPHA, self.BETA, nchunks=1)
        self.assertLess(auto, one)  # 流水切块显著优于不切块
        with self.assertRaises(ValueError):
            cm.double_binary_tree_allreduce(n, size, self.ALPHA, self.BETA, nchunks=0)

    def test_bruck_vs_pairwise_all2all(self):
        n = 64
        self.assertLess(cm.bruck_all2all(n, 1 << 10, self.ALPHA, self.BETA), cm.pairwise_all2all(n, 1 << 10, self.ALPHA, self.BETA))
        self.assertGreater(cm.bruck_all2all(n, 1 << 28, self.ALPHA, self.BETA), cm.pairwise_all2all(n, 1 << 28, self.ALPHA, self.BETA))

    def test_single_rank_costs_zero(self):
        for name, fn in cm.ALLREDUCE_MODELS.items():
            with self.subTest(model=name):
                self.assertEqual(fn(1, 1 << 20, self.ALPHA, self.BETA), 0.0)


class TestCli(unittest.TestCase):
    def test_main_prints_tables(self):
        buf = io.StringIO()
        self.assertEqual(cli_main(["--nranks", "8"], out=buf), 0)
        text = buf.getvalue()
        for token in ["AllReduce", "ring(us)", "halving_doubling(us)", "hierarchical(us)", "All2All", "1 GiB"]:
            self.assertIn(token, text)

    def test_main_non_power_of_two_marks_na(self):
        buf = io.StringIO()
        self.assertEqual(cli_main(["--nranks", "6"], out=buf), 0)
        self.assertIn("n/a", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
