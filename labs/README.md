# labs：可运行的学习实验

只依赖 Python 3 标准库（已在 Python 3.9 验证），不需要 GPU、numpy 或 pytest。所有命令都在仓库根目录执行。

## ccl_sim：集合通信算法教学模拟器

配套笔记：[`notes/03-ccl-algorithms/`](../notes/03-ccl-algorithms/README.md)。

### 跑测试

```bash
python3 -m unittest discover -s labs -t .
```

成功时末尾是 `Ran N tests ... OK`，退出码 0。`-t .` 让测试以 `labs.ccl_sim...` 的包名导入，所以必须在仓库根目录执行，`labs/__init__.py` 不能删。

只跑一个文件、看每个用例：

```bash
python3 -m unittest labs.ccl_sim.tests.test_algorithms -v
```

### 打印 α-β 估算对比表

```bash
python3 -m labs.ccl_sim                                   # 默认 n=8，α=5 us，25 GB/s
python3 -m labs.ccl_sim --nranks 64 --alpha-us 2 --bw-gbps 50
python3 -m labs.ccl_sim --nranks 6                        # 非 2 的幂，halving_doubling 显示 n/a
python3 -m labs.ccl_sim --help
```

输出三张表：

1. AllReduce：Ring、Recursive Doubling、Halving-Doubling、不切块二叉树、流水线 Double Binary Tree 的耗时（微秒）和最优者，以及 Ring 对应的 busbw。
2. 分层 AllReduce 与"把所有 rank 排成一个大环"的对比（节点内快、节点间慢）。
3. All2All：Pairwise 与 Bruck 的对比。

### 在 Python 里单步观察

```python
from labs.ccl_sim import ring_allreduce, halving_doubling_allreduce
from labs.ccl_sim import cost_model as cm

res = ring_allreduce([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400], [1000, 2000, 3000, 4000]])
print(res.outputs[0])            # [1111, 2222, 3333, 4444]
print(res.num_steps)             # 6 = 2(N-1)
for step in res.steps:
    print(step.index, step.phase, step.transfers)
print(res.sent_bytes_per_rank()) # 每 rank 发送 2(N-1)/N * L * 4 字节
print(cm.cost_from_trace(res, alpha=5e-6, beta=cm.gbps_to_beta(25)))
```

模拟器是同步 step 的教学模型，不代表 NCCL 的 channel、chunk、协议和真实拓扑；真实性能请用 nccl-tests 测量。
