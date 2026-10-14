# labs/ Agent 规则

> 先遵循仓库根 `AGENTS.md`。本文只补充可运行实验代码的局部约定。

## 模块职责

- 负责：配合 `notes/` 的可运行教学实验，用代码验证笔记中的算法推导和代价模型。
- 不负责：外部源码（在 `third_party/`，只读）；笔记正文（在 `notes/`）；真实 GPU / 网络性能测量（属于 `notes/07-perf-troubleshooting/` 中的 nccl-tests 实践）。
- 对外接口：每个实验是 `labs/` 下的一个 Python 包，可用 `python3 -m labs.<pkg>` 运行，测试统一用 `unittest` 发现。

## 目录与入口

| 路径 | 作用 | 入口 |
|---|---|---|
| `labs/__init__.py` | 让 `labs` 成为可导入包（`unittest discover -t .` 需要） | 无 |
| `labs/ccl_sim/algorithms.py` | 用 Python list 模拟 N 个 rank，逐 step 执行 Ring / Recursive Doubling / Halving-Doubling / Binary Tree AllReduce、Ring RS/AG、Pairwise All2All(v)，返回结果和每步通信记录 | `from labs.ccl_sim import ring_allreduce` 等 |
| `labs/ccl_sim/cost_model.py` | α-β(-γ) 解析公式、busbw 系数、`cost_from_trace`（按模拟记录计时） | `from labs.ccl_sim import cost_model` |
| `labs/ccl_sim/__main__.py` | 打印各算法在不同消息大小下的估算对比表 | `python3 -m labs.ccl_sim [--nranks N ...]` |
| `labs/ccl_sim/tests/` | 正确性、step 数、发送量、负向用例、代价模型单调性与"公式 = 模拟记录"一致性 | `python3 -m unittest discover -s labs -t .` |

对应笔记：`notes/03-ccl-algorithms/`（尤其 `02-alpha-beta-model.md`）。

## 依赖、数据与外部边界

- 只用 Python 标准库（当前在 Python 3.9 上验证）；不得引入 numpy、pytest 或任何需要 `pip install` 的依赖。
- 不访问网络、GPU、文件系统外部路径；测试用固定种子的 `random.Random`，结果可复现。
- 不从 `third_party/` 导入代码；需要对照外部实现时在注释里写 `third_party/<repo>/<path>`，并只引用本地实际看到的路径。

## 模块约定

- 注释与文档字符串用中文；公式记号与 `notes/03-ccl-algorithms/02-alpha-beta-model.md` 保持一致（n、M、α、β、γ）。
- 模拟器语义：同一 step 内的发送基于 step 开始时的快照，step 之间严格同步；不原地修改输入。
- 每个模拟函数返回 `CommResult`（`outputs`、`steps`、`elem_bytes`），每个 `StepRecord` 记录该步所有 `Transfer(src, dst, nelems)`。
- 新增算法时同时补：正确性测试（与朴素实现对比）、step 数/发送量测试、至少一个负向测试；若有解析公式，补"解析公式 = `cost_from_trace`"一致性测试。
- 不支持的输入抛 `ValueError`（例如 Halving-Doubling 的 N 非 2 的幂、buffer 长度不一致）。

## 命令与验证

所有命令在仓库根目录执行。

| 目的 | 命令 | 成功信号 | 不能证明什么 |
|---|---|---|---|
| 全部单元测试 | `python3 -m unittest discover -s labs -t .` | 末行 `OK`，退出码 0 | 模拟结果代表 NCCL 的真实调度或性能 |
| 单个测试文件 | `python3 -m unittest labs.ccl_sim.tests.test_algorithms -v` | 每个用例 `ok`，退出码 0 | 同上 |
| 打印 α-β 对比表 | `python3 -m labs.ccl_sim` | 打印三张表，退出码 0 | 表中数值接近实测；α、β 只是参数假设 |
| 非 2 的幂场景 | `python3 -m labs.ccl_sim --nranks 6` | `halving_doubling` 列显示 `n/a` | 真实库不能处理非 2 的幂（真实实现会折叠） |
| 仓库机械检查（含 labs 测试） | `python3 scripts/check_repo.py` | 输出 `labs: passed`；整体退出码 0 需其他模块也无问题 | 笔记技术内容正确 |
