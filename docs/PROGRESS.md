# 当前进度

- 最后更新：2026-10-14 23:40 CST
- 当前分支：`main`
- 基线提交：本轮成果即首个提交（用 `git log -1` 查看）；远程 `origin` = https://github.com/philo-zhou/ai-learning（用户 2026-10-14 授权推送）
- 工作树状态：提交后为 clean；`third_party/*/` 被 `.gitignore` 排除

## 可核实基线

- 9 个外部资料按 `third_party/repos.lock` 固定版本拉取完毕，且均无本地改动 — 证据：`bash scripts/fetch_third_party.sh check` 全部 OK；逐个 `git -C third_party/<repo> status --porcelain` 为空。
- `notes/00~08` 九个章节共 43 篇笔记；学习入口 `notes/00-roadmap/00-start-here.md`，章节索引 `notes/00-roadmap/README.md`。
- 推理并行与 EP 通信调研：`docs/references/2026-10-14-inference-parallelism-ep-research.md`（17 个来源，13 条结论经过验证）。
- `labs/ccl_sim` 算法模拟器与 α-β 代价模型，30 个单元测试。
- 治理入口：`AGENTS.md`（`CLAUDE.md`、`CODEBUDDY.md` 软链接）、`docs/`、`scripts/check_repo.py`、`scripts/fetch_third_party.sh`。

## 当前焦点

仓库建设已完成（Spec：`docs/superpowers/specs/2026-10-14-ccl-learning-repo-design.md`；Plan：`docs/superpowers/plans/2026-10-14-ccl-learning-repo.md`，状态 Completed）。下一焦点：通读 `notes/00-roadmap/00-start-here.md`，再按其第 7 节的主线顺序进行，前两步对应 `notes/00-roadmap/03-learning-path.md` 第 1 阶段（集合通信算法与代价模型）。

## 本轮完成并验证

- 外部资料拉取与版本锁定 — 证据：`bash scripts/fetch_third_party.sh check`；克隆路径在临时目录用单行 lock 验证通过。
- 笔记首批覆盖全部章节，且笔记中引用的 `third_party/...` 路径均存在 — 证据：`python3 scripts/check_repo.py docs`。
- 模拟器正确性、步数与通信量 — 证据：`python3 scripts/check_repo.py labs`。

## 验证证据

| 状态 | 命令或检查 | 范围 | 结果摘要 |
|---|---|---|---|
| passed | `python3 scripts/check_repo.py` | docs + labs + third-party | 三步均 passed，退出码 0（macOS，Python 3，2026-10-14） |
| passed | `python3 -m labs.ccl_sim` | 模拟器 CLI | 打印 AllReduce、分层、All2All 三张对比表，退出码 0 |
| passed | `python3 scripts/check_repo.py bogus`；`bash scripts/fetch_third_party.sh bogus` | 负向：非法参数 | 均退出 2 |
| not-run | nccl-tests、perftest、vLLM/SGLang/DeepEP/Triton-distributed 示例 | GPU/RDMA 实验 | 本机无 GPU 与 RDMA 网卡 |
| passed | deep-research 工作流 | 推理并行与 EP 调研 | 102 个 agent 无错误；25 条结论经 3 票验证，24 条确认、1 条推翻 |
| partial | 笔记技术内容 | `notes/` | 由并行 Agent 依据本地源码撰写，只有路径经机械核对；函数名、结论需阅读时抽查 |

## 阻塞、授权与恢复条件

- GPU/RDMA 实验：需要一台带 NVIDIA GPU（最好 ≥2 卡 NVLink）和 RDMA 网卡的机器；拿到后按 `notes/07-perf-troubleshooting/README.md` 运行，结果补入对应笔记。

## 风险与未知项

- 笔记技术细节可能有误：各篇末尾"待深入"列出了待确认项及确认方式；阅读时对照源码修正。
- PyTorch 源码未克隆：`ProcessGroupNCCL` C++ 内部标为待确认；确认方式：经用户同意后把 pytorch/pytorch 加入 `repos.lock`。
- 版本时效：本地 NCCL 2.32.3、DeepEP V2.5 与网上多数解读不同（如 DeepEP 已移除 NVSHMEM、NCCL 无 `ext-net/` 目录），笔记以本地版本为准。
- JD 中"scale up 等互联技术""自研 RDMA 协议"的具体所指：见 `docs/CONTEXT.md` 待澄清表。

## 下一步

1. 通读学习入口 `notes/00-roadmap/00-start-here.md`，做完其中的面试自测，并更新 `notes/00-roadmap/02-jd-breakdown.md` 自评。
2. 阶段 1：读 `notes/03-ccl-algorithms/02-alpha-beta-model.md` 与 ai-infra-book §6.4，用 `python3 -m labs.ccl_sim --alpha-us ... --bw-gbps ...` 代入书中参数复现 Ring/Tree 约 680 KiB 交点，更新 `notes/00-roadmap/02-jd-breakdown.md` 自评。
3. 为本仓库选定 License（远程是公开仓库时尤其需要）。
4. 补齐学习入口"待深入"中的调研缺口：CP/SP 推理通信、vLLM/SGLang 其余 all2all 后端、MegaScale-Infer M2N 细节。
