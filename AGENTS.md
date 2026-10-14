# Agent 工作协议

> `README.md` 面向使用者；本文件是 Agent 的仓库入口和跨模块规则源。`CLAUDE.md`、`CODEBUDDY.md` 是指向本文件的软链接。

## 项目与职责边界

本仓库是个人的**集合通信（CCL）学习仓库**，服务于求职目标"字节跳动基础设施网络团队 · 集合通信库研发"（JD 拆解见 `notes/00-roadmap/02-jd-breakdown.md`）。内容按三层组织：下层 RDMA / scale-up 互联 → 中层集合通信库（NCCL）→ 上层训练与推理框架。

- Agent 可以自行完成：在 `notes/`、`labs/`、`docs/`、`scripts/` 内写笔记、实验代码和文档；运行 `scripts/check_repo.py`；只读阅读 `third_party/`。
- 必须由用户决定或授权：修改 `third_party/repos.lock` 的版本、`git commit`/`push`、创建远程仓库、任何需要 GPU/RDMA 机器或付费云资源的实验、对学习路线和求职策略的取舍。

## 进入仓库

1. 读取 `README.md` 和 `docs/PROGRESS.md`。
2. 读取 `docs/AGENTS.md`，再按任务路由到下方知识位置。
3. 运行 `git status --short` 与 `git log --oneline -5`，区分已有改动和本轮改动。
4. 写笔记前读 `notes/AGENTS.md`；改实验代码前读 `labs/AGENTS.md`；读外部源码前读 `third_party/AGENTS.md`。
5. 运行 `python3 scripts/check_repo.py` 建立基线。

## 知识路由

| 要回答的问题 | 权威位置 |
|---|---|
| 仓库用途和第一次成功 | `README.md` |
| 当前学习状态、证据、阻塞和下一步 | `docs/PROGRESS.md` |
| 学习入口（推理并行 × 通信 × EP 主线） | `notes/00-roadmap/00-start-here.md` |
| 学习路线、JD 拆解、笔记总入口 | `notes/00-roadmap/README.md` |
| 外部调研结论（带来源与置信度） | `docs/references/`（如 `docs/references/2026-10-14-inference-parallelism-ep-research.md`） |
| 术语（algbw/busbw、QP、channel、scale-up 等） | `docs/CONTEXT.md` |
| 仓库结构与三层知识映射 | `docs/architecture/overview.md` |
| 验证选择和证据边界 | `docs/TESTING.md` |
| 外部资料版本与取舍 | `third_party/repos.lock`、`docs/references/third-party.md` |
| 仓库建设的设计与验收 | `docs/superpowers/specs/` |
| 实施顺序 | `docs/superpowers/plans/` |
| 可观察变化 | `CHANGELOG.md` |

## 执行能力

| 能力 | 权威入口 | 前置条件 | 副作用 | 成功信号 | 失败或阻塞语义 |
|---|---|---|---|---|---|
| setup | `bash scripts/fetch_third_party.sh` | git、可访问 github.com | 联网；在 `third_party/<name>/` 创建浅克隆；已有目录不覆盖 | 每行 `OK <name> <commit>`，退出码 0 | 缺失/克隆失败/版本不一致退出 1；未知 mode 退出 2 |
| dev/run | `python3 -m labs.ccl_sim` | Python 3（仅标准库） | 无，只打印 | 输出各算法 α-β 估算表，退出码 0 | 非零即失败 |
| verify | 最小：`python3 scripts/check_repo.py docs`；完整：`python3 scripts/check_repo.py` | Python 3；完整模式需 setup 完成 | 只读（运行 unittest 会生成 `__pycache__`，已 gitignore） | 每步打印 `passed`，退出码 0 | 任一步失败退出 1；参数错误退出 2 |

GPU/RDMA 实验（nccl-tests、perftest、vLLM/SGLang/DeepEP/Triton-distributed 示例）本机不可运行：默认 `not-run`，在笔记中写明所需硬件，不得声称跑过。

## 工作边界

- 不覆盖或清理用户已有改动；发现重叠时先核对差异。
- `third_party/` 只读：不修改、不在其中构建、不 `git pull`；升级版本只改 `repos.lock` 且需用户同意。
- 笔记只引用在本地克隆中实际看到的路径、函数和环境变量；未核实的写"待确认 + 确认方式"。不大段复制外部源码或书籍原文。
- 不把凭证、公司内部信息写入仓库。不自动 commit、push 或发布。

## Spec、Plan 与 PROGRESS

- Spec（`docs/superpowers/specs/`）说明仓库或某个学习专题为什么做、做什么、怎样验收。
- Plan（`docs/superpowers/plans/`）说明按什么顺序产出。
- `docs/PROGRESS.md` 是唯一当前状态；学习进度、本周焦点只写在这里。`notes/00-roadmap/02-jd-breakdown.md` 的掌握程度自评是知识地图，不是进度日志。

## 验证与完成

- 改笔记/文档：`python3 scripts/check_repo.py docs` 通过。
- 改 `labs/`：`python3 scripts/check_repo.py labs` 通过，并手动跑一次 `python3 -m labs.ccl_sim`。
- 完成一个学习切片：完整 `python3 scripts/check_repo.py` 通过，更新 `docs/PROGRESS.md` 的证据表；未运行项写 `not-run` 及原因。
- `docs` 检查只证明链接与格式，不证明技术内容正确；内容正确性靠对照源码和自测题。

## 文档维护路由

| 变化 | 必须检查或更新 |
|---|---|
| 新增/移动笔记章节 | `notes/00-roadmap/README.md`、`notes/00-roadmap/00-start-here.md` 第 6 节、对应章节 `README.md`、`docs/architecture/overview.md` |
| 新的外部调研 | `docs/references/YYYY-MM-DD-<topic>.md`（带来源、访问日期、置信度），学习笔记只引用其编号 |
| 外部资料新增或升级 | `third_party/repos.lock`、`docs/references/third-party.md`、引用该版本的笔记头部 |
| 执行入口或验证方式 | 本文件执行能力表、`docs/TESTING.md`、`README.md` |
| 新术语 | `docs/CONTEXT.md` |
| 学习状态、阻塞或下一步 | `docs/PROGRESS.md` |
| 可观察的仓库能力变化 | `CHANGELOG.md` |
