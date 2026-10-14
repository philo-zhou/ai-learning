# 集合通信学习仓库建设实现计划

- 关联 Spec：`docs/superpowers/specs/2026-10-14-ccl-learning-repo-design.md`
- 日期：2026-10-14
- 状态：Completed（2026-10-14；结果见 `docs/PROGRESS.md`）

## 目标与执行摘要

先拉取并锁定资料，再搭治理骨架与笔记格式约定，然后按章节并行生成笔记与实验，最后统一验证。

## 前置条件与全局约束

- git、Python 3、可访问 github.com。
- 笔记格式与引用规则以 `notes/AGENTS.md` 为准。

## 排序理由

笔记要引用真实源码路径，因此资料拉取必须最先；格式约定必须在并行写作前确定，否则各章节风格与引用方式不一致。

## Milestones

### Milestone 1：外部资料与 setup

- Create：`third_party/repos.lock`、`scripts/fetch_third_party.sh`
- 命令：`bash scripts/fetch_third_party.sh check`
- 预期：9 行 OK，退出码 0；`bash scripts/fetch_third_party.sh bogus` 退出 2。

### Milestone 2：治理骨架与笔记约定

- Create：`AGENTS.md`、`CLAUDE.md`/`CODEBUDDY.md` 软链接、`notes/AGENTS.md`、`third_party/AGENTS.md`、`docs/` 全部文档、`scripts/check_repo.py`
- 命令：`python3 scripts/check_repo.py bogus`
- 预期：退出 2。

### Milestone 3：并行生成笔记与实验

- Create：`notes/00-roadmap/`、`notes/08-interview/`（路线与面试）；`notes/01-rdma/`；`notes/03-ccl-algorithms/` 与 `labs/`；`notes/02-scale-up/`、`notes/04-nccl-internals/`、`notes/07-perf-troubleshooting/`；`notes/05-framework-integration/`；`notes/06-comm-compute-fusion/`
- Test：`python3 -m unittest discover -s labs -t .`
- 预期：全部 OK。

### Milestone 4：统一验证与状态记录

- Modify：修复 `check_repo.py docs` 报告的断链与失效源码路径；更新 `docs/PROGRESS.md`、`CHANGELOG.md`
- 命令：`python3 scripts/check_repo.py`
- 预期：三步全部 passed，退出码 0。

## 迁移与回滚

不适用：新仓库，无迁移。回滚即删除新建文件。

## 风险、并行与变更切片

Milestone 3 的六个切片写入互不重叠的目录，可并行；跨章节链接只指向各章节 `README.md`，由 Milestone 4 统一校验。

## 最终验证

`python3 scripts/check_repo.py` 与 `python3 -m labs.ccl_sim` 退出码均为 0。

## 文档同步

`docs/PROGRESS.md`、`CHANGELOG.md`、`docs/architecture/overview.md`（如章节结构有变）。
