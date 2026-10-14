# 可靠性

## 黄金路径

`bash scripts/fetch_third_party.sh` → `python3 scripts/check_repo.py` 全部 `passed` → 按 `notes/00-roadmap/README.md` 阅读。

## 健康与可观测信号

| 信号 | 来源 | 正常范围 | 异常含义 |
|---|---|---|---|
| `OK <name> <commit>` | `scripts/fetch_third_party.sh check` | 9 行全部 OK | `MISSING` 未拉取；`MISMATCH` 外部仓库被切换或更新 |
| `docs: passed` | `scripts/check_repo.py docs` | 0 problems | 断链、占位残留或源码路径失效 |
| `labs: passed` | `scripts/check_repo.py labs` | unittest 全部 OK | 模拟器或代价模型回归 |

## 失败模式与恢复

| 失败模式 | 检测 | 安全恢复 | 验证 |
|---|---|---|---|
| 外部仓库目录被误删 | `MISSING` | `bash scripts/fetch_third_party.sh` | check 全部 OK |
| 外部仓库被 `git pull` 或切换 | `MISMATCH` | `git -C third_party/<name> fetch --depth 1 origin <lock commit> && git -C third_party/<name> checkout FETCH_HEAD` | check 全部 OK |

## 容量与限制

不适用：无运行服务。`third_party/` 浅克隆合计约 3.8 GB（2026-10-14 `du -sh third_party` 实测），新机器 setup 前确认磁盘空间。
