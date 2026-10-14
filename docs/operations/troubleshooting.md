# 故障排查

## 先收集什么

`git status --short`、`python3 --version`、失败命令的完整输出与退出码、`bash scripts/fetch_third_party.sh check` 的输出。

## 症状与处理

### `fetch_third_party.sh` 输出 `FAILED <name>`

- 可能原因：无法访问 github.com；锁定 commit 已被上游强推删除；残留了半成品目录。
- 安全检查：`git ls-remote <url> | head -1`；`ls -la third_party/<name>`
- 修复动作：网络恢复后重跑 `bash scripts/fetch_third_party.sh`；若目录是失败残留（无有效 HEAD），经用户确认后删除该目录再重跑。
- 成功信号：`OK <name> <commit>`，退出码 0。
- 升级条件：commit 在上游不可获取——请用户决定新的锁定版本。

### `docs` 检查报 `源码路径不存在`

- 可能原因：笔记写错路径；路径在锁定版本中已改名；`third_party` 版本不一致。
- 安全检查：`bash scripts/fetch_third_party.sh check`；`find third_party/<repo> -name '<文件名>'`
- 修复动作：改正笔记中的路径；找不到时改写为"待确认 + 确认方式"。
- 成功信号：`python3 scripts/check_repo.py docs` 退出码 0。
- 升级条件：无。

### `labs` 测试失败

- 可能原因：修改了 `labs/ccl_sim/algorithms.py` 或 `cost_model.py` 导致回归。
- 安全检查：`python3 -m unittest discover -s labs -t . -v`
- 修复动作：对照 `notes/03-ccl-algorithms/02-alpha-beta-model.md` 的推导修正实现；不要放宽断言。
- 成功信号：unittest 全部 OK。
- 升级条件：推导本身可疑时，记录到 `docs/PROGRESS.md` 风险项。

## 已知限制

- 无法在本机复现任何 GPU/RDMA 层面的问题；相关排障知识见 `notes/07-perf-troubleshooting/README.md`。
