# 测试说明

本仓库的"测试"分两类：对文档结构的机械检查，以及对 `labs/` 教学代码的单元测试。两者都不能证明笔记的技术内容正确。

## 测试层级

| 层级 | 解决的风险 | 测试位置 | 真实命令 | 前置条件 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|---|---|
| docs 检查 | 断链、遗留模板占位、可复用模板缺标记、笔记引用了不存在的源码路径 | `scripts/check_repo.py` | `python3 scripts/check_repo.py docs` | Python 3；源码路径检查需 setup 完成，否则跳过并提示 | 本仓库 Markdown 的相对链接都存在；无双花括号模板占位；行内代码中的 `third_party/...` 路径在锁定版本中存在 | 笔记技术内容正确；函数名、环境变量名是否存在（只查路径）；含通配符的路径 |
| unit | 算法模拟器结果错误、步数/通信量不符合理论 | `labs/ccl_sim/tests/` | `python3 scripts/check_repo.py labs` | Python 3 | 模拟的 AllReduce/All2All 与朴素实现一致；步数与通信量符合推导；非法输入报错 | 真实 GPU/网络上的性能；NCCL 的真实行为 |
| third-party 核对 | 外部资料缺失或版本漂移，导致笔记引用失效 | `scripts/fetch_third_party.sh` | `python3 scripts/check_repo.py third-party` | setup 已完成 | 每个外部仓库 HEAD 等于 `repos.lock` | 上游是否有新版本 |
| manual：源码对照 | 笔记把函数、路径、环境变量写错 | `notes/` | 在 `third_party/` 中 `grep -rn <符号>` | setup 已完成 | 被抽查的引用真实存在 | 未抽查部分 |
| 站点构建 | 站内死链、笔记页面缺失、公式未渲染、未渲染文件的链接未改写、中文分词失效 | `site/scripts/verify.mjs`（由 `npm run build` 调用） | `cd site && npm run build` | Node；已 `npm install` | VitePress 构建成功、无站内死链；每篇笔记都有对应页面；`00-start-here` 公式渲染；`labs/README.md` 链接被改写为 GitHub；tokenizer 产出中文双字词 | 视觉效果（需截图人工检查）；浏览器中的交互 |
| GPU/RDMA 实验 | 对性能结论只停留在纸面 | nccl-tests、perftest | 见 `notes/07-perf-troubleshooting/README.md` | GPU、RDMA 网卡 | 真实带宽/延迟 | — 本机默认 `not-run` |

## 变更与验证矩阵

| 变更类型 | 最小验证 | 模块验证 | 完整验证 | 升级条件 |
|---|---|---|---|---|
| 新增/修改笔记 | `python3 scripts/check_repo.py docs` | 抽查新引用的源码路径 | `python3 scripts/check_repo.py` | 移动或重命名章节时 |
| 修改 `labs/` | `python3 -m unittest discover -s labs -t .` | `python3 -m labs.ccl_sim` | `python3 scripts/check_repo.py` | 改动 cost model 公式时同步核对 `notes/03-ccl-algorithms/02-alpha-beta-model.md` |
| 修改 `repos.lock` | `python3 scripts/check_repo.py third-party` | 复核引用该仓库的笔记 | `python3 scripts/check_repo.py` | 任何版本升级 |
| 修改 `site/` 或新增章节 | `cd site && npm run build` | 浏览器打开首页与一篇笔记检查样式 | `cd site && npm run build` 且 `python3 scripts/check_repo.py` | 改 `content.mts` 白名单或主题 CSS 时截图核对 |
| 修改 `scripts/` | 对应 mode 的正常与非法参数各跑一次（非法 mode 必须退出 2） | — | `python3 scripts/check_repo.py` | — |

## Fixture、Mock 与隔离

`labs/ccl_sim` 用 Python list 模拟各 rank 的 buffer，随机数据使用固定种子；不访问网络和文件系统。`fetch_third_party.sh` 的克隆路径可在临时目录中复制脚本和单行 lock 测试，不触碰真实 `third_party/`。

## CI 门禁

不适用：仓库尚无远程与 CI。本地 `python3 scripts/check_repo.py` 即完整门禁。

## 外部、付费和生产边界

- `bash scripts/fetch_third_party.sh`（fetch 模式）会联网，只在 setup 或用户同意升级时运行。
- 需要 GPU/RDMA/云资源的实验默认 `not-run`，记录在 `docs/PROGRESS.md` 并写明所需硬件。

## 失败证据

统一使用 `passed / failed / partial / blocked / not-run / skipped / not-applicable`，在 `docs/PROGRESS.md` 记录命令、环境、退出码和关键输出。

## 已知缺口

- 笔记中的函数名、环境变量名不被自动检查，只检查路径；关闭方式：手动 `grep -rn` 抽查，或给 `check_repo.py` 增加符号检查。
- 无 GPU/RDMA 环境，性能类结论缺少实测；关闭方式：获得 GPU 机器后跑 nccl-tests 与 perftest，结果写入 `notes/07-perf-troubleshooting/`。
