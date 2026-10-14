# 集合通信学习仓库建设设计规格

- 状态：Approved（用户 2026-10-14 在会话中提出需求）
- 日期：2026-10-14
- 关联需求：用户请求"创建学习通信的仓库，clone ai-infra-book 作学习资料，补充 RDMA 与推理框架，面向字节跳动集合通信库研发岗位"

## 问题与现有证据

- 目标目录 `ai-learning/` 为空，没有任何学习记录与治理结构（Agent-Native 基线 0 分）。
- 岗位 JD 要求覆盖：CCL 设计优化、RDMA 与 scale-up、通算融合（Triton）、AllReduce/All2All 性能、Ring/Tree/Halving-Doubling、NCCL 原理、性能调优与排障。
- 用户明确：通信库要理解上下层调用，因此除 ai-infra-book 外还需 RDMA 与推理框架资料。

## 目标

- 新会话的 Agent 或作者本人不依赖聊天记录，就能找到学习路线、按层次阅读笔记、对照固定版本源码、运行实验并知道下一步。
- JD 每条要求都映射到具体笔记目录与实践。

## 范围

- 固定版本拉取 9 个外部仓库（`third_party/repos.lock`）。
- `notes/00~08` 九个章节的首批笔记。
- `labs/ccl_sim` 纯 Python 算法模拟器与 α-β 代价模型及其单元测试。
- setup / run / verify 入口与 Agent-Native 治理文档。

## 非目标

- 不在本机构建或运行 NCCL、GPU/RDMA 实验。
- 不克隆 PyTorch（体积大；相关内容标待确认）。
- 不创建远程仓库、不 commit/push（需用户另行授权）。

## 用户与场景

- 作者：按路线学习，每完成一个专题更新 `docs/PROGRESS.md`。
- Agent：被要求"写某专题笔记"时，按 `notes/AGENTS.md` 格式、引用真实路径，并跑 `scripts/check_repo.py`。

## 约束与不变量

- `third_party/` 只读，版本唯一来源 `repos.lock`。
- 笔记只引用本地实际存在的路径；未核实写"待确认"。
- 当前状态只在 `docs/PROGRESS.md`。

## 方案

### 架构与流程

见 `docs/architecture/overview.md` 的分层拓扑：下层（01 RDMA、02 scale-up）→ 中层（03 算法、04 NCCL）→ 上层（05 框架、06 通算融合），贯穿 07 性能排查、00 路线、08 面试。

### 接口与数据

- `third_party/repos.lock`：`<目录名> <URL> <commit>`，`#` 开头为注释。
- `scripts/check_repo.py [docs|labs|third-party|all]`：0 通过、1 失败、2 参数错误。

### 错误、安全、隐私与可靠性

见 `docs/operations/security.md` 与 `docs/operations/troubleshooting.md`。

## 替代方案

- git submodule 管理外部资料：可固定版本，但对 3.8 GB 的上游仓库要求完整历史或额外配置，且会把 submodule 指针变更混入学习提交；选用 lock 文件 + 浅克隆脚本。
- 按资料来源（书/NCCL/vLLM）组织笔记：不利于理解调用链；选用按通信层次组织。

## 验收标准

- [ ] `bash scripts/fetch_third_party.sh check` 9 项全部 OK，退出码 0。
- [ ] `python3 scripts/check_repo.py` 输出 docs/labs/third-party 均 passed，退出码 0。
- [ ] `python3 -m labs.ccl_sim` 打印算法对比表，退出码 0。
- [ ] `notes/00-roadmap/README.md` 链接到 01~08 每个章节，且每个章节至少一篇导读。
- [ ] `notes/00-roadmap/02-jd-breakdown.md` 覆盖 JD 全部职责与要求条目。
- [ ] 非法参数时 `check_repo.py` 和 `fetch_third_party.sh` 退出 2。

## 测试、发布与文档

验证按 `docs/TESTING.md`；不发布。

## 风险与未决问题

- 笔记由多个 Agent 并行生成，技术细节可能有误；关闭方式：作者阅读时对照源码抽查，修正后在 `docs/PROGRESS.md` 记录。
- JD 中"scale up 等互联技术""自研 RDMA 协议"的具体所指未知，见 `docs/CONTEXT.md` 待澄清表。

## 参考资料

- `docs/references/third-party.md`（各仓库固定 commit）
