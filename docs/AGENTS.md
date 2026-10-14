# 文档工作协议

## 权威顺序

发生冲突时，按以下顺序核实：当前代码与可复现验证、当前配置与契约、当前正式文档、已批准 Spec、Plan、历史材料和聊天记录。

## 文档路由

| 问题 | 权威位置 | 何时更新 |
|---|---|---|
| 产品用途和用户路径 | `../README.md` | 用户、安装、配置或支持范围变化 |
| 领域术语和数据口径 | `CONTEXT.md` | 术语、状态或统计口径变化 |
| 当前架构和依赖方向 | `architecture/overview.md` | 模块、数据流、部署或信任边界变化 |
| 测试方法和证据边界 | `TESTING.md` | 测试、CI、fixture 或外部依赖变化 |
| 当前状态 | `PROGRESS.md` | 开始、暂停、恢复、完成、阻塞或验证变化 |
| 设计目标与验收 | `superpowers/specs/` | shaping 阶段设计发生变化 |
| 实现步骤 | `superpowers/plans/` | 实施顺序、文件或验证步骤变化 |
| 运行知识 | `operations/` | 部署、恢复、安全或故障事实变化 |
| 外部证据 | `references/`（外部资料总表：`references/third-party.md`；版本以 `../third_party/repos.lock` 为准） | 来源版本、证据或借鉴取舍变化 |
| 学习笔记正文 | `../notes/`（入口 `../notes/00-roadmap/README.md`，格式见 `../notes/AGENTS.md`） | 学习新专题；docs/ 不存放笔记正文 |
| 版本详情 | `changelog/` | 形成发布候选或正式版本 |
| 可复用模板 | Spec、Plan、reference 与 version 的标准模板文件 | 模板字段、职责或格式变化 |

## 维护矩阵

| 文档 | 必须回答 | 禁止内容 |
|---|---|---|
| `CONTEXT.md` | 术语定义、区别、代码或契约依据、歧义 | 实现流水账、当前任务 |
| `TESTING.md` | 层级、真实命令、隔离、CI、证据边界、缺口 | 编造结果、把静态检查当行为验证 |
| `PROGRESS.md` | 当前基线、证据、阻塞、风险和下一步 | 会话日志、另一份长期任务清单 |
| `architecture/overview.md` | 当前拓扑、职责、依赖、数据流、事实源、信任边界 | 只画图、不对应代码；混入短期计划 |
| Spec | 问题、目标、非目标、约束、方案和验收 | 实时进度、无依据的实现细节 |
| Plan | 关联 Spec、实施顺序、文件、验证和回滚 | 改写设计目标、积累会话状态 |
| Operations | 真实部署、恢复、安全和排障步骤 | 无证据的 SLO、端口、容量和环境 |
| 版本文档 | 用户影响、迁移、验证、限制和回滚 | commit 倾倒、计划中的变化 |

## 生命周期与去重规则

- 每项事实只有一个权威位置，其他文档只保留摘要和链接。
- 当前文档只描述当前事实；历史 Spec/Plan 使用 `Superseded` 或 `Archived` 明示。
- `YYYY-MM-DD-feature-design.md`、`YYYY-MM-DD-feature.md`、`reference.md` 与 `version.md` 只提供可复制结构，不代表当前事实；具体文档必须改为真实文件名并删除“模板状态”。
- 新增、移动、合并或删除文档时同步本文件和所有入站链接。
- 除上述四个可复用模板外，模板填写提示必须在文档交付前删除；未知事实使用“待确认 + 确认方式”。
- 每次文档改动后运行 `python3 scripts/check_repo.py docs`，它检查断链、占位残留和失效的 `third_party/...` 源码路径。
