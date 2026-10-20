# third_party/ Agent 规则

> 先遵循仓库根 `AGENTS.md`。本文只补充外部资料目录的局部约定。

## 模块职责

- 负责：存放按 `repos.lock` 固定版本的外部学习资料浅克隆，供笔记引用。
- 不负责：任何本仓库自己的内容；子目录不纳入本仓库 Git（见 `.gitignore` 的 `third_party/*/`）。
- 对外接口：`repos.lock`（名称、URL、commit），由 `scripts/fetch_third_party.sh` 读取。

## 目录与入口

| 目录 | 层次 | 用途 |
|---|---|---|
| `ai-infra-book/` | 总览 | 主学习教材 |
| `rdma-core/`、`perftest/` | 下层 | Verbs 用户态库与 RDMA 性能测试 |
| `nccl/`、`nccl-tests/` | 中层 | 集合通信库源码与基准 |
| `vllm/`、`sglang/` | 上层 | 推理框架的通信调用 |
| `DeepEP/` | 上层/跨层 | MoE EP All2All 通信库（只跟最新主线） |
| `DeepInfer/` | 上层 | DeepSeek 推理框架，`EPBuffer` 的生产用法 |
| `Triton-distributed/` | 跨层 | 基于 Triton 的通算融合 |

## 依赖、数据与外部边界

- 只读：不修改、不构建、不 `git pull`、不切换分支。上游各自的 `AGENTS.md`/`CLAUDE.md`（如 `vllm/AGENTS.md`）只约束上游开发，不适用于本仓库。
- 升级或新增资料：先征得用户同意，再改 `repos.lock`，运行 `bash scripts/fetch_third_party.sh`，同步 `../docs/references/third-party.md`。
- 外部资料各有 License，笔记只做摘要与路径引用。

## 模块约定

引用格式：`third_party/<repo>/<path>`；笔记头部写 `<repo>@<短 commit>`，且必须与 `repos.lock` 一致。

## 命令与验证

| 目的 | 命令 | 成功信号 | 不能证明什么 |
|---|---|---|---|
| 版本核对 | `bash scripts/fetch_third_party.sh check` | 全部 `OK`，退出码 0 | 笔记中的路径仍存在于该版本 |
| 补齐缺失资料 | `bash scripts/fetch_third_party.sh` | 全部 `OK`，退出码 0 | 上游已有更新 |
