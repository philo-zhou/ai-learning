# notes/ Agent 规则

> 先遵循仓库根 `AGENTS.md`。本文只补充学习笔记的局部约定。

## 模块职责

- 负责：按"下层 RDMA / scale-up → 中层集合通信库 → 上层训练推理框架"组织的学习笔记、源码导读、面试题。
- 不负责：外部源码本身（在 `third_party/`，只读）；可运行实验代码（在 `labs/`）；学习进度（在 `docs/PROGRESS.md`）。
- 对外接口：`notes/00-roadmap/README.md` 是笔记总入口，其他章节都从这里链接。

## 目录与入口

| 目录 | 层次 | 主题 |
|---|---|---|
| `00-roadmap/` | 总览 | JD 拆解、学习路线、ai-infra-book 阅读地图 |
| `01-rdma/` | 下层 scale-out | Verbs、QP、RoCEv2/IB、拥塞控制、perftest |
| `02-scale-up/` | 下层 scale-up | NVLink/NVSwitch、P2P/SHM、NVLS、IPC |
| `03-ccl-algorithms/` | 中层 | Ring/Tree/Halving-Doubling 等算法与 α-β 代价模型 |
| `04-nccl-internals/` | 中层 | NCCL 初始化、拓扑、channel、协议、transport、proxy |
| `05-framework-integration/` | 上层 | PyTorch/vLLM/SGLang/DeepEP 中的通信调用链 |
| `06-comm-compute-fusion/` | 跨层 | 通算融合、Triton-distributed、NVSHMEM |
| `07-perf-troubleshooting/` | 实践 | nccl-tests、环境变量、性能分析与故障排查 |
| `08-interview/` | 求职 | 面试题与项目表述 |
| `09-ep-kernels/` | 中层 | EP 专用通信库：DeepEP V2.5 direct / hybrid 源码精读，谱系对照 V1、Hybrid-EP、NCCL EP |

每个目录有一个 `README.md` 作为章节索引，具体笔记用 `NN-<topic>.md` 命名。

## 依赖、数据与外部边界

- 源码引用格式：`third_party/<repo>/<path>`，并在笔记头部写明所引用仓库的 commit（见 `docs/references/third-party.md`）。
- 只能引用在本地克隆中实际看到的路径、函数名、环境变量；没读到的写 `待确认：<内容>；确认方式：<文件/命令>`。
- 不大段复制外部源码或书籍原文；摘要 + 路径 + 自己的理解。
- 机械检查：`python3 scripts/check_repo.py`（链接有效、无模板占位符）。

## 模块约定

每篇笔记使用以下结构（不适用的节删除，不留空标题）：

```markdown
# 标题

> 层次：下层/中层/上层 · 来源：third_party/<repo>@<commit>、ai-infra-book 第 X 章

## 一句话
## 在调用链中的位置（上层谁调用它 / 它调用下层什么）
## 核心概念
## 源码导读（路径 + 函数 + 作用）
## 性能要点与常见坑
## 面试自测（问题 + 要点答案）
## 待深入
```

## 命令与验证

| 目的 | 命令 | 成功信号 | 不能证明什么 |
|---|---|---|---|
| 链接与占位符检查 | `python3 scripts/check_repo.py` | 退出码 0 | 笔记内容在技术上正确 |
