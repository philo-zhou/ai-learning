# 集合通信（CCL）学习仓库

面向"集合通信库研发"岗位的个人学习仓库：沿 **RDMA / scale-up 互联 → NCCL 集合通信库 → 训练与推理框架** 的调用链，记录源码导读、算法推导、可运行实验和面试准备。

## 项目解决什么问题

- 目标用户：仓库作者本人（以及读到它的 AI Agent），目标岗位为字节跳动基础设施网络团队的集合通信库研发。
- 核心场景：
  - 弄清通信库的上下层：上层框架（vLLM、SGLang、DeepEP、PyTorch）怎样调用通信，下层（RDMA Verbs、NVLink/NVSwitch）怎样承载通信；
  - 掌握 Ring / Tree / Halving-Doubling 等算法及 α-β 代价模型，并用纯 Python 模拟器验证；
  - 读懂 NCCL 的初始化、拓扑、channel、协议、transport 与 proxy；
  - 了解通算融合（Triton-distributed、NVSHMEM）与性能排查方法。
- 当前边界：本机为 macOS、无 GPU/RDMA 网卡，nccl-tests、perftest 和各框架示例只能阅读、不能运行；外部源码只读引用，不在本仓库修改。

## 快速开始

以下命令都在仓库根目录执行（`labs` 以包形式导入，在子目录运行会报 `No module named 'labs'`）。

### 前置条件

- `git`，可访问 `github.com`（拉取外部资料）
- Python 3（实验与检查只用标准库；已在 Python 3 + macOS 上运行）

### 安装与初始化

```bash
bash scripts/fetch_third_party.sh
```

按 `third_party/repos.lock` 浅克隆 9 个外部仓库并检出固定 commit；已存在的目录只核对、不覆盖。

### 启动

```bash
python3 -m labs.ccl_sim
```

打印 Ring、Tree、Recursive Doubling、Halving-Doubling 等算法在不同消息大小下的 α-β 耗时估算表。

### 验证第一次成功

```bash
python3 scripts/check_repo.py
```

依次输出 `docs: passed`、`labs: passed`、`third-party: passed`，退出码 0。

## 使用方式

1. 从 [学习入口](notes/00-roadmap/00-start-here.md) 开始：从大模型推理出发，讲清并行策略、对应通信和 EP 通信方案，再按其中的主线顺序和 [学习路线](notes/00-roadmap/03-learning-path.md) 推进。
2. 读某章节时对照 `third_party/` 中的源码路径；每篇笔记末尾有面试自测题。
3. 动手：改 `labs/ccl_sim/` 中的算法或代价模型参数，再跑测试验证。
4. 当前学习进度与下一步记录在 [docs/PROGRESS.md](docs/PROGRESS.md)。

## 系统概览

| 目录 | 内容 |
|---|---|
| `notes/` | 按层组织的学习笔记（00 路线 · 01 RDMA · 02 scale-up · 03 算法 · 04 NCCL 内部 · 05 框架集成 · 06 通算融合 · 07 性能排查 · 08 面试） |
| `labs/` | 可运行实验：集合通信算法模拟器与 α-β 代价模型 |
| `third_party/` | 固定版本的外部资料（不纳入本仓库 Git） |
| `scripts/` | setup 与 verify 入口 |
| `docs/` | 仓库治理：进度、术语、架构、测试、设计与计划 |

详见 [架构概览](docs/architecture/overview.md)。

## 配置

无运行时配置和环境变量。外部资料版本只由 `third_party/repos.lock` 决定。

## 文档

- [学习入口（从这里开始）](notes/00-roadmap/00-start-here.md)
- [笔记总入口](notes/00-roadmap/README.md)
- [领域术语](docs/CONTEXT.md)
- [架构概览](docs/architecture/overview.md)
- [测试说明](docs/TESTING.md)
- [当前进度](docs/PROGRESS.md)
- [外部资料说明](docs/references/third-party.md)

## 安全与支持

个人学习仓库，不提供支持承诺。不存放凭证或公司内部信息。笔记内容是学习理解，可能有误，以上游源码为准。

## License

待确认：仓库自身的 License；确认方式：作者决定是否公开及采用的许可证。`third_party/` 下各仓库遵循其各自 License。
