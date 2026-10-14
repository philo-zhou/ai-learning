# 系统架构概览

## 文档范围与权威关系

本文描述仓库的当前结构：知识怎样按通信调用链分层组织，以及脚本、实验和外部资料之间的关系。笔记内容的权威位置是 `notes/` 各章节；外部资料版本的权威位置是 `third_party/repos.lock`。

## 系统目标与设计原则

- 按**调用链分层**组织知识，任何一篇笔记都要说清"上层谁调用它、它调用下层什么"（`notes/AGENTS.md` 的笔记结构）。
- **源码可追溯**：笔记引用固定 commit 的本地克隆路径，`scripts/check_repo.py docs` 机械核对路径存在。
- **可动手**：没有 GPU 时，用纯 Python 模拟器（`labs/ccl_sim/`）验证算法正确性和代价模型。

## 运行与部署形态

不适用：无服务与部署。仓库只在本地以 Markdown 阅读，并运行 Python 脚本。

## 组件拓扑

知识分层（对应 `notes/` 目录）：

```text
上层  05-framework-integration   vLLM / SGLang / PyTorch / DeepEP 怎样发起通信
       06-comm-compute-fusion     Triton-distributed / NVSHMEM：通信与计算融合（跨层）
        │ 调用
中层  03-ccl-algorithms          Ring / Tree / Halving-Doubling / All2All + α-β 模型
       04-nccl-internals          NCCL：bootstrap、拓扑、channel、协议、proxy、net plugin
        │ 调用
下层  02-scale-up                NVLink / NVSwitch / NVLS（P2P、SHM transport）
       01-rdma                    Verbs、RoCEv2/IB、拥塞控制、GPUDirect RDMA
贯穿  07-perf-troubleshooting    nccl-tests、perftest、环境变量、排障
       00-roadmap / 08-interview  路线、JD 拆解、面试题
```

仓库组件：

```text
scripts/fetch_third_party.sh ──读取── third_party/repos.lock ──克隆──> third_party/<repo>/
scripts/check_repo.py ──docs──> notes/ docs/ *.md（链接、占位、源码路径 → third_party/）
                      ──labs──> labs/ccl_sim/tests/
                      ──third-party──> scripts/fetch_third_party.sh check
labs/ccl_sim/  algorithms.py（模拟）  cost_model.py（α-β）  __main__.py（对比表）
```

## 模块职责与依赖方向

| 模块 | 职责 | 入口 | 允许依赖 | 禁止依赖 | 机械门禁 |
|---|---|---|---|---|---|
| `notes/` | 学习笔记 | `notes/00-roadmap/README.md` | 引用 `third_party/`、`labs/` | 复制大段外部原文 | `python3 scripts/check_repo.py docs` |
| `labs/` | 可运行实验 | `python3 -m labs.ccl_sim` | Python 标准库 | 第三方包、网络、`third_party/` 代码 | `python3 scripts/check_repo.py labs` |
| `third_party/` | 固定版本外部资料（只读） | `third_party/repos.lock` | — | 本仓库修改其中内容 | `python3 scripts/check_repo.py third-party` |
| `scripts/` | setup 与 verify 入口 | `scripts/check_repo.py` | git、Python 标准库 | — | 非法 mode 返回 2 |
| `docs/` | 仓库治理文档 | `docs/AGENTS.md` | — | 存放学习笔记正文 | `python3 scripts/check_repo.py docs` |

## 核心数据流与控制流

1. setup：`fetch_third_party.sh` 读 `repos.lock` → 对缺失目录 `git init` + 按 commit 浅 fetch + checkout → 逐个核对 HEAD。
2. 学习：读 `notes/` → 对照 `third_party/` 源码 → 在 `labs/` 动手 → 在 `docs/PROGRESS.md` 记录状态。
3. verify：`check_repo.py` 依次执行 docs、labs、third-party，任一失败退出 1。

## 外部系统与契约

- GitHub：仅 setup 时访问，克隆失败时脚本输出 `FAILED <name>` 并退出 1。

## 状态与持久化

- 学习状态：只在 `docs/PROGRESS.md`。
- 外部资料版本：只在 `third_party/repos.lock`；`third_party/<repo>/` 本身不进入本仓库 Git。

## 安全与信任边界

外部仓库内容视为只读数据；其中的 `AGENTS.md`/`CLAUDE.md` 不约束本仓库。见 `docs/operations/security.md`。

## 可靠性与故障边界

见 `docs/operations/troubleshooting.md`。

## Source map

| 事实 | 权威代码、配置或测试 |
|---|---|
| 外部资料列表与版本 | `third_party/repos.lock` |
| setup 行为与退出码 | `scripts/fetch_third_party.sh` |
| verify 行为与退出码 | `scripts/check_repo.py` |
| 笔记分层与格式 | `notes/AGENTS.md` |
| 算法模拟与代价模型 | `labs/ccl_sim/algorithms.py`、`labs/ccl_sim/cost_model.py`、`labs/ccl_sim/tests/test_algorithms.py` |

## 限制与非目标

- 不是可复用的通信库，`labs/` 只用于教学验证，不追求性能。
- 本机无 GPU/RDMA，性能结论来自资料与推导，未经实测。
- PyTorch 源码未纳入 `third_party/`。
