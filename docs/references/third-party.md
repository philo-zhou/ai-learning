# 外部学习资料（third_party）

- 来源：见下表各上游组织
- URL：见 `third_party/repos.lock`
- 固定版本或 commit：以 `third_party/repos.lock` 为唯一权威
- 访问时间：2026-10-14
- License：各仓库自带 License，本仓库只做摘要和路径引用

## 它是什么

| 仓库 | 层次 | 短 commit | 为什么选它 |
|---|---|---|---|
| `ai-infra-book`（bojieli） | 总览 | `3bdcb4fc` | 用户指定的主教材，覆盖 AI 基础设施全栈 |
| `rdma-core`（linux-rdma） | 下层 | `bd3282a1` | libibverbs/librdmacm 用户态实现与示例，NCCL IB transport 的直接下层 |
| `perftest`（linux-rdma） | 下层 | `05200c2f` | RDMA 带宽/延迟基准，排查网络问题的第一工具 |
| `nccl`（NVIDIA） | 中层 | `3e46c089` | JD 点名的集合通信库，学习实现原理的主对象 |
| `nccl-tests`（NVIDIA） | 中层 | `afd59aba` | 集合通信性能基准，定义 algbw/busbw 口径 |
| `vllm` | 上层 | `02653fa2` | 主流推理框架，含 pynccl、custom allreduce、EP all2all |
| `sglang` | 上层 | `d2a43644` | 另一主流推理框架，用于对比通信层设计 |
| `DeepEP`（deepseek-ai） | 上层/跨层 | `93eb6eb2` | MoE All2All 专用通信库，代表 device-initiated 通信趋势 |
| `Triton-distributed`（ByteDance-Seed） | 跨层 | `7908e4ea` | 基于 Triton 的通算融合，对应 JD "利用 Triton 开发通信计算融合算子" |

## 为什么与本项目有关

岗位要求理解通信库的上下层：上层框架如何调用、下层 RDMA 与 scale-up 互联如何承载。上述仓库分别覆盖这条链路的每一层。

## 关键证据

- 各仓库具体的源码导读与路径见 `notes/` 对应章节，笔记头部注明引用的 commit。

## 我们借鉴什么

- 源码结构、接口、环境变量和基准方法，作为笔记的事实来源。

## 我们明确不借鉴什么

- 不复制大段源码或书籍原文；不在本仓库修改或构建上游代码。
- PyTorch 源码未克隆，torch.distributed 相关内部实现在笔记中标"待确认"。

## 影响的本地决策

- `third_party/AGENTS.md` 的只读规则；`notes/AGENTS.md` 的引用格式。

## 新鲜度与更新触发器

- 上游大版本发布（例如 NCCL 新增算法/transport）或笔记引用的路径失效时，经用户同意后更新 `repos.lock`，重新运行 `bash scripts/fetch_third_party.sh` 并复核受影响笔记。
