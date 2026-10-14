# Changelog

本文记录用户、集成方或维护者可以观察到的变化，按时间倒序维护。仓库没有正式版本，所有变化记在 `[Unreleased]`。

## [Unreleased]

### Added

- **学习入口** — `notes/00-roadmap/00-start-here.md`：从大模型推理出发，梳理并行策略与通信的对应总表、EP dispatch/combine 的数据量与难点、主流 EP 方案全景（DeepEP V1/V2、NCCL EP、pplx、UCCL-EP、MegaScale-Infer 等）、发展趋势，以及把各章挂到主线上的学习顺序；README 与笔记总入口都指向它。
- **推理并行与 EP 通信调研报告** — `docs/references/2026-10-14-inference-parallelism-ep-research.md`：17 个来源，13 条结论经 3 票对抗验证并标注置信度，另有 1 条被推翻的结论和未决问题清单。
- **外部学习资料** — `third_party/repos.lock` 固定 ai-infra-book、nccl、nccl-tests、rdma-core、perftest、vllm、sglang、DeepEP、Triton-distributed 九个仓库版本；`bash scripts/fetch_third_party.sh` 拉取，`check` 模式只读核对。
- **分层学习笔记** — `notes/00~08`：路线与 JD 拆解、RDMA、scale-up、集合通信算法、NCCL 内部、框架集成、通算融合、性能排查、面试题库；入口 `notes/00-roadmap/README.md`。
- **集合通信模拟器** — `python3 -m labs.ccl_sim` 输出各算法 α-β 估算对比；`labs/ccl_sim/tests/` 验证正确性、步数和通信量。
- **统一验证入口** — `python3 scripts/check_repo.py [docs|labs|third-party|all]`，检查断链、模板占位、笔记引用的源码路径、实验测试和外部资料版本。
- **Agent 入口** — `AGENTS.md`（`CLAUDE.md`、`CODEBUDDY.md` 软链接）与 `docs/` 治理文档。
