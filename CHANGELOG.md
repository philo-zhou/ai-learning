# Changelog

本文记录用户、集成方或维护者可以观察到的变化，按时间倒序维护。仓库没有正式版本，所有变化记在 `[Unreleased]`。

## [Unreleased]

### Changed

- **学习站点手机端** — ASCII 图行距收紧到 1.25，竖线与方框角不再断开；屏宽 ≤ 640px 时图的字号降到 11px、表头允许换行（iPhone 13 尺寸下 55 页均无整页横向溢出）。

### Added

- **RDMA 建链与端到端发包** — `notes/01-rdma/01-verbs-basics.md` 新增 RDMA CM 建链一节（REQ/REP/RTU 时序、CM 事件 / CM 消息 / QP 状态三套状态的对照、RTU 与 RTR/RTS 的区别、与手工建链对比），依据 `third_party/rdma-core/librdmacm/cma.c`；`notes/01-rdma/02-transport-and-ops.md` 新增一次 RDMA WRITE_WITH_IMM 的端到端流程（MR/lkey/rkey、WQE 各段布局、doorbell、组包与 PSN、rkey 校验、CQE owner 位、出错表），依据 mlx5 provider 源码。
- **RoCE QoS 配置** — `notes/01-rdma/03-roce-ib-congestion.md` 新增两节：PFC 帧、CNP、RoCEv2 数据、TCP 的优先级顺序与队列关系，以及网卡侧配置命令（`mlnx_qos`、ECN sysfs、`pfc-prevention-tout`、iproute2 `dcb`、计数器验证）与常见取值；配套调研 `docs/references/2026-10-17-roce-qos-config.md`（12 条结论，逐条标注一手来源与置信度）；术语表新增 PFC/ECN/CNP/ETS。
- **笔记配图** — 全部 49 篇笔记新增约 110 张 ASCII 示意图（原语前后状态、Ring/Tree/Halving-Doubling 逐步图、QP 状态机、DCQCN/PFC 反馈环、GDR 数据路径、NCCL init/bootstrap 时序、LL/LL128/Simple 布局、proxy 握手、vLLM 后端选择树、TBO 时序、DeepEP 缓冲区布局、通算融合的重叠层次时间线、AG+GEMM 信号与 swizzle、GEMM+RS 计数器判定、对称堆布局、GIN 强/弱信号等），图中涉及的源码细节对照 `third_party/` 本地克隆核实。
- **PD 分离 + AF 分离全流程** — `notes/05-framework-integration/06-pd-af-end-to-end.md`：一次请求从 prefill、KV 传输到 AF 分离 decode 的逐层流程图，每层 QKV 与 KV cache 的计算/复用、MoE dispatch/combine 的含义与使用场景、AF 分离为何只用于 MoE 的 decode、prefill 与 decode 的 token 用量对照表、通信清单；配套调研 `docs/references/2026-10-17-megascale-infer-phase.md`，以及 vLLM KVConnector、SGLang disaggregation 的源码位置。
- **Ring / Full-mesh 手写题** — `notes/03-ccl-algorithms/05-handwrite-ring-fullmesh.md`：Ring 与 Full-mesh 各原语的伪代码、GPU one-shot / two-shot kernel 与追问；`labs/ccl_sim` 新增 `fullmesh_*` 模拟与代价模型（CLI 表格新增 `fm_oneshot`、`fm_twoshot` 两列）及 `TestFullMesh` 测试。
- **学习站点** — `site/`：VitePress 直接渲染 `notes/`、`docs/references/` 与术语表，技术手册风格主题（暖白/深墨蓝、三层知识色标）、首页路线卡片、中文全文搜索、公式渲染；`cd site && npm install && npm run dev` 启动，`npm run build` 构建并自检；`.github/workflows/deploy-site.yml` 把站点发布到 GitHub Pages（https://philo-zhou.github.io/ai-learning/）。`scripts/check_repo.py` 改为跳过 `site/node_modules` 与构建产物。
- **EP 专用通信库专章** — `notes/09-ep-kernels/`：EP All2All 设计空间、DeepEP low-latency（固定槽位、同 QP 计数、IBGDA、recv hook、LogFMT）、Hybrid-EP（路由预计算、分层 RDMA/NVLink、TMA warp 专业化、节点内预归约）、谱系与选型；外部资料新增 `DeepEP-hybrid-ep`（上游 `hybrid-ep` 分支 @10d4dd73）。
- **学习入口** — `notes/00-roadmap/00-start-here.md`：从大模型推理出发，梳理并行策略与通信的对应总表、EP dispatch/combine 的数据量与难点、主流 EP 方案全景（DeepEP V1/V2、NCCL EP、pplx、UCCL-EP、MegaScale-Infer 等）、发展趋势，以及把各章挂到主线上的学习顺序；README 与笔记总入口都指向它。
- **推理并行与 EP 通信调研报告** — `docs/references/2026-10-14-inference-parallelism-ep-research.md`：17 个来源，13 条结论经 3 票对抗验证并标注置信度，另有 1 条被推翻的结论和未决问题清单。
- **外部学习资料** — `third_party/repos.lock` 固定 ai-infra-book、nccl、nccl-tests、rdma-core、perftest、vllm、sglang、DeepEP、Triton-distributed 九个仓库版本；`bash scripts/fetch_third_party.sh` 拉取，`check` 模式只读核对。
- **分层学习笔记** — `notes/00~08`：路线与 JD 拆解、RDMA、scale-up、集合通信算法、NCCL 内部、框架集成、通算融合、性能排查、面试题库；入口 `notes/00-roadmap/README.md`。
- **集合通信模拟器** — `python3 -m labs.ccl_sim` 输出各算法 α-β 估算对比；`labs/ccl_sim/tests/` 验证正确性、步数和通信量。
- **统一验证入口** — `python3 scripts/check_repo.py [docs|labs|third-party|all]`，检查断链、模板占位、笔记引用的源码路径、实验测试和外部资料版本。
- **Agent 入口** — `AGENTS.md`（`CLAUDE.md`、`CODEBUDDY.md` 软链接）与 `docs/` 治理文档。
