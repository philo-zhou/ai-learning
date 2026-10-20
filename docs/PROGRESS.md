# 当前进度

- 最后更新：2026-10-21 CST
- 当前分支：`main`
- 基线提交：本轮成果即首个提交（用 `git log -1` 查看）；远程 `origin` = https://github.com/philo-zhou/ai-learning（用户 2026-10-14 授权推送）
- 工作树状态：提交后为 clean；`third_party/*/` 被 `.gitignore` 排除

## 可核实基线

- 10 个外部资料按 `third_party/repos.lock` 固定版本拉取完毕（2026-10-20：DeepEP 升到最新主线 `b5a33eef`，移除 `DeepEP-hybrid-ep`，新增 `DeepInfer`），且均无本地改动 — 证据：`bash scripts/fetch_third_party.sh check` 全部 OK；逐个 `git -C third_party/<repo> status --porcelain` 为空。
- `notes/00~09` 十个章节共 50 篇笔记（2026-10-15 新增 `notes/09-ep-kernels/` 五篇；2026-10-17 新增 `05-framework-integration/06-pd-af-end-to-end.md`；2026-10-21 09 章改为 DeepEP V2.5 精读，共六篇）；学习入口 `notes/00-roadmap/00-start-here.md`，章节索引 `notes/00-roadmap/README.md`。
- 推理并行与 EP 通信调研：`docs/references/2026-10-14-inference-parallelism-ep-research.md`（17 个来源，13 条结论经过验证）。
- `labs/ccl_sim` 算法模拟器与 α-β 代价模型（含 Full-mesh），36 个单元测试。
- 治理入口：`AGENTS.md`（`CLAUDE.md`、`CODEBUDDY.md` 软链接）、`docs/`、`scripts/check_repo.py`、`scripts/fetch_third_party.sh`。

## 当前焦点

仓库建设已完成（Spec：`docs/superpowers/specs/2026-10-14-ccl-learning-repo-design.md`；Plan：`docs/superpowers/plans/2026-10-14-ccl-learning-repo.md`，状态 Completed）。下一焦点：通读 `notes/00-roadmap/00-start-here.md`，再按其第 7 节的主线顺序进行，前两步对应 `notes/00-roadmap/03-learning-path.md` 第 1 阶段（集合通信算法与代价模型）。

## 本轮完成并验证

- 2026-10-21：`notes/09-ep-kernels/` 改为 DeepEP V2.5 源码精读：新增 `02-deepep-v25-overview.md`（调用链、JIT、对称内存与 GIN、token 布局、workspace、`NCCLGin`、`gpu_barrier`、deferred epilogue、SM/QP 估算）、`03-deepep-v25-direct.md`、`04-deepep-v25-hybrid.md`；删除基于上游 `hybrid-ep` 分支的旧 02（V1 LL）、03（Hybrid-EP）；谱系篇改名为 `05-lineage-and-selection.md`；`01-design-space.md` 每个决定补 V2.5 做法。所有 V2.5 结论对照 `third_party/DeepEP@b5a33eef` 逐文件阅读，GIN 批量门铃与信号强弱对照 `third_party/nccl@3e46c089` 核实；kernel 行为未实测（not-run） — 证据：`python3 scripts/check_repo.py`、`cd site && npm run build`。
- 2026-10-20：DeepEP 只保留最新主线（`93eb6eb2` → `b5a33eef`，两者只差 lint 配置文件，代码相同），移除 `hybrid-ep` 分支克隆；新增 DeepSeek 推理框架 `DeepInfer@774ce244`。`notes/09-ep-kernels/04-lineage-and-selection.md`（2026-10-21 改名为 `05-lineage-and-selection.md`）新增"V1 与 Hybrid-EP 在 V2.5 中变成了什么"（对照 `dispatch.cuh`、`hybrid_dispatch.cuh` 与 DeepInfer 的 `EPBuffer` 用法），解决原"V2.5 slot 分配"待确认；09 章与其他笔记中对 `hybrid-ep` 分支的引用改为 `hybrid-ep@10d4dd73/<路径>` 历史标注 — 证据：`bash scripts/fetch_third_party.sh` 全部 OK；`python3 scripts/check_repo.py`；`cd site && npm run build`。
- 2026-10-17：`notes/01-rdma/01-verbs-basics.md` 补 RDMA CM 建链，`02-transport-and-ops.md` 补端到端 WRITE_WITH_IMM 流程；用户态部分对照 `librdmacm/cma.c`、`providers/mlx5/{qp.c,cq.c,mlx5dv.h}` 核实，网卡内部行为与 CM 消息格式按 IB 规范描述（内核 `ib_cm` 未克隆，列为待确认）— 证据：`python3 scripts/check_repo.py docs`、`cd site && npm run build`。
- 2026-10-17：`notes/01-rdma/03-roce-ib-congestion.md` 补充 RoCE 优先级与网卡 QoS 配置，解决原笔记中 TC→DSCP 映射、ECN sysfs 路径两项待确认；数值依据 `docs/references/2026-10-17-roce-qos-config.md`，网卡 CNP 默认值等仍待真机确认 — 证据：`python3 scripts/check_repo.py docs`。
- 2026-10-17：给 `notes/` 下 49 篇笔记补充 ASCII 示意图（按章节由并行 Agent 完成，只增不删；源码细节对照本地克隆核实）；`notes/06-comm-compute-fusion/` 5 篇（13 张图）在中断后补完，图中标识符（`symm_workspaces`、`rowise_ag_gemm_dispatcher`、GIN 强/弱信号等）已对照 `third_party/Triton-distributed`、`third_party/nccl` 核实— 证据：`python3 scripts/check_repo.py docs` 与 `cd site && npm run build` 均 passed；中文与右边框对齐用脚本按显示宽度扫描。
- 2026-10-17：新增 `notes/05-framework-integration/06-pd-af-end-to-end.md`（PD 分离 + AF 分离的一次完整推理），源码位置对照 `third_party/vllm@02653fa`、`third_party/sglang@d2a4364` 核实；含 MoE dispatch/combine 的含义与使用场景；AF 分离部分本地无开源实现，依据调研报告 F13 与新增的 `docs/references/2026-10-17-megascale-infer-phase.md`（论文原文确认 AF 分离只针对 decode），M2N 细节列为待确认 — 证据：`python3 scripts/check_repo.py docs`。
- 2026-10-15：Ring / Full-mesh 手写题笔记与 `labs/ccl_sim` 的 Full-mesh 实现 — 证据：`python3 scripts/check_repo.py`（labs 新增 6 个测试，共 36 个通过）；笔记中 Ring 伪代码的下标另用逐 rank 的锁步模拟在 N=1..9 验证；学习站点已上线 https://philo-zhou.github.io/ai-learning/（Pages 工作流第 2 次运行成功）。
- 2026-10-15：学习站点 `site/`（Spec：`docs/superpowers/specs/2026-10-15-learning-site-design.md`；Plan：`docs/superpowers/plans/2026-10-15-learning-site.md`，Completed） — 证据：`cd site && npm run build` 输出 `site verify: passed`；中文搜索用构建出的索引离线测试，"拥塞控制""IBGDA""busbw""自适应路由"均命中相关笔记；首页、`00-start-here`（公式）、暗色模式截图人工检查。
- 2026-10-15：新增 EP 专用通信库专章 `notes/09-ep-kernels/`（设计空间、DeepEP LL、Hybrid-EP、谱系与选型），依据 `third_party/DeepEP-hybrid-ep@10d4dd73` 源码撰写；同步更新 `05-framework-integration/04-deepep.md` 中原先的 V1 待确认项、学习入口、术语表 — 证据：`python3 scripts/check_repo.py`。
- 外部资料拉取与版本锁定 — 证据：`bash scripts/fetch_third_party.sh check`；克隆路径在临时目录用单行 lock 验证通过。
- 笔记首批覆盖全部章节，且笔记中引用的 `third_party/...` 路径均存在 — 证据：`python3 scripts/check_repo.py docs`。
- 模拟器正确性、步数与通信量 — 证据：`python3 scripts/check_repo.py labs`。

## 验证证据

| 状态 | 命令或检查 | 范围 | 结果摘要 |
|---|---|---|---|
| passed | `python3 scripts/check_repo.py` | docs + labs + third-party | 三步均 passed，退出码 0（macOS，Python 3，2026-10-14） |
| passed | `python3 -m labs.ccl_sim` | 模拟器 CLI | 打印 AllReduce、分层、All2All 三张对比表，退出码 0 |
| passed | `python3 scripts/check_repo.py bogus`；`bash scripts/fetch_third_party.sh bogus` | 负向：非法参数 | 均退出 2 |
| not-run | nccl-tests、perftest、vLLM/SGLang/DeepEP/Triton-distributed 示例 | GPU/RDMA 实验 | 本机无 GPU 与 RDMA 网卡 |
| passed | deep-research 工作流 | 推理并行与 EP 调研 | 102 个 agent 无错误；25 条结论经 3 票验证，24 条确认、1 条推翻 |
| passed | `cd site && npm run build` | 学习站点 | 52 个页面，无死链，`site verify: passed`（Node v24，VitePress 1.6.4，2026-10-15） |
| passed | `bash scripts/fetch_third_party.sh` | DeepEP 升级、新增 DeepInfer | `CLONE` 后 `OK DeepEP b5a33eef`、`OK DeepInfer 774ce244`，其余 8 个 OK（2026-10-20） |
| not-run | `third_party/DeepEP/tests/ep/test_ep.py`、DeepInfer `examples/offline_inference.py` | EP kernel 与推理实测 | 需要 Hopper/Blackwell 多卡 + IB/RoCE；DeepInfer 需 SM100 |
| partial | 笔记技术内容 | `notes/` | 由并行 Agent 依据本地源码撰写，只有路径经机械核对；函数名、结论需阅读时抽查 |

## 阻塞、授权与恢复条件

- GPU/RDMA 实验：需要一台带 NVIDIA GPU（最好 ≥2 卡 NVLink）和 RDMA 网卡的机器；拿到后按 `notes/07-perf-troubleshooting/README.md` 运行，结果补入对应笔记。

## 风险与未知项

- 配图过程中发现若干原文与源码不一致之处尚未修正：`07-perf-troubleshooting/03-troubleshooting.md`（`/etc/nccl.conf` 不会覆盖已设置的环境变量）、`01-rdma/05-gpudirect-and-nccl-ib.md`（GDR 启用条件两处）、`04-nccl-internals/02-topology-graph.md`（DupChannels 时机、BALANCED_TREE 回退）、`04-nccl-internals/01-init-bootstrap.md`（bootstrap allgather 为双向环）、`03-ccl-algorithms/04-all2all.md`（Bruck 描述）、`03-ccl-algorithms/05-handwrite-ring-fullmesh.md`（two-shot 屏障）、`09-ep-kernels/README.md`（warp group 数）、`00-roadmap/00-start-here.md` 头部缺 `DeepEP-hybrid-ep` 版本、Halving-Doubling 结论在三篇中不一致。
- `09-ep-kernels` 中"同 QP 上 WRITE 后 ATOMIC 可见 ⇒ 数据可见"的结论依赖 RC 按序语义，在乱序放置网卡上的保证待确认（见该章 README 待深入）。
- 学习站点：`npm install` 报告 esbuild 的 postinstall 被 npm allow-scripts 拦截，但构建正常（使用平台预编译包）；如在其他机器构建失败，运行 `npm approve-scripts esbuild`。
- 学习路线是否改为"推理优先"尚未决定；若调整，需同步 `site/.vitepress/theme/components/RoadmapCards.vue` 的卡片数据。
- 笔记技术细节可能有误：各篇末尾"待深入"列出了待确认项及确认方式；阅读时对照源码修正。
- PyTorch 源码未克隆：`ProcessGroupNCCL` C++ 内部标为待确认；确认方式：经用户同意后把 pytorch/pytorch 加入 `repos.lock`。
- 版本时效：本地 NCCL 2.32.3、DeepEP V2.5 与网上多数解读不同（如 DeepEP 已移除 NVSHMEM、NCCL 无 `ext-net/` 目录），笔记以本地版本为准。
- JD 中"scale up 等互联技术""自研 RDMA 协议"的具体所指：见 `docs/CONTEXT.md` 待澄清表。

## 下一步

1. 通读学习入口 `notes/00-roadmap/00-start-here.md`，做完其中的面试自测，并更新 `notes/00-roadmap/02-jd-breakdown.md` 自评。
2. 阶段 1：读 `notes/03-ccl-algorithms/02-alpha-beta-model.md` 与 ai-infra-book §6.4，用 `python3 -m labs.ccl_sim --alpha-us ... --bw-gbps ...` 代入书中参数复现 Ring/Tree 约 680 KiB 交点，更新 `notes/00-roadmap/02-jd-breakdown.md` 自评。
3. 为本仓库选定 License（远程是公开仓库时尤其需要）。
4. 补齐学习入口"待深入"中的调研缺口：CP/SP 推理通信、vLLM/SGLang 其余 all2all 后端、MegaScale-Infer M2N 细节。
