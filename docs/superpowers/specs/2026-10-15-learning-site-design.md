# 学习站点（VitePress）设计规格

> 状态：Approved（用户 2026-10-15 审阅后同意实施）
>
> 日期：2026-10-15
>
> 关联：用户请求"把本仓库的学习路径文档构建成一个网页，要有审美"

## 问题与现有证据

- 学习内容全部是 Markdown：`notes/00~09` 十个章节（`notes/00-roadmap/README.md` 为总入口，`notes/00-roadmap/00-start-here.md` 为学习入口），加上 `docs/references/`、`docs/CONTEXT.md`。
- 在编辑器里读不方便：章节之间跳转靠相对链接，没有侧边导航、全文搜索，公式（7 篇笔记用了 `$…$` / `$$…$$`）不渲染。
- 本机没有 pandoc、mkdocs；有 Node v24 与 npm。

## 目标

- 在本地一条命令启动一个可阅读全部笔记的网站，笔记修改后刷新即可看到最新内容，不需要复制或转换笔记。
- 视觉上是"技术手册风"：适合长时间阅读中文长文，有清晰的章节导航和三层知识的颜色区分，支持暗色模式。

## 范围

- 新建 `site/` 目录，内含独立的 VitePress 项目（`package.json`、`.vitepress/config.mts`、`.vitepress/theme/`、首页 `site/index.md`）。
- 渲染 `notes/` 全部笔记，以及 `docs/references/`、`docs/CONTEXT.md`。
- 自定义主题样式、首页、侧边栏、中文全文搜索、公式渲染。
- 仓库治理同步：`.gitignore`、`scripts/check_repo.py`、`AGENTS.md` 执行能力表、`docs/TESTING.md`、`README.md`、`CHANGELOG.md`、`docs/PROGRESS.md`、`docs/architecture/overview.md`。

## 非目标

- 不做阅读进度/"标记已读"（用户 2026-10-15 明确不要）。
- 不部署到 GitHub Pages 或任何远程（发布属于对外操作，需用户另行授权）；只保证 `npm run build` 产物可部署。
- 不修改笔记正文的写法来迁就站点；站点适配笔记，而不是反过来。
- 不渲染 `third_party/`、`labs/`、`scripts/`、各级 `AGENTS.md` 和 `docs/superpowers/` 等治理文档。

## 用户与场景

- 主路径：用户执行 `cd site && npm install`（首次）→ `npm run dev` → 浏览器打开本地地址 → 首页点击"从这里开始"进入 `00-start-here` → 按侧边栏或"下一篇"顺序阅读 → 用搜索框查术语（如"IBGDA""busbw"）。
- 失败路径：
  - 未安装依赖就运行 → npm 报错，README 中写明先 `npm install`。
  - 笔记里出现指向未渲染文件的链接（如 `../../AGENTS.md`、`labs/...`）→ 不得导致构建失败，见"接口与数据"。

## 约束与不变量

- `notes/`、`docs/` 是内容的唯一权威来源；`site/` 不存放笔记副本。
- 现有验证入口 `python3 scripts/check_repo.py` 在安装站点依赖后必须仍然通过（不能扫描 `site/node_modules` 里的 Markdown）。
- `third_party/` 只读，不被站点构建读取。
- `npm install` 需要联网，只在用户执行或授权时运行；`node_modules`、构建缓存与产物不进入 Git。
- 不依赖外部 CDN 字体或脚本：字体使用系统字体栈，页面离线可用。

## 方案

### 架构与流程

```text
site/
├── package.json            vitepress、markdown-it-mathjax3（devDependencies）；scripts: dev / build / preview
├── index.md                首页（layout: home + 自定义组件）
└── .vitepress/
    ├── config.mts          srcDir='..'、srcExclude、rewrites、sidebar、nav、search、markdown.math
    ├── sidebar.mts         从 notes/ 目录生成侧边栏（章节顺序固定，文章按文件名排序、标题取 H1）
    └── theme/
        ├── index.ts        extends DefaultTheme，注册首页组件
        ├── style.css       配色、字体、排版变量
        └── components/     RoadmapCards.vue（S1~S8 路线卡片）、LayerStrip.vue（三层调用链示意）
```

- **数据来源**：`srcDir` 指向仓库根目录；`srcExclude` 排除 `third_party/**`、`labs/**`、`scripts/**`、`site/node_modules/**`、`**/AGENTS.md`、`CLAUDE.md`、`CODEBUDDY.md`、`docs/superpowers/**`、`docs/operations/**`、`docs/changelog/**` 等非学习内容。`rewrites` 把 `site/index.md` 映射为站点根 `/`。
- **导航**：
  - 顶栏三项："路线"（`00-start-here`、`03-learning-path`、`02-jd-breakdown`）、"笔记"（章节总入口）、"参考"（调研报告、术语表 `docs/CONTEXT.md`）。
  - 侧边栏按章节 00~09 分组，章节顺序与 `notes/00-roadmap/README.md` 的"全部章节"一致，写死在 `sidebar.mts`；每章内 `README.md` 放第一项，其余按文件名排序，显示名取文件的第一个 `# ` 标题。
  - 每页右侧显示 H2/H3 大纲，底部上一篇/下一篇按侧边栏顺序。
- **三层颜色**：下层（01-rdma、02-scale-up）铜橙，中层（03、04、09）靛蓝，上层与跨层（05、06）青绿，实践与求职（07、08）及总览（00）中性灰。用于侧边栏分组标记和首页卡片。
- **首页**：
  1. 标题"从推理出发，看懂并行、通信与 EP"和一句话简介；
  2. `LayerStrip`：上层框架 → 中层通信库 → 下层互联的横向示意，每段链接到对应章节；
  3. `RoadmapCards`：S1~S8 八张卡片，每张显示周次、目标一句话、对应章节链接，数据写在组件内（来源 `notes/00-roadmap/03-learning-path.md`）；
  4. 主按钮"从这里开始"→ `00-start-here`。

### 接口与数据

- **命令**（在 `site/` 下）：`npm install`；`npm run dev`（本地开发服务器，热更新）；`npm run build`（输出 `site/.vitepress/dist`）；`npm run preview`。
- **链接处理**：
  - 笔记之间的 `../xx/yy.md` 相对链接由 VitePress 原生解析。
  - 指向未渲染文件（`AGENTS.md`、`labs/`、`scripts/`、`docs/superpowers/` 等）的链接，通过一个 markdown-it 插件在渲染时改写为 GitHub 链接（`https://github.com/philo-zhou/ai-learning/blob/main/<path>`，仓库远程见 `docs/PROGRESS.md`），并且不计入死链检查；其余站内死链保持 VitePress 默认的构建失败行为。
  - 行内代码里的 `third_party/...` 路径保持为代码文本，不做链接。
- **公式**：`markdown.math: true`（依赖 `markdown-it-mathjax3`），支持 `$…$` 与 `$$…$$`。
- **代码块**：Shiki 高亮；`text` 语言块（ASCII 图）使用等宽字体、不换行、可横向滚动。
- **搜索**：VitePress 本地搜索（minisearch），自定义 tokenizer：英文按词切分，中文按单字 + 相邻双字切分，保证"拥塞控制""自适应路由"之类的中文词能被搜到。

### 视觉规范

| 项 | 亮色 | 暗色 |
|---|---|---|
| 背景 | 暖白 `#FAF8F3` | 深墨蓝 `#141821` |
| 正文文字 | `#2B2A27` | `#D9D6CF` |
| 主色（链接、强调） | 靛蓝 `#3B4A9C` | `#8C9BEA` |
| 下层 / 中层 / 上层 | 铜橙 `#B5652B` / 靛蓝 `#3B4A9C` / 青绿 `#2F7D74` | 相应提亮 |

- 字体：正文衬线（`"Source Han Serif SC", "Noto Serif CJK SC", "Songti SC", Georgia, serif`），标题与界面无衬线（`"PingFang SC", "Source Han Sans SC", system-ui, sans-serif`），代码等宽（`"JetBrains Mono", "SF Mono", Menlo, monospace`）。
- 排版：正文最大宽度约 72 个中文字符，行高 1.8，段间距加大；表格横向可滚动，表头浅底色；引用块左侧细竖线 + 浅底。
- 暗色模式通过 VitePress 自带切换按钮。

### 错误、安全、隐私与可靠性

- 站点只读取本仓库 Markdown，不读取 `third_party/`，不发起网络请求（构建和运行时都不依赖 CDN）。
- 不渲染治理文档与 `AGENTS.md`，避免把 Agent 规则当作学习内容展示。
- 依赖版本由 `package-lock.json` 固定；`node_modules` 与构建产物被 `.gitignore` 排除。

## 替代方案

| 方案 | 放弃原因 |
|---|---|
| 零构建的单页（浏览器运行时读取 md 渲染） | 用户在 2026-10-15 选择了 VitePress 方案 |
| 自写 Node 构建脚本生成静态 HTML | 用户选择 VitePress；自写方案要重复实现导航、搜索、热更新 |
| 把笔记复制到 `site/docs/` | 产生两份内容，必然漂移，违反"`notes/` 是唯一权威"的约束 |
| 保留"标记已读"进度功能 | 用户明确不要 |

## 验收标准

- [ ] `cd site && npm install && npm run build` 退出码 0，且 VitePress 死链检查无报错。
- [ ] `npm run dev` 后首页可见标题、三层示意、S1~S8 路线卡片和"从这里开始"按钮。
- [ ] 侧边栏列出 00~09 全部章节，章节内文章完整；在任一笔记中点击另一笔记的相对链接能正确跳转。
- [ ] 含公式的笔记（如 `notes/00-roadmap/00-start-here.md` 第 3.2 节 $V \approx 2nkfd$）公式正确渲染。
- [ ] 搜索"拥塞控制""IBGDA""busbw"都能返回相关笔记。
- [ ] 指向 `AGENTS.md`、`labs/` 的链接跳转到 GitHub 而不是 404。
- [ ] 暗色模式切换正常，代码块和表格在两种模式下都可读。
- [ ] 安装依赖后 `python3 scripts/check_repo.py` 仍然全部通过。
- [ ] 截图首页、一篇带公式的笔记、暗色模式各一张，人工确认视觉效果。

## 测试、发布与文档

- 验证：上面的验收清单；`npm run build` 作为站点的机械门禁，记入 `AGENTS.md` 执行能力表与 `docs/TESTING.md` 的测试层级。
- `scripts/check_repo.py` 的 Markdown 遍历排除 `site/node_modules/` 与 `site/.vitepress/cache/`、`site/.vitepress/dist/`。
- 文档同步：`README.md`（如何启动站点）、`AGENTS.md` 执行能力表（新增 site 行）、`docs/TESTING.md`、`docs/architecture/overview.md`（仓库组件图加入 `site/`）、`CHANGELOG.md`、`docs/PROGRESS.md`。
- 发布：不在本次范围。

## 风险与未决问题

- 新增或移动章节时 `sidebar.mts` 的章节顺序需要手动更新。缓解：把它加入 `AGENTS.md` 的"文档维护路由"表中"新增/移动笔记章节"一行。
- 首页路线卡片的数据写在组件里，与 `03-learning-path.md` 可能不同步；如果用户决定按推理优先重排学习路线（2026-10-15 会话中提出、尚未决定），需要同步改卡片。
- 中文分词是简化的单字 + 双字方案，可能出现少量无关结果；可接受。
- VitePress 对 `srcDir` 指向项目目录之外的支持需在实现第一步验证；若不可行，改为把 VitePress 项目根设为仓库根、配置目录放在 `site/.vitepress`（通过 `vitepress dev ..` 的方式），不改变内容来源原则。

## 参考资料

- VitePress 官方文档（配置项 `srcDir`、`srcExclude`、`rewrites`、`markdown.math`、`search.provider: 'local'`），实现时按 `package-lock.json` 锁定的版本核对。
- 本仓库 `notes/00-roadmap/README.md`（章节顺序）、`notes/00-roadmap/03-learning-path.md`（路线卡片数据）。
