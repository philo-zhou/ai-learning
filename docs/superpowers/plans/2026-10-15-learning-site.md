# 学习站点（VitePress）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> 状态：Completed（2026-10-15）
>
> Spec：`docs/superpowers/specs/2026-10-15-learning-site-design.md`

**Goal:** 在 `site/` 建一个 VitePress 站点，直接渲染 `notes/`、`docs/references/`、`docs/CONTEXT.md`，技术手册风格、中文可搜索、公式可渲染。

**Architecture:** VitePress 项目放在 `site/`，`srcDir: '..'` 指向仓库根；页面清单由 `content.mts` 中的白名单函数统一决定（`srcExclude`、侧边栏、链接改写共用）。因为被渲染的 Markdown 在 `site/` 之外，Node 解析不到 `site/node_modules/vue`，所以在 `vite.resolve.alias` 中把 `vue` 与 `vue/server-renderer` 指回 `site/node_modules`（已在 /tmp 原型中验证可构建）。

**Tech Stack:** VitePress 1.6.4、Vue 3（随 VitePress）、markdown-it-mathjax3 4.x、Node v24；验证用 Python 3（`scripts/check_repo.py`）与本机 Google Chrome headless 截图。

## Global Constraints

- `notes/`、`docs/` 是内容唯一来源，`site/` 不存放笔记副本。
- 不依赖外部 CDN 字体或脚本；字体用系统字体栈。
- 不做"标记已读"。不部署、不 commit/push（`AGENTS.md`：commit 需用户决定）。
- 安装站点依赖后 `python3 scripts/check_repo.py` 必须仍全部通过。
- 配色：亮色背景 `#FAF8F3`、正文 `#2B2A27`、主色 `#3B4A9C`；暗色背景 `#141821`、正文 `#D9D6CF`、主色 `#8C9BEA`；下层 `#B5652B`、中层 `#3B4A9C`、上层 `#2F7D74`。
- 未渲染文件的链接改写为 `https://github.com/philo-zhou/ai-learning/blob/main/<path>`（目录用 `/tree/main/`）。

## 文件结构

| 文件 | 职责 |
|---|---|
| `site/package.json`、`site/package-lock.json` | 依赖与 `dev`/`build`/`preview` 脚本 |
| `site/.vitepress/content.mts` | 白名单 `isRendered(rel)`、`listExcluded()`、章节表 `CHAPTERS`、`githubUrl()` |
| `site/.vitepress/sidebar.mts` | 由 `CHAPTERS` 和 `notes/` 文件生成侧边栏 |
| `site/.vitepress/linkRewrite.mts` | markdown-it 插件：未渲染目标改写为 GitHub 链接 |
| `site/.vitepress/search.mts` | 自包含的中英文 tokenizer（会被序列化到客户端） |
| `site/.vitepress/config.mts` | 组装以上模块、nav、alias、math、search |
| `site/.vitepress/theme/index.ts`、`style.css`、`components/LayerStrip.vue`、`components/RoadmapCards.vue` | 主题与首页组件 |
| `site/index.md` | 首页 |
| `site/scripts/verify.mjs` | 构建后检查：页面数、链接改写、公式、搜索 tokenizer |

---

### Task 1：骨架、构建与仓库门禁

**Files:** Create `site/package.json`、`site/.vitepress/content.mts`、`site/.vitepress/config.mts`、`site/index.md`；Modify `.gitignore`、`scripts/check_repo.py`

- [x] Step 1：`site/package.json`：

```json
{
  "name": "ccl-learning-site",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vitepress dev .",
    "build": "vitepress build . && node scripts/verify.mjs",
    "preview": "vitepress preview ."
  },
  "devDependencies": {
    "markdown-it-mathjax3": "^4.3.2",
    "vitepress": "1.6.4",
    "vue": "^3.5.0"
  }
}
```

- [x] Step 2：`content.mts` 写白名单（渲染 `notes/**`（不含 AGENTS.md）、`docs/references/**`（不含模板 `reference.md`）、`docs/CONTEXT.md`、`site/index.md`），`listExcluded()` 遍历仓库（跳过 `.git`、`third_party`、`node_modules`、`site/.vitepress`）返回所有非白名单 `.md`，加上 `third_party/**`、`site/node_modules/**` 两个 glob。
- [x] Step 3：`config.mts` 设 `srcDir: '..'`、`srcExclude: listExcluded()`、`rewrites: {'site/index.md': 'index.md'}`、`markdown.math: true`、vue alias。
- [x] Step 4：`cd site && npm install && npx vitepress build .`，预期构建成功、无死链报错。
- [x] Step 5：`.gitignore` 加 `site/node_modules/`、`site/.vitepress/cache/`、`site/.vitepress/dist/`；`check_repo.py` 的 `markdown_files()` 跳过 `site/node_modules/` 与 `site/.vitepress/`。运行 `python3 scripts/check_repo.py`，预期全部 passed（此时 `site/node_modules` 中已有大量 `.md`，这一步就是回归测试）。

### Task 2：侧边栏、导航与链接改写

**Files:** Create `site/.vitepress/sidebar.mts`、`linkRewrite.mts`；Modify `config.mts`

- [x] Step 1：`CHAPTERS` 表（目录、层级 `overview|lower|middle|upper|practice`），00~09 按编号排序；`sidebar.mts` 为每章生成 `{ text: '<span class="layer-dot layer-X"></span>' + 章节 README 的 H1, collapsed, items }`，items 中 README 第一，其余按文件名排序，text 取 H1。
- [x] Step 2：`linkRewrite.mts`：core rule 遍历 `link_open`，对相对 href 用 `env.relativePath` 解析为仓库相对路径；若不是 `isRendered()` 的 `.md`，改写为 `githubUrl()`。
- [x] Step 3：nav：路线（start-here / learning-path / jd-breakdown）、笔记（`/notes/00-roadmap/README`）、参考（调研报告、术语表）；`outline: [2,3]`、中文的 docFooter/outline 文案。
- [x] Step 4：构建，确认未渲染文件的链接已变为 GitHub URL，且无死链报错。（实施偏差：笔记中实际指向未渲染文件的只有 `notes/03-ccl-algorithms/README.md` → `labs/README.md`，`verify.mjs` 改为检查这一处。）

### Task 3：中文搜索

**Files:** Create `site/.vitepress/search.mts`；Modify `config.mts`

- [x] Step 1：`tokenize(text)`：按 `[A-Za-z0-9_]+` 取英文词（小写），按 `[㐀-鿿]+` 取中文串并输出单字与相邻双字；函数体自包含（VitePress 用 `toString` 序列化到客户端）。
- [x] Step 2：`search: { provider: 'local', options: { miniSearch: { options: { tokenize }, searchOptions: { combineWith: 'AND', fuzzy: 0.1, prefix: true } } } }`。
- [x] Step 3：`verify.mjs` 中 import `search.mts` 的 tokenize 并断言 `tokenize('拥塞控制 IBGDA')` 包含 `拥塞`、`控制`、`ibgda`。

### Task 4：主题与首页

**Files:** Create `theme/index.ts`、`theme/style.css`、`components/LayerStrip.vue`、`components/RoadmapCards.vue`；Modify `site/index.md`

- [x] Step 1：`style.css` 覆盖 VitePress CSS 变量（配色、字体），正文 `max-width: 72ch`、`line-height: 1.8`，表格/引用/代码块样式，`.layer-dot` 四种颜色。
- [x] Step 2：`LayerStrip.vue`：三段（上层框架 / 中层通信库 / 下层互联）横向卡片，每段链接到对应章节。
- [x] Step 3：`RoadmapCards.vue`：S1~S8 卡片，数据来自 `notes/00-roadmap/03-learning-path.md` 各阶段"目标"。
- [x] Step 4：`index.md` 使用 `layout: home` 的 hero（标题、简介、"从这里开始"按钮）+ 两个组件。

### Task 5：验证脚本、截图与文档同步

**Files:** Create `site/scripts/verify.mjs`；Modify `README.md`、`AGENTS.md`、`docs/TESTING.md`、`docs/architecture/overview.md`、`CHANGELOG.md`、`docs/PROGRESS.md`

- [x] Step 1：`verify.mjs` 检查 dist：章节 README 页面 10 个都存在；`00-start-here.html` 含 `mjx-container`；至少一个 `github.com/philo-zhou/ai-learning` 链接；tokenizer 断言。失败退出 1。
- [x] Step 2：`npm run build` 通过；`python3 scripts/check_repo.py` 通过。
- [x] Step 3：`npm run preview` 后用 Chrome headless 截图首页、`00-start-here`、暗色模式，人工检查。
- [x] Step 4：文档同步：README 启动方式；`AGENTS.md` 执行能力表加 site 行、文档维护路由"新增/移动笔记章节"加 `site/.vitepress/content.mts`；TESTING 加站点构建层级；overview 组件图加 `site/`；CHANGELOG、PROGRESS 记录。
