import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'
import { listExcluded } from './content.mts'
import { linkRewrite } from './linkRewrite.mts'
import { tokenize } from './search.mts'
import { buildSidebar } from './sidebar.mts'

// 被渲染的 Markdown 位于 site/ 之外，Node 从那里解析不到 site/node_modules/vue，需要显式指回。
const nm = (p: string) => fileURLToPath(new URL(`../node_modules/${p}`, import.meta.url))

export default defineConfig({
  lang: 'zh-CN',
  title: 'CCL 学习手册',
  description: '从推理出发，看懂并行、通信与 EP',
  // 本地为 '/'；GitHub Pages 项目站点由工作流设置 SITE_BASE=/ai-learning/。
  base: process.env.SITE_BASE || '/',
  srcDir: '..',
  srcExclude: listExcluded(),
  rewrites: { 'site/index.md': 'index.md' },
  lastUpdated: false,
  markdown: {
    math: true,
    lineNumbers: false,
    config: (md) => md.use(linkRewrite),
  },
  vite: {
    resolve: {
      alias: [
        { find: /^vue$/, replacement: nm('vue/dist/vue.runtime.esm-bundler.js') },
        { find: /^vue\/server-renderer$/, replacement: nm('vue/server-renderer/index.mjs') },
      ],
    },
  },
  themeConfig: {
    nav: [
      {
        text: '路线',
        items: [
          { text: '从这里开始', link: '/notes/00-roadmap/00-start-here' },
          { text: '10 周学习路线', link: '/notes/00-roadmap/03-learning-path' },
          { text: 'JD 拆解与自评', link: '/notes/00-roadmap/02-jd-breakdown' },
          { text: 'ai-infra-book 阅读地图', link: '/notes/00-roadmap/01-ai-infra-book-map' },
        ],
      },
      { text: '笔记', link: '/notes/00-roadmap/README' },
      {
        text: '参考',
        items: [
          { text: '推理并行与 EP 调研', link: '/docs/references/2026-10-14-inference-parallelism-ep-research' },
          { text: '外部资料与版本', link: '/docs/references/third-party' },
          { text: '术语表', link: '/docs/CONTEXT' },
        ],
      },
    ],
    sidebar: buildSidebar(),
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          options: { tokenize },
          searchOptions: { combineWith: 'AND', fuzzy: 0.1, prefix: true, tokenize },
        },
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到',
            resetButtonTitle: '清除',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
  },
})
