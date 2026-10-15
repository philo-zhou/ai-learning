import fs from 'node:fs'
import path from 'node:path'
import type { DefaultTheme } from 'vitepress'
import { CHAPTERS, REPO_ROOT, readTitle } from './content.mts'

function chapterItems(dir: string): DefaultTheme.SidebarItem[] {
  const files = fs
    .readdirSync(path.join(REPO_ROOT, 'notes', dir))
    .filter((f) => f.endsWith('.md') && f !== 'AGENTS.md' && f !== 'README.md')
    .sort()
  return ['README.md', ...files].map((f) => ({
    text: f === 'README.md' ? '本章导读' : readTitle(`notes/${dir}/${f}`),
    link: `/notes/${dir}/${f.replace(/\.md$/, '')}`,
  }))
}

export function buildSidebar(): DefaultTheme.Sidebar {
  return {
    '/': CHAPTERS.map(({ dir, layer }) => ({
      text: `<span class="layer-dot layer-${layer}"></span>${readTitle(`notes/${dir}/README.md`)}`,
      collapsed: dir !== '00-roadmap',
      items: chapterItems(dir),
    })),
  }
}
