// 站点渲染哪些 Markdown 的唯一决定点：srcExclude、侧边栏、链接改写都从这里取。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
export const GITHUB_BASE = 'https://github.com/philo-zhou/ai-learning'

export type Layer = 'overview' | 'lower' | 'middle' | 'upper' | 'practice'

// 章节顺序与 notes/00-roadmap/README.md 的"全部章节"一致；新增章节时同步这里。
export const CHAPTERS: { dir: string; layer: Layer }[] = [
  { dir: '00-roadmap', layer: 'overview' },
  { dir: '01-rdma', layer: 'lower' },
  { dir: '02-scale-up', layer: 'lower' },
  { dir: '03-ccl-algorithms', layer: 'middle' },
  { dir: '04-nccl-internals', layer: 'middle' },
  { dir: '05-framework-integration', layer: 'upper' },
  { dir: '06-comm-compute-fusion', layer: 'upper' },
  { dir: '07-perf-troubleshooting', layer: 'practice' },
  { dir: '08-interview', layer: 'practice' },
  { dir: '09-ep-kernels', layer: 'middle' },
]

// rel：相对仓库根的 POSIX 路径。
export function isRendered(rel: string): boolean {
  if (!rel.endsWith('.md') || path.posix.basename(rel) === 'AGENTS.md') return false
  if (rel === 'site/index.md' || rel === 'docs/CONTEXT.md') return true
  if (rel.startsWith('notes/')) return true
  return rel.startsWith('docs/references/') && rel !== 'docs/references/reference.md'
}

const SKIP_DIRS = new Set(['.git', 'third_party', 'node_modules', '.vitepress'])

function walk(dir: string, out: string[]) {
  for (const entry of fs.readdirSync(path.join(REPO_ROOT, dir), { withFileTypes: true })) {
    const rel = dir ? `${dir}/${entry.name}` : entry.name
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name)) walk(rel, out)
    } else if (entry.name.endsWith('.md')) {
      out.push(rel)
    }
  }
}

export function listExcluded(): string[] {
  const files: string[] = []
  walk('', files)
  return ['third_party/**', 'site/node_modules/**', ...files.filter((f) => !isRendered(f))]
}

export function githubUrl(rel: string, hash = ''): string {
  const full = path.join(REPO_ROOT, rel)
  const kind = fs.existsSync(full) && fs.statSync(full).isDirectory() ? 'tree' : 'blob'
  return `${GITHUB_BASE}/${kind}/main/${rel.replace(/\/$/, '')}${hash}`
}

export function readTitle(rel: string): string {
  const text = fs.readFileSync(path.join(REPO_ROOT, rel), 'utf-8')
  const m = text.match(/^#\s+(.+)$/m)
  return m ? m[1].trim() : path.posix.basename(rel, '.md')
}
