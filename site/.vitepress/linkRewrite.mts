// 指向未渲染文件（AGENTS.md、labs/、docs/PROGRESS.md 等）的相对链接改写为 GitHub 链接。
import path from 'node:path'
import type MarkdownIt from 'markdown-it'
import { githubUrl, isRendered } from './content.mts'

export function linkRewrite(md: MarkdownIt) {
  md.core.ruler.push('repo-link-rewrite', (state) => {
    const from: string | undefined = state.env?.relativePath
    if (!from) return
    // relativePath 已经过 rewrites，site/index.md 会变成 index.md；它没有相对链接，不影响结果。
    for (const block of state.tokens) {
      for (const token of block.children ?? []) {
        if (token.type !== 'link_open') continue
        const href = token.attrGet('href')
        if (!href || /^([a-z]+:|#|\/)/i.test(href)) continue
        const [target, hash = ''] = href.split(/(?=#)/)
        const rel = path.posix.normalize(path.posix.join(path.posix.dirname(from), decodeURI(target)))
        if (rel.startsWith('..') || isRendered(rel)) continue
        token.attrSet('href', githubUrl(rel, hash))
        token.attrSet('target', '_blank')
      }
    }
  })
}
