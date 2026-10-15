// 构建后检查：章节页面齐全、公式已渲染、未渲染文件的链接已改写、中文分词可用。失败退出 1。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const dist = path.join(siteDir, '.vitepress/dist')
const notesDir = path.join(siteDir, '../notes')
const errors = []
const read = (rel) => fs.readFileSync(path.join(dist, rel), 'utf-8')

for (const ch of fs.readdirSync(notesDir).filter((d) => /^\d\d-/.test(d))) {
  for (const f of fs.readdirSync(path.join(notesDir, ch)).filter((f) => f.endsWith('.md') && f !== 'AGENTS.md')) {
    const html = `notes/${ch}/${f.replace(/\.md$/, '.html')}`
    if (!fs.existsSync(path.join(dist, html))) errors.push(`缺少页面 ${html}`)
  }
}
if (fs.existsSync(path.join(dist, 'notes/AGENTS.html'))) errors.push('AGENTS.md 不应被渲染')
if (!fs.existsSync(path.join(dist, 'index.html'))) errors.push('缺少首页 index.html')

const startHere = read('notes/00-roadmap/00-start-here.html')
if (!startHere.includes('mjx-container')) errors.push('00-start-here 中公式未渲染')
// notes/03-ccl-algorithms/README.md 链接到未渲染的 labs/README.md，应被改写到 GitHub。
if (!read('notes/03-ccl-algorithms/README.html').includes('https://github.com/philo-zhou/ai-learning/blob/main/labs/README.md'))
  errors.push('指向 labs/README.md 的链接未改写为 GitHub 链接')

const { tokenize } = await import(path.join(siteDir, '.vitepress/search.mts'))
const tokens = tokenize('拥塞控制 与 IBGDA、busbw')
for (const t of ['拥塞', '控制', 'ibgda', 'busbw'])
  if (!tokens.includes(t)) errors.push(`tokenize 缺少 ${t}：${JSON.stringify(tokens)}`)

for (const e of errors) console.error('FAIL', e)
console.log(`site verify: ${errors.length ? 'failed' : 'passed'} (${errors.length} problems)`)
process.exit(errors.length ? 1 : 0)
