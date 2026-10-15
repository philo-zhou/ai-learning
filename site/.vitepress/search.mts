// 中英文混合分词。VitePress 会把函数 toString 后发到浏览器，所以函数体必须自包含。
export function tokenize(text: string): string[] {
  const tokens: string[] = []
  const re = /[A-Za-z0-9_.+-]+|[㐀-鿿]+/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const s = m[0]
    if (/^[㐀-鿿]/.test(s)) {
      for (let i = 0; i < s.length; i++) {
        tokens.push(s[i])
        if (i + 1 < s.length) tokens.push(s.slice(i, i + 2))
      }
    } else {
      const w = s.replace(/^[.+-]+|[.+-]+$/g, '').toLowerCase()
      if (w) tokens.push(w)
    }
  }
  return tokens
}
