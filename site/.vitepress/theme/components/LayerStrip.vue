<script setup lang="ts">
import { withBase } from 'vitepress'

const layers = [
  {
    key: 'upper',
    name: '上层 · 框架',
    desc: 'vLLM / SGLang 的并行层决定谁和谁通信、传多少',
    items: [
      { text: '框架调用链', link: '/notes/05-framework-integration/README' },
      { text: '通算融合', link: '/notes/06-comm-compute-fusion/README' },
    ],
  },
  {
    key: 'middle',
    name: '中层 · 通信库',
    desc: '算法、NCCL 实现，以及 DeepEP 这类 EP 专用库',
    items: [
      { text: '算法与代价模型', link: '/notes/03-ccl-algorithms/README' },
      { text: 'NCCL 内部', link: '/notes/04-nccl-internals/README' },
      { text: 'EP 专用库', link: '/notes/09-ep-kernels/README' },
    ],
  },
  {
    key: 'lower',
    name: '下层 · 互联',
    desc: '机间 RDMA 与机内 NVLink 如何承载字节',
    items: [
      { text: 'RDMA', link: '/notes/01-rdma/README' },
      { text: 'Scale-up', link: '/notes/02-scale-up/README' },
    ],
  },
]
</script>

<template>
  <section class="ls">
    <div v-for="(l, i) in layers" :key="l.key" class="ls-row">
      <div class="ls-card" :class="`ls-${l.key}`">
        <div class="ls-name">{{ l.name }}</div>
        <p class="ls-desc">{{ l.desc }}</p>
        <div class="ls-links">
          <a v-for="it in l.items" :key="it.link" :href="withBase(it.link)">{{ it.text }}</a>
        </div>
      </div>
      <div v-if="i < layers.length - 1" class="ls-arrow" aria-hidden="true">调用<br>→</div>
    </div>
  </section>
</template>

<style scoped>
.ls { max-width: 1152px; margin: 8px auto 0; display: flex; align-items: stretch; }
.ls-row { flex: 1; display: flex; align-items: stretch; }
.ls-card { flex: 1; }
.ls-card {
  border: 1px solid var(--vp-c-divider);
  border-left: 4px solid var(--layer-color);
  border-radius: 6px;
  padding: 14px 20px;
  background: var(--vp-c-bg-soft);
}
.ls-upper { --layer-color: var(--ccl-upper); }
.ls-middle { --layer-color: var(--ccl-middle); }
.ls-lower { --layer-color: var(--ccl-lower); }
.ls-name { font-family: var(--vp-font-family-base); font-weight: 600; color: var(--layer-color); letter-spacing: 0.02em; }
.ls-desc { margin: 4px 0 8px; font-size: 14px; color: var(--vp-c-text-2); }
.ls-links { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; }
.ls-links a { color: var(--vp-c-text-1); text-decoration: none; border-bottom: 1px solid var(--vp-c-divider); }
.ls-links a:hover { color: var(--layer-color); border-color: var(--layer-color); }
.ls-arrow { align-self: center; text-align: center; font-size: 12px; line-height: 1.4; color: var(--vp-c-text-3); padding: 0 10px; letter-spacing: 0.1em; }
@media (max-width: 768px) {
  .ls, .ls-row { flex-direction: column; }
  .ls-arrow { padding: 6px 0; }
  .ls-arrow br { display: none; }
}
</style>
