<script setup lang="ts">
import { withBase } from 'vitepress'

// 数据来源：notes/00-roadmap/03-learning-path.md 各阶段的"目标"；路线调整时同步这里。
const stages = [
  { id: 'S1', weeks: '第 1–2 周', title: '算法与代价模型', layer: 'middle', goal: '推导 Ring / Tree / Halving-Doubling 的轮数与字节数，用 α-β 模型判断谁占主导', link: '/notes/03-ccl-algorithms/README' },
  { id: 'S2', weeks: '第 3 周', title: 'RDMA 与数据中心网络', layer: 'lower', goal: 'Verbs 与 QP 建链、单边/双边语义、IB 与 RoCEv2、PFC + ECN/DCQCN、GPUDirect', link: '/notes/01-rdma/README' },
  { id: 'S3', weeks: '第 4 周', title: 'Scale-up 互联', layer: 'lower', goal: '画出 GPU / NVSwitch / PCIe / NIC / NUMA 拓扑，讲清 P2P、SHM、NVLS 何时被选用', link: '/notes/02-scale-up/README' },
  { id: 'S4', weeks: '第 5–6 周', title: 'NCCL 内部实现', layer: 'middle', goal: '从 ncclCommInitRank 讲到 runRing：拓扑、图搜索、channel、协议、tuning、proxy', link: '/notes/04-nccl-internals/README' },
  { id: 'S5', weeks: '第 7 周', title: '框架集成', layer: 'upper', goal: 'DDP / FSDP / TP / EP / PD 分离各调用哪些集合操作，推理框架为何自带 custom allreduce', link: '/notes/05-framework-integration/README' },
  { id: 'S6', weeks: '第 8 周', title: '通信与计算融合', layer: 'upper', goal: '对称内存、signal/wait 的 tile 级融合，读懂一个 AllGather+GEMM kernel', link: '/notes/06-comm-compute-fusion/README' },
  { id: 'S7', weeks: '第 9 周', title: '性能调优与排查', layer: 'practice', goal: '网卡 → 库 → 框架的分层排查，定位慢、hang、带宽不达标', link: '/notes/07-perf-troubleshooting/README' },
  { id: 'S8', weeks: '第 10 周', title: '面试准备', layer: 'practice', goal: '题库全部能口述，准备 2–3 个可讲 5 分钟的项目表述', link: '/notes/08-interview/README' },
]
</script>

<template>
  <section class="rc">
    <h2 class="rc-title">10 周路线</h2>
    <p class="rc-sub">按周推进；每阶段结束回到 JD 拆解更新自评。完整说明见 <a :href="withBase('/notes/00-roadmap/03-learning-path')">学习路线</a>。</p>
    <div class="rc-grid">
      <a v-for="s in stages" :key="s.id" class="rc-card" :class="`rc-${s.layer}`" :href="withBase(s.link)">
        <div class="rc-head"><span class="rc-id">{{ s.id }}</span><span class="rc-weeks">{{ s.weeks }}</span></div>
        <div class="rc-name">{{ s.title }}</div>
        <p class="rc-goal">{{ s.goal }}</p>
      </a>
    </div>
  </section>
</template>

<style scoped>
.rc { max-width: 1152px; margin: 56px auto 0; }
.rc-title { font-size: 22px; font-weight: 600; margin: 0; border: 0; padding: 0; }
.rc-sub { color: var(--vp-c-text-2); font-size: 14px; margin: 6px 0 20px; }
.rc-sub a { color: var(--vp-c-brand-1); }
.rc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; }
.rc-card {
  display: block; text-decoration: none; color: inherit;
  border: 1px solid var(--vp-c-divider); border-top: 3px solid var(--layer-color);
  border-radius: 6px; padding: 14px 16px 12px; background: var(--vp-c-bg-soft);
  transition: transform .15s ease, box-shadow .15s ease;
}
.rc-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0, 0, 0, .06); }
.rc-upper { --layer-color: var(--ccl-upper); }
.rc-middle { --layer-color: var(--ccl-middle); }
.rc-lower { --layer-color: var(--ccl-lower); }
.rc-practice { --layer-color: var(--ccl-practice); }
.rc-head { display: flex; justify-content: space-between; align-items: baseline; }
.rc-id { font-family: var(--vp-font-family-mono); font-size: 13px; font-weight: 600; color: var(--layer-color); }
.rc-weeks { font-size: 12px; color: var(--vp-c-text-3); }
.rc-name { font-weight: 600; margin: 6px 0 4px; font-size: 16px; }
.rc-goal { margin: 0; font-size: 13px; line-height: 1.6; color: var(--vp-c-text-2); font-family: var(--ccl-font-serif); }
</style>
