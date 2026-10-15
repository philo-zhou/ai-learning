import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import LayerStrip from './components/LayerStrip.vue'
import RoadmapCards from './components/RoadmapCards.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('LayerStrip', LayerStrip)
    app.component('RoadmapCards', RoadmapCards)
  },
} satisfies Theme
