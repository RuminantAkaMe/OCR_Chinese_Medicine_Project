import { createApp } from 'vue'
import App from './App.vue'
import { createVuetify } from 'vuetify'
import 'vuetify/styles'
import { aliases, mdi } from 'vuetify/iconsets/mdi'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import { createPinia } from 'pinia'
import router from './router'

const vuetify = createVuetify({
  components,
  directives,
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: { mdi },
  },
  theme: {
    defaultTheme: 'FauTheme',
    themes: {
      FauTheme: {
        dark: false,
        colors: {
          primary: '#8C9FB1',
          secondary: '#D3DDE6',
          surface: '#2F586E',
          background: '#D3DDE6',
          button: '#204251',
          warning: '#204251',
        },
      },
    },
  },
})

createApp(App).use(router).use(vuetify).use(createPinia()).mount('#app')
