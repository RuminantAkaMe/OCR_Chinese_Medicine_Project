<template>
  <v-app>
    <v-main>
      <v-container>
        <v-snackbar v-model="store.snackbar.visible" :timeout="3000" color="primary">
          {{ store.snackbar.message }}
        </v-snackbar>
        <!-- Tabs Navigation -->
        <v-tabs v-model="tab" background-color="primary" dark grow>
          <v-tab to="/">Home</v-tab>
          <v-tab to="/about">About</v-tab>
        </v-tabs>

        <!-- Page Content -->
        <span style="color: red; font-weight: normal;">Only for representative purpose, result are pre-computed.</span>
        <router-view />
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

const route = useRoute()
const tab = ref(route.path)

watch(
  () => route.path,
  (newPath) => {
    tab.value = newPath
  },
)
</script>
