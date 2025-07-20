<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>🡇 Download Result</v-card-title>
    <v-card-text>
      <v-btn :href="downloadUrl" download :disabled="!store.processedFilename" color="button">
        Download
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 📥 DownloadPanel.vue
 *
 * Displays a button to download the currently processed file.
 * Uses `processedFilename` from the UploadStore to generate the download link.
 *
 * Requirements:
 * - `store.processedFilename` must be set (e.g., after an upload or an operation)
 *
 * Display:
 * - Button is disabled if no file is available
 * - Uses `:href="downloadUrl"` with /api/download/<filename>
 */


import { computed } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

const downloadUrl = computed(() =>
  store.processedFilename ? `http://localhost:8000/api/download/${store.processedFilename}` : '',
)
</script>
