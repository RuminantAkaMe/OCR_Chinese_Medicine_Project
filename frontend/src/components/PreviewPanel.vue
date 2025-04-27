<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>🖼 Preview</v-card-title>
    <v-card-text>
      <div v-if="imageUrl">
        <img
          :src="imageUrl"
          alt="Preview"
          style="max-width: 100%; max-height: 400px; border: 1px solid #ccc"
        />
      </div>
      <div v-else class="text-grey">No image avaiable.</div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 🖼 PreviewPanel.vue
 *
 * Displays a live preview of the currently processed image.
 * Uses `processedFilename` from the UploadStore to load the image via `/api/download/<filename>`.
 *
 * Display:
 * - No image: text hint
 * - Image: `<img>` element with a fixed maximum size
 *
 * Reactive:
 * - Automatically updates when `processedFilename` changes
 */

import { computed } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

const imageUrl = computed(() =>
  store.processedFilename ? `http://localhost:8000/api/download/${store.processedFilename}` : '',
)
</script>
