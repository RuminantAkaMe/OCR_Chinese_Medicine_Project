<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>🖼 Preview</v-card-title>
    <v-card-text>
      <!-- No file available -->
      <div v-if="!store.processedFilename" class="text-grey">
        No file available.
      </div>

      <!-- Image preview -->
      <div v-else-if="currentType === 'image'">
        <img
          :src="fileUrl"
          alt="Preview"
          style="max-width: 100%; max-height: 400px; border: 1px solid #ccc"
        />
      </div>

      <!-- PDF preview -->
      <div v-else-if="currentType === 'pdf'">
        <iframe
          :src="fileUrl"
          style="width: 100%; height: 400px; border: 1px solid #ccc"
        ></iframe>
      </div>

      <!-- JSON preview -->
      <div v-else-if="currentType === 'json'">
        <v-code style="max-height: 400px; overflow-y: auto">
          {{ jsonContent }}
        </v-code>
      </div>

      <!-- Unsupported format -->
      <div v-else class="text-grey">
        Unsupported file type.
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 🖼 PreviewPanel.vue
 *
 * Displays a preview of the currently processed file.
 * Supports:
 * - Images (.png, .jpg, etc.): rendered via <img>
 * - PDFs (.pdf): rendered via <iframe>
 * - JSON (.json): loaded and shown as formatted text
 * - Other: fallback message
 */

import { computed, ref, watch } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

// Build file URL
const fileUrl = computed(() =>
  store.processedFilename
    ? `http://localhost:8000/api/preview/${store.processedFilename}`
    : '',
)

// Track file type explicitly
const currentType = ref<'image' | 'pdf' | 'json' | 'other'>('other')

watch(
  () => store.processedFilename,
  (filename) => {
    if (!filename) {
      currentType.value = 'other'
      return
    }

    const ext = filename.split('.').pop()?.toLowerCase()
    console.log(ext)
    if (!ext) {
      currentType.value = 'other'
    } else if (['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'].includes(ext)) {
      currentType.value = 'image'
    } else if (ext === 'pdf') {
      currentType.value = 'pdf'
    } else if (ext === 'json') {
      currentType.value = 'json'
    } else {
      currentType.value = 'other'
    }
  },
  { immediate: true }
)

// Load JSON content when needed
const jsonContent = ref('')
watch(fileUrl, async (url) => {
  if (currentType.value === 'json' && url) {
    try {
      const res = await fetch(url)
      const data = await res.json()
      jsonContent.value = JSON.stringify(data, null, 2)
    } catch (err) {
      jsonContent.value = `⚠️ Failed to load JSON: ${err}`
    }
  } else {
    jsonContent.value = ''
  }
})
</script>

<style scoped>
v-code {
  font-family: monospace;
  white-space: pre-wrap;
  background-color: #f5f5f5;
  padding: 8px;
  border-radius: 4px;
  display: block;
}
</style>
