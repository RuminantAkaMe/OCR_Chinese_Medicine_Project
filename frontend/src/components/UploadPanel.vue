<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>🡅 Upload file</v-card-title>
    <v-card-text>
      <v-file-input
        label="Select file"
        v-model="selectedFile"
        accept="image/*,.pdf"
        outlined
        dense
        :disabled="loading"
      />
      <v-btn class="mt-3" :disabled="!selectedFile || loading" @click="upload" color="button">
        <template v-if="loading">
          <v-progress-circular indeterminate color="white" size="16" class="me-2" />
          Loading...
        </template>
        <template v-else> Upload </template>
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 📤 UploadPanel.vue
 *
 * Displays a file input field for uploading images.
 * Sends the file to the backend (/api/upload).
 * Shows a snackbar notification upon successful upload and activates the preview.
 *
 * Props: none
 * State: uses `useUploadStore` (selectedFile, processedFilename)
 * Dependencies: Snackbar, FastAPI endpoint
 */

import { ref } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'
import { defineEmits } from 'vue'

const emit = defineEmits(['upload-done'])

const store = useUploadStore()
const selectedFile = ref<File | null>(null)
const loading = ref(false)

const upload = async () => {
  if (!selectedFile.value) return

  loading.value = true

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  const response = await fetch('http://localhost:8000/api/upload', {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()
  loading.value = false

  if (data.filename) {
    store.setFile(selectedFile.value)
    store.setProcessedFilename(data.filename)
    store.showSnackbar('✅ File uploaded successfully')
    emit('upload-done')
  } else {
    store.showSnackbar('❌ Upload failed')
  }
}
</script>
