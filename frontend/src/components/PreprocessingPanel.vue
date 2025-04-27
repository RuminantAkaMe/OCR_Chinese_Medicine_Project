<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>1. Preprocessing</v-card-title>
    <v-alert v-if="!store.selectedFile" type="warning" dense class="pa-1 text-caption">
      Please upload a file.
    </v-alert>

    <v-card-text>
      <v-btn :disabled="!store.selectedFile || disabled" @click="preprocess">
        Preprocess
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * PreprocessingPanel.vue
 *
 * Preprocesses the uploaded file.
 * Sends a request to the backend (/api/preprocess).
 * Emits `operation-done` event on success.
 * Fully respects HomeView disabled state.
 */

import { useUploadStore } from '@/stores/uploadStore'
import { defineEmits, defineProps } from 'vue'

const store = useUploadStore()
const emit = defineEmits(['operation-done'])

const props = defineProps<{
  disabled: boolean
}>()

const preprocess = async () => {
  if (!store.selectedFile) return

  const response = await fetch('http://localhost:8000/api/preprocess', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('Preprocessing applied')
    emit('operation-done')
  } else {
    store.showSnackbar('Preprocessing failed')
  }
}
</script>



