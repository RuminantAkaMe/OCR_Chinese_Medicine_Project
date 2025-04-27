<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>4. Character Recognition (OCR)</v-card-title>
    <v-alert v-if="disabled" type="warning" dense class="pa-1 text-caption">
      Please complete Character Segmentation first.
    </v-alert>


    <v-card-text>
      <v-btn :disabled="!store.selectedFile || disabled" @click="recognize"> Recognize Characters </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * CharacterRecognitionPanel.vue
 *
 * Recognizes segmented characters (OCR).
 * Sends a request to the backend (/api/recognize).
 * Emits `operation-done` event on success.
 */

import { useUploadStore } from '@/stores/uploadStore'
import { defineEmits, defineProps } from 'vue'

const store = useUploadStore()
const emit = defineEmits(['operation-done'])

const props = defineProps<{
  disabled: boolean
}>()

const recognize = async () => {
  if (!store.selectedFile) return

  const response = await fetch('http://localhost:8000/api/recognize', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('Character Recognition applied')
    emit('operation-done')
  } else {
    store.showSnackbar('Character Recognition failed')
  }
}
</script>
