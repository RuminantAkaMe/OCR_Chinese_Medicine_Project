<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>3. Character Segmentation</v-card-title>
    <v-alert v-if="disabled" type="warning" dense class="pa-1 text-caption">
      Please complete Character Detection first.
    </v-alert>


    <v-card-text>
      <v-btn :disabled="!store.selectedFile || disabled" @click="segment"> Segment Characters </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * CharacterSegmentationPanel.vue
 *
 * Segments detected characters from the uploaded file.
 * Sends a request to the backend (/api/segment).
 * Emits `operation-done` event on success.
 */

import { useUploadStore } from '@/stores/uploadStore'
import { defineEmits, defineProps } from 'vue'

const store = useUploadStore()
const emit = defineEmits(['operation-done'])

const props = defineProps<{
  disabled: boolean
}>()

const segment = async () => {
  if (!store.selectedFile) return

  const response = await fetch('http://localhost:8000/api/segment', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('Character Segmentation applied')
    emit('operation-done')
  } else {
    store.showSnackbar('Character Segmentation failed')
  }
}
</script>
