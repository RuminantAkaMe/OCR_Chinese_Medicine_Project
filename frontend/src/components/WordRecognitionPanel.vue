<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>4. Word Recognition</v-card-title>
    <v-alert v-if="disabled" type="warning" dense class="pa-1 text-caption">
      Please complete Character Recognition first.
    </v-alert>


    <v-card-text>
      <v-btn :disabled="!store.selectedFile || disabled" @click="recognize" color="button"> Recognize Words </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * WordRecognitionPanel.vue
 *
 * Recognizes Words our of sequence of characters.
 * Sends a request to the backend (/api/recognize_words).
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

  const response = await fetch('http://localhost:8000/api/recognize_words', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('Word Recognition applied')
    emit('operation-done')
  } else {
    store.showSnackbar('Word Recognition failed')
  }
}
</script>
