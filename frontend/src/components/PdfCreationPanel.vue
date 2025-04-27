<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>5. PDF Creation</v-card-title>
    <v-alert v-if="disabled" type="warning" dense class="pa-1 text-caption">
      Please complete Character Recognition first.
    </v-alert>


    <v-card-text>
      <v-btn :disabled="!store.selectedFile || disabled" @click="createPdf"> Create PDF </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * PdfCreationPanel.vue
 *
 * Creates an annotated PDF from recognized characters.
 * Sends a request to the backend (/api/create-pdf).
 * No operation-done event needed because this is the last step.
 */

import { useUploadStore } from '@/stores/uploadStore'
import { defineProps } from 'vue'

const store = useUploadStore()

const props = defineProps<{
  disabled: boolean
}>()

const createPdf = async () => {
  if (!store.selectedFile) return

  const response = await fetch('http://localhost:8000/api/create-pdf', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('PDF created')
  } else {
    store.showSnackbar('PDF creation failed')
  }
}
</script>
