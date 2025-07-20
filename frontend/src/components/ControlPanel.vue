<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>⚙ Options</v-card-title>
    <v-card-text>
      <v-btn color="error" @click="reset" :disabled="!store.processedFilename">
        Reset
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * ⚙️ ControlPanel.vue
 *
 * Displays the reset button to revert the image processing.
 * Calls the backend (/api/reset) and shows the original image again.
 *
 * State:
 * - uses `store.processedFilename`
 * - calls `store.showSnackbar()` on success
 */

import { useUploadStore } from '@/stores/uploadStore'
import { defineEmits } from 'vue'

const emit = defineEmits(['reset-done'])

const store = useUploadStore()

const reset = async () => {
  const response = await fetch('http://localhost:8000/api/reset', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('Reset – Original restored')
    emit('reset-done')
  } else {
    store.setProcessedFilename(null)
    store.showSnackbar('No original image avaiable')
  }
}
</script>
