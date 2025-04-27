<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>⚙️ Optionen</v-card-title>
    <v-card-text>
      <v-btn color="error" @click="reset" :disabled="!store.processedFilename">
        Zurücksetzen
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * ⚙️ ControlPanel.vue
 *
 * Zeigt den Reset-Button, um die Bearbeitung zurückzusetzen.
 * Ruft das Backend (/api/reset) auf und zeigt das Originalbild wieder an.
 *
 * Zustand:
 * - nutzt `store.processedFilename`
 * - ruft `store.showSnackbar()` bei Erfolg
 */

import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

const reset = async () => {
  const response = await fetch('http://localhost:8000/api/reset', {
    method: 'POST',
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.showSnackbar('Zurückgesetzt – Original wiederhergestellt')
  } else {
    store.setProcessedFilename(null)
    store.showSnackbar('Kein Originalbild vorhanden')
  }
}
</script>
