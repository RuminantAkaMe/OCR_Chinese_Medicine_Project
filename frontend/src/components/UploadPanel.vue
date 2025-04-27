<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>📤 Datei hochladen</v-card-title>
    <v-card-text>
      <v-file-input
        label="Datei auswählen"
        v-model="selectedFile"
        accept="image/*"
        outlined
        dense
        :disabled="loading"
      />
      <v-btn class="mt-3" :disabled="!selectedFile || loading" @click="upload">
        <template v-if="loading">
          <v-progress-circular indeterminate color="white" size="16" class="me-2" />
          Lädt...
        </template>
        <template v-else> Hochladen </template>
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 📤 UploadPanel.vue
 *
 * Zeigt ein File-Input-Feld für den Upload von Bildern.
 * Sendet die Datei an das Backend (/api/upload).
 * Zeigt nach erfolgreichem Upload ein Snackbar-Feedback und aktiviert die Vorschau.
 *
 * Props: keine
 * Zustand: verwendet `useUploadStore` (selectedFile, processedFilename)
 * Abhängigkeiten: Snackbar, FastAPI-Endpoint
 */

import { ref } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'

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
    store.showSnackbar('✅ Datei erfolgreich hochgeladen')
  } else {
    store.showSnackbar('❌ Upload fehlgeschlagen')
  }
}
</script>
