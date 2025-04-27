<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>{{ label }}</v-card-title>
    <v-alert v-if="!store.selectedFile" type="warning" dense text class="pa-1 text-caption">
      Bitte zuerst eine Datei hochladen.
    </v-alert>

    <v-card-text>
      <v-btn :disabled="!store.selectedFile" @click="apply"> Anwenden </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 🧠 ImageOperationPanel.vue
 *
 * Führt eine einzelne Bildoperation aus (rotate, resize, grayscale).
 * Die Operation wird ans Backend gesendet (/api/operate).
 * Bei Erfolg wird das neue Bild als Vorschau und Download verfügbar gemacht.
 *
 * Props:
 * - mode: string → Modus der Operation
 * - label: string → Anzeigename des Panels
 *
 * Zustand:
 * - verwendet `store.selectedFile`
 * - aktualisiert `store.processedFilename`
 */

import { useUploadStore } from '@/stores/uploadStore'

const props = defineProps<{
  mode: string
  label: string
}>()

const store = useUploadStore()

const apply = async () => {
  if (!store.selectedFile) return

  const formData = new FormData()
  formData.append('mode', props.mode)

  const response = await fetch('http://localhost:8000/api/operate', {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()

  if (data.filename) {
    store.setProcessedFilename(data.filename)
    store.addToHistory(props.mode)
    store.showSnackbar(`Operation "${props.label}" angewendet`)
  } else {
    store.showSnackbar('Verarbeitung fehlgeschlagen')
  }
}
</script>
