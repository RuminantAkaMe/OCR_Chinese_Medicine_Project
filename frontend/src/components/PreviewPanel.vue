<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>🖼 Vorschau</v-card-title>
    <v-card-text>
      <div v-if="imageUrl">
        <img
          :src="imageUrl"
          alt="Vorschau"
          style="max-width: 100%; max-height: 400px; border: 1px solid #ccc"
        />
      </div>
      <div v-else class="text-grey">Kein Bild ausgewählt oder verarbeitet.</div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 🖼 PreviewPanel.vue
 *
 * Zeigt eine Live-Vorschau des aktuell bearbeiteten Bildes.
 * Nutzt `processedFilename` aus dem UploadStore, um das Bild über `/api/download/<filename>` anzuzeigen.
 *
 * Anzeige:
 * - Kein Bild: Text-Hinweis
 * - Bild: `<img>`-Element mit festgelegter max. Größe
 *
 * Reaktiv:
 * - Wird automatisch aktualisiert, wenn `processedFilename` sich ändert
 */

import { computed } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

const imageUrl = computed(() =>
  store.processedFilename ? `http://localhost:8000/api/download/${store.processedFilename}` : '',
)
</script>
