<template>
  <v-card class="pa-4" elevation="3">
    <v-card-title>📥 Ergebnis herunterladen</v-card-title>
    <v-card-text>
      <v-btn :href="downloadUrl" download :disabled="!store.processedFilename">
        Herunterladen
      </v-btn>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
/**
 * 📥 DownloadPanel.vue
 *
 * Zeigt einen Button zum Herunterladen der aktuell bearbeiteten Datei.
 * Verwendet den `processedFilename` aus dem UploadStore, um den Download-Link zu generieren.
 *
 * Voraussetzungen:
 * - `store.processedFilename` muss gesetzt sein (z. B. nach Upload oder Operation)
 *
 * Anzeige:
 * - Button ist deaktiviert, wenn keine Datei verfügbar ist
 * - Nutzt `:href="downloadUrl"` mit /api/download/<filename>
 */

import { computed } from 'vue'
import { useUploadStore } from '@/stores/uploadStore'

const store = useUploadStore()

const downloadUrl = computed(() =>
  store.processedFilename ? `http://localhost:8000/api/download/${store.processedFilename}` : '',
)
</script>
