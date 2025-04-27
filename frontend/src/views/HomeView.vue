<template>
  <v-container class="py-6">
    <UploadPanel />

    <v-row dense class="mt-6">
      <v-col cols="12" md="4" v-for="op in operations" :key="op.mode">
        <ImageOperationPanel :mode="op.mode" :label="op.label" />
      </v-col>
    </v-row>

    <v-row class="mt-6" dense>
      <v-col cols="12" md="6">
        <DownloadPanel />
      </v-col>
      <v-col cols="12" md="6">
        <PreviewPanel />
        <!-- 🖼 Vorschau rechts -->
      </v-col>
    </v-row>

    <v-row class="mt-6">
      <v-col>
        <ControlPanel />
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import UploadPanel from '@/components/UploadPanel.vue'
import DownloadPanel from '@/components/DownloadPanel.vue'
import ImageOperationPanel from '@/components/ImageOperationPanel.vue'
import PreviewPanel from '@/components/PreviewPanel.vue'
import ControlPanel from '@/components/ControlPanel.vue'

const operations = [
  { mode: 'resize', label: '📏 Verkleinern' },
  { mode: 'rotate', label: '🔄 Drehen' },
  { mode: 'grayscale', label: '🌑 Graustufen' },
]
</script>
