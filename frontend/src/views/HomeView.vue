<template>
  <v-container class="py-6">
    <UploadPanel @upload-done="resetActiveStep" />

    <v-row class="mt-6" dense>
      <v-col cols="12" md="6">
        <!-- 📋 Grouped operations panel -->
        <v-card class="pa-4" elevation="3">
          <v-card-title class="text-h6 mb-2">📋 Processing Steps</v-card-title>

          <v-row dense style="max-height: 60vh; overflow-y: auto;">
            <v-col cols="12">
              <PreprocessingPanel :disabled="!isOperationEnabled(0)" @operation-done="markOperationDone(0)" />
            </v-col>
            <v-col cols="12">
              <CharacterDetectionPanel :disabled="!isOperationEnabled(1)" @operation-done="markOperationDone(1)" />
            </v-col>
            <v-col cols="12">
              <CharacterSegmentationPanel :disabled="!isOperationEnabled(2)" @operation-done="markOperationDone(2)" />
            </v-col>
            <v-col cols="12">
              <CharacterRecognitionPanel :disabled="!isOperationEnabled(3)" @operation-done="markOperationDone(3)" />
            </v-col>
            <v-col cols="12">
              <WordRecognitionPanel :disabled="!isOperationEnabled(4)" @operation-done="markOperationDone(4)" />
            </v-col>
            <v-col cols="12">
              <PdfCreationPanel :disabled="!isOperationEnabled(5)" />
            </v-col>
          </v-row>
        </v-card>

        <v-row class="mt-6">
          <v-col>
            <ControlPanel  @reset-done="resetActiveStep"/>
          </v-col>
          <v-col>
            <DownloadPanel />
          </v-col>
        </v-row>
      </v-col>

      <v-col cols="12" md="6">
        <PreviewPanel />
        <!-- 🗾 Preview always on the right -->
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import UploadPanel from '@/components/UploadPanel.vue'
import DownloadPanel from '@/components/DownloadPanel.vue'
import PreprocessingPanel from '@/components/PreprocessingPanel.vue'
import CharacterDetectionPanel from '@/components/CharacterDetectionPanel.vue'
import CharacterSegmentationPanel from '@/components/CharacterSegmentationPanel.vue'
import CharacterRecognitionPanel from '@/components/CharacterRecognitionPanel.vue'
import PdfCreationPanel from '@/components/PdfCreationPanel.vue'
import PreviewPanel from '@/components/PreviewPanel.vue'
import ControlPanel from '@/components/ControlPanel.vue'
import WordRecognitionPanel from '@/components/WordRecognitionPanel.vue'

/**
 * 🏠 HomeView.vue
 *
 * Main view of the application.
 *
 * Components:
 * - UploadPanel: Upload image.
 * - Processing Steps (grouped in a card):
 *   - PreprocessingPanel
 *   - CharacterDetectionPanel
 *   - CharacterSegmentationPanel
 *   - CharacterRecognitionPanel
 *   - PdfCreationPanel
 * - ControlPanel: Reset to original file.
 * - DownloadPanel: Download current processed file.
 * - PreviewPanel: Show live preview.
 *
 * Behavior:
 * - Only one processing step active at a time (controlled by activeStep).
 * - After each successful operation, the next step becomes active.
 * - Uploading or resetting resets the workflow.
 */

// Tracks which processing step is currently active (0 = Preprocessing, 1 = Detection, etc.)
const activeStep = ref(0)

/**
 * Checks if a given step is currently enabled (active).
 * @param index - Step index
 * @returns boolean - true if the panel should be enabled
 */
function isOperationEnabled(index: number) {
  return activeStep.value === index
}

/**
 * Advances to the next step after a successful operation.
 * @param index - Step index that was completed
 */
function markOperationDone(index: number) {
  if (activeStep.value === index) {
    activeStep.value++
  }
}

/**
 * Resets the active step to the beginning (after upload or reset).
 */
function resetActiveStep() {
  activeStep.value = 0
}

</script>
