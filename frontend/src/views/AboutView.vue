<template>
  <v-container class="py-10">
    <v-row justify="center">
      <v-col cols="12" md="8">
        <v-card class="pa-6" elevation="2">
          <v-card-title class="text-h5 mb-4">About this Application</v-card-title>
          <v-card-text>
            <p>
              This application demonstrates a complete OCR processing workflow for handwritten Chinese characters
              using a modern frontend and a FastAPI backend.
            </p>

            <p>
              You can upload images and guide them step-by-step through the following processing stages:
              preprocessing, character detection, character segmentation, character recognition (OCR), and PDF generation.
            </p>

            <p>
              Each step is executed individually, with only the next available processing step being active at a time.
            </p>

            <p class="mt-4">
              <strong>Technologies:</strong> Vue 3, TypeScript, Pinia, Vuetify, FastAPI, Pillow
            </p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

