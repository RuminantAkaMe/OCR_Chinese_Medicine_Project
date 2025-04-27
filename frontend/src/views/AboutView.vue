<template>
  <v-container class="py-10">
    <v-row justify="center">
      <v-col cols="12" md="8">
        <v-card class="pa-6" elevation="2">
          <v-card-title class="text-h5 mb-4">Über diese Anwendung</v-card-title>
          <v-card-text>
            <p>
              Diese Anwendung demonstriert den Upload und die Bearbeitung von Bildern im Browser mit
              einem FastAPI-Backend und einem modernen Vuetify-Frontend.
            </p>

            <p>
              Du kannst Bilder hochladen, bearbeiten (z. B. drehen, verkleinern, in Graustufen
              umwandeln) und das Ergebnis sofort ansehen und herunterladen.
            </p>

            <p class="mt-4">
              <strong>Technologien:</strong> Vue 3, TypeScript, Pinia, Vuetify, FastAPI, Pillow
            </p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>
