<template>
  <v-app-bar flat color="secondary" dark elevation="3">
    <v-container class="d-flex align-center justify-space-between">
      <div class="d-flex align-center">
        <v-img :src="logoUrl" alt="Logo" max-height="50" width="auto" />
        <span class="text-h6 font-weight-bold">Recognition in Veterinary Chinese Medicine</span>
      </div>
      <span class="text-caption">by BIMAP P1 🧙‍♂️</span>
    </v-container>
  </v-app-bar>
</template>

<script setup lang="ts">
const logoUrl = new URL('../assets/FAU-Logo.png', import.meta.url).href
</script>
