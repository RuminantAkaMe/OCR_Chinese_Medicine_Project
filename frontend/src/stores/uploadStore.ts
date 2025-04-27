import { defineStore } from 'pinia'

export const useUploadStore = defineStore('upload', {
  state: () => ({
    selectedFile: null as File | null,
    processedFilename: null as string | null,
    snackbar: {
      visible: false,
      message: '',
    },
  }),
  actions: {
    setFile(file: File) {
      this.selectedFile = file
      this.processedFilename = null
    },
    setProcessedFilename(name: string | null) {
      this.processedFilename = name
    },
    showSnackbar(message: string) {
      this.snackbar.message = message
      this.snackbar.visible = true
    },
  },
})
