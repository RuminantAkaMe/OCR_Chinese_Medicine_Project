# 🖼️ Bildbearbeitungs-Webapp mit Vue & FastAPI

Ein moderner Webservice zum Hochladen, Bearbeiten und Herunterladen von Bildern.  
Das Projekt nutzt ein Vuetify-Frontend mit Vue 3 & Pinia und ein FastAPI-Backend mit Pillow zur Bildverarbeitung.

---

## 🚀 Features

- 📤 Bild-Upload im Frontend
- 🧠 Bildverarbeitung im Backend (z. B. drehen, verkleinern, Graustufen)
- 📥 Ergebnis sofort im Browser anzeigen & herunterladen
- 🔄 Operationen können nacheinander angewendet werden
- 🧹 Zurücksetzen-Funktion (Reset auf Original)
- 🖼 Live-Vorschau des bearbeiteten Bildes
- 🔔 Snackbar-Feedback bei allen Aktionen

---

## 🛠️ Technologien

- Frontend Vue 3, TypeScript, Pinia, Vuetify, Vite
- Backend FastAPI, Python, Pillow (PIL)
- Weitere Tools Git, Node.js, pip, virtualenv

---

## 📂 Projektstruktur

```plaintext
Template_Project
├── frontend              # Vue + Vuetify App
│   └── src               # Komponenten, Views, Store
├── backend               # FastAPI Backend
│   ├── app
│   │   ├── main.py        # API-Endpunkte
│   │   └── processing.py  # Bildbearbeitungslogik
│   └── uploaded_files    # Temporär gespeicherte Dateien (wird ignoriert)
└── README.md              # Dieses Dokument
