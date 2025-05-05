
# 🎙️ Grammar Scoring Engine

An AI-powered web app that scores spoken grammar using both audio and transcribed text. It combines features from speech and text to give a final grammar score via a hybrid deep learning model.

---

## ⚠️ Python Version

**This app is tested and works only on:**

```
Python 3.10
```

Make sure you're using **Python 3.10** before proceeding. Other versions may lead to compatibility issues with packages like `librosa`, `tensorflow`, etc.

---

## 📁 Folder Structure

```
grammer-scroing-engine/
│
├── app/
│   ├── app.py                  # Streamlit app entry point
│   ├── temp.py                 # Helper functions
│   ├── train-model.py          # Training script
│   │
│   ├── data/                   # Must contain dataset files
│   │   ├── audios_train/
│   │   ├── audios_test/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   │
│   ├── models/                 # Saved models & scalers (auto-created)
│   │   ├── hybrid_model.h5
│   │   ├── scaler_audio.pkl
│   │   └── scaler_text.pkl
│
├── tfenv/                      # Virtual environment (excluded from git)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📂 Required Dataset

The project uses this dataset:  
🔗 [SHL Dataset – Grammar Error Audio](https://www.kaggle.com/datasets/saurabhkumargupta23/shl-dataset?resource=download)

➡️ **Download and extract it into this path:**

```
grammer-scroing-engine/app/data/
```

Place the following:
- `train.csv`
- `test.csv`
- `sample_submission.csv`
- Folders `audios_train/` and `audios_test/`

---

## 🚀 Features

- 🧠 Transcription using Whisper model
- 📖 Grammar checking using LanguageTool
- 🔊 Audio features (MFCC, Chroma, Zero-crossing rate)
- 🧮 Scoring via a hybrid neural network
- 🌐 Web interface using Streamlit

---

## ✅ Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/grammer-scroing-engine.git
cd grammer-scroing-engine
```

### 2. Set Up Virtual Environment

```bash
python -m venv tfenv
.	fenv\Scriptsctivate      # On Windows
# source tfenv/bin/activate  # On macOS/Linux
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app/app.py
```

---

## 🧠 Train the Model

If you want to retrain the model from scratch:

```bash
python app/train-model.py
```

This will generate:
- `models/hybrid_model.h5`
- `models/scaler_audio.pkl`
- `models/scaler_text.pkl`

---

## 🧾 .gitignore (important)

```gitignore
tfenv/
app/data/
app/models/
```

These directories are ignored in version control to avoid uploading large data/models.

---

## 📜 License

MIT License © 2025 [Your Name]
