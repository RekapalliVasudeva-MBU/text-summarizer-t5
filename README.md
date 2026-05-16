# 🤖 AI Text Summarizer

A text summarization web app built with **FastAPI** and **HuggingFace Transformers (T5)**. Paste any long text and get a concise summary.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App

**Option A: Use local model (default)**
```bash
python app.py
```

**Option B: Set custom model path**
```bash
# Windows
set MODEL_PATH=C:\path\to\your\model
python app.py

# Linux/Mac
export MODEL_PATH=/path/to/your/model
python app.py
```

**Option C: Download from HuggingFace Hub (auto-fallback)**
If no local model is found, the app automatically downloads from:
```
ValtareVasu/text_summarizer
```

### 3. Open Browser
Go to: **http://localhost:8000**

Paste text → Click **Summarize** → Get summary!

## 📁 Project Structure
```
text_summarizer/
├── app.py                 # FastAPI backend
├── index.html             # Frontend UI
├── requirements.txt       # Python dependencies
├── saved_summary_model/   # Trained T5 model files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── README.md
```

## 🧠 Model
- **Base Model:** T5-Small
- **Fine-tuned on:** SAMSum dataset (4000 dialogues, 6 epochs)
- **Task:** Abstractive text summarization

## 🛠️ Tech Stack
- Python
- FastAPI
- HuggingFace Transformers
- PyTorch
- HTML/CSS/JavaScript
