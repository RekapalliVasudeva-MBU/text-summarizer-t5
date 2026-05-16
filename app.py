import os
import re
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# ============================================
# APP SETUP
# ============================================
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=BASE_DIR)

# ============================================
# LOAD MODEL
# ============================================
# Use your trained model from HuggingFace Hub
HF_MODEL = "ValtareVasu/text_summarizer"

print("Loading model from HuggingFace Hub...")
model = T5ForConditionalGeneration.from_pretrained(HF_MODEL)
tokenizer = T5Tokenizer.from_pretrained(HF_MODEL)

# Device selection (FIXED: typo is_availanle -> is_available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)
model.eval()  # Evaluation mode for less memory usage
print("Model loaded successfully!")

# ============================================
# INPUT MODEL
# ============================================
class DialogueInput(BaseModel):
    dialogue: str

# ============================================
# TEXT CLEANING
# ============================================
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# ============================================
# SUMMARIZATION
# ============================================
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():  # Saves memory during inference
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# ============================================
# API ENDPOINTS
# ============================================
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
