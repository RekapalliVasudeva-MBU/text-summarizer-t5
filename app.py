import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Load model from HuggingFace Hub
HF_MODEL = "ValtareVasu/text_summarizer"
print("Loading model from HuggingFace Hub:", HF_MODEL)
model = T5ForConditionalGeneration.from_pretrained(HF_MODEL)
tokenizer = T5Tokenizer.from_pretrained(HF_MODEL)
print("Model loaded successfully!")

# device - CPU only for Render (no GPU on free tier)
device = torch.device("cpu")
model.to(device)
print("Using device:", device)

templates = Jinja2Templates(directory=os.path.dirname(os.path.abspath(__file__)))

class DialogueInput(BaseModel):
     dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

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
