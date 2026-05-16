import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_summary_model")

# Load model from HuggingFace Hub at startup
print("Loading model from HuggingFace Hub...")
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("ValtareVasu/text_summarizer")
tokenizer = T5Tokenizer.from_pretrained("ValtareVasu/text_summarizer")
device = torch.device("cpu")
model.to(device)
model.eval()  # Evaluation mode uses less memory
print("Model loaded successfully!")

templates = Jinja2Templates(directory=BASE_DIR)

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
