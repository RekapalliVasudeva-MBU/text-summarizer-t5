import os
import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Use HuggingFace Inference API (free, no model loading on Render)
HF_API_URL = "https://api-inference.huggingface.co/models/ValtareVasu/text_summarizer"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=BASE_DIR)

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

async def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    
    # Add T5 prefix for summarization
    input_text = "summarize: " + dialogue
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                HF_API_URL,
                json={"inputs": input_text, "parameters": {"max_length": 150, "num_beams": 4}}
            )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", "")
            return str(result)
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = await summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "healthy"}
