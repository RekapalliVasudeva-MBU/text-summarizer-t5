import os
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as gr

# ============================================
# CONFIGURATION
# ============================================
# Your trained model from HuggingFace Hub
MODEL_NAME = "ValtareVasu/text-summarizer"

# ============================================
# LOAD MODEL
# ============================================
print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Use GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)
model.eval()
print("Model loaded successfully!")

# ============================================
# TEXT CLEANING
# ============================================
def clean_text(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text

# ============================================
# SUMMARIZATION FUNCTION
# ============================================
def summarize(text):
    if not text or not text.strip():
        return "⚠️ Please enter some text to summarize."

    text = clean_text(text)

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if not summary.strip():
        return "⚠️ Could not generate summary. Try with longer text."

    return summary

# ============================================
# GRADIO UI
# ============================================
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your text here...",
        label="📝 Input Text"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="📋 Summary"
    ),
    title="🤖 AI Text Summarizer",
    description="Paste any long text and get a concise summary using fine-tuned T5 model.",
    examples=[
        ["The Eiffel Tower is 324 metres tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world."],
        ["Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves."],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting AI Text Summarizer...")
    print("="*50)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
