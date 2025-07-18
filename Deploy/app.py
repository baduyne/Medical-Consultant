from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Rag.redis_client import *


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import torch

def get_response(question):
    quantized_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./Model/vnt5-qa-final")

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "PhucDanh/vit5-fine-tuning-for-question-answering",  # hoặc model gốc
        # device_map="auto",
        # quantization_config=quantized_config
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, "./Model/vnt5-qa-final")
    model.eval()
    context = search_redis(question)
    input_text = f"Answer the question: {question} Context: {context}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=2048)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


app = FastAPI()

# Mount static folders
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Route: Trang chính
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route: API xử lý câu hỏi từ frontend
@app.post("/chat")
async def chat_response(message: str = Form(...)):
    response = get_response(message)
    return JSONResponse({"reply": response})
