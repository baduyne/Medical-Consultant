from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Rag.rag_pipeline import *
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "VietAI/vit5-base"
saved_model_path = "./vit5-base-qa-final"

def get_response(question):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        # device_map="auto", # Uncomment if you want to use automatic device mapping
        load_in_4bit=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, saved_model_path)
    model.eval()
    context = search_redis(question) # Đảm bảo hàm search_redis hoạt động đúng và trả về dữ liệu

    if len(context) == 0:
        return "Xin lỗi! Câu hỏi bạn nằm ngoài sự hiểu biết của tôi."

    input_text = f"question: {question} context: {context}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=2048)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


app = FastAPI()

class ChatMessage(BaseModel):
    message: str
    
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
async def chat_response(chat_message: ChatMessage):
    # BỎ COMMENT DÒNG NÀY ĐỂ KÍCH HOẠT HỆ THỐNG PHẢN HỒI THỰC SỰ
    # response = get_response(ChatMessage.message)
    # XÓA HOẶC COMMENT DÒNG NÀY VÌ NÓ CHỈ DÙNG ĐỂ THỬ NGHIỆM
    response = f"Bạn đã ghi: {ChatMessage.message}"
    return JSONResponse({"response": response}) # Đổi key từ "reply" sang "response" để khớp với frontend của bạn