from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Model.inference import *

app = FastAPI()

class ChatMessage(BaseModel):
    message: str
    
# Setup templates
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def on_startup():
    global model, tokenizer
    model, tokenizer = load_model()


# Route: Trang chính
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route: API xử lý câu hỏi từ frontend
@app.post("/chat")
async def chat_response(chat_message: ChatMessage):
    response = get_response(model, tokenizer, chat_message.message) # get inference from model
    return JSONResponse({"response": response})