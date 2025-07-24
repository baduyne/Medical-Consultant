import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Rag.rag_pipeline import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "VietAI/vit5-base"
saved_model_path = "vit5-base-qa-final"

def get_response(question):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_path, device_map="cpu")

    model.eval()
    context = search_redis(question)

    if len(context) == 0:
        return "Xin lỗi! Câu hỏi bạn nằm ngoài sự hiểu biết của tôi."

    input_text = f"question: {question} context: {context}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=2048)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
