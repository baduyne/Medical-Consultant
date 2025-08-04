import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Rag.rag_pipeline import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import torch

fine_tuned_model = "baduyne/vnt5-medical-gqa" # mô hình được fine tuning trước đó 

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)  # hoặc model gốc bạn dùng để fine-tune
    model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model)
    # Padding config
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer


def get_response(model,tokenizer, question):
    
    context = search_redis(question)
    if len(context) == 0:
        return "Xin lỗi! Câu hỏi bạn nằm ngoài sự hiểu biết của tôi."
    with torch.no_grad():
        # Chuẩn bị input
        inputs = tokenizer(f"question: {question} context: {context}", return_tensors="pt", max_length=256, truncation=True)
        # Sinh văn bản
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
