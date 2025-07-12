# Load model directly
from transformers import AutoTokenizer, AutoModel, from

tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-word")
model = AutoModel.from_pretrained("demdecuong/vihealthbert-base-word")
