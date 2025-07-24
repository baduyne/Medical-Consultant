import redis
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rag_pipeline import *
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition
import torch

r = redis.Redis(host="localhost", port=3107, decode_responses=False)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
model.eval() 

def merge_and_dedup_words(x):
    merged = f"{x['title']} {x['question']} {x['context']}"
    words = merged.split()
    seen = set()
    deduped_words = [word for word in words if not (word in seen or seen.add(word))]
    return " ".join(deduped_words)

def save_embedding(full_dataset):
    full_dataset["information"] = full_dataset.apply(merge_and_dedup_words, axis=1)

    dim = 768 
    try:
        r.ft("doc_index").info()
        print("Index 'doc_index' đã tồn tại. Đang xoá và tạo lại...")
        r.ft("doc_index").dropindex(delete_documents=True)
    except:
        print("Tạo index 'doc_index'...")

    r.ft("doc_index").create_index(
        fields=[
            TextField("question"),
            TextField("answer"),
            TextField("context"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": 1000,
                "M": 16,
                "EF_CONSTRUCTION": 200
            })
        ],
        definition=IndexDefinition(prefix=["doc:"])
    )

    for i, row in full_dataset.iterrows():
        text = row["information"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            vector = embedding.detach().numpy().astype(np.float32).tobytes()

        r.hset(f"doc:00{i + 1}", mapping={
            "question": row["question"],
            "answer": row["answer"],
            "context": row["context"],
            "embedding": vector
        })

def main():
    train = pd.read_parquet("./Data/Dataset/train.parquet")
    valid = pd.read_parquet("./Data/Dataset/validation.parquet")
    test = pd.read_parquet("./Data/Dataset/test.parquet")
    full_dataset = pd.concat([train, valid, test])

    save_embedding(full_dataset)

if __name__ == "__main__":
    main()
