import pandas as pd
import numpy as np
import redis
import json
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition
from redis.commands.search.query import Query


def merge_and_dedup_words(x):
    merged = f"{x['keyword']} {x['title']} {x['question']}"
    words = merged.split()
    seen = set()
    deduped_words = [word for word in words if not (word in seen or seen.add(word))]
    return " ".join(deduped_words)

def search_redis(query_text, top_k=5, similiarity=0.7):
    score_threshold = 1 - similiarity
    vec = model.encode(query_text).astype(np.float32).tobytes()
    query_str = f'*=>[KNN {top_k} @embedding $vec AS score]'
    
    q = Query(query_str)\
        .return_fields("context", "score")\
        .sort_by("score")\
        .dialect(2)

    results = r.ft("doc_index").search(q, query_params={"vec": vec})

    # Lọc theo ngưỡng score
    filtered = ""
    for doc in results.docs:
        try:
            score = float(doc.score)
            if score <= score_threshold:
                filtered += doc.context
        except:
            pass

    return filtered


if __name__ == "__main__":
    # 1. Load dữ liệu
    train = pd.read_parquet("./Dataset/train-00000-of-00001.parquet")
    valid = pd.read_parquet("./Dataset/validation-00000-of-00001.parquet")
    test = pd.read_parquet("./Dataset/test-00000-of-00001.parquet")
    full_dataset = pd.concat([train, valid, test])

    # 2. Xử lý thông tin tìm kiếm
    full_dataset["information"] = full_dataset.apply(merge_and_dedup_words, axis=1)

    # 3. Khởi tạo model & Redis
    model = SentenceTransformer("all-MiniLM-L6-v2")
    r = redis.Redis(host="localhost", port=6379, decode_responses=False)  # decode_responses=False vì vector là bytes

    # 4. Tạo index nếu chưa có
    dim = 384
    try:
        r.ft("doc_index").info()
        print(" Index 'doc_index' đã tồn tại.")
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

    # 5. Lưu vector vào Redis
    for i, row in full_dataset.iterrows():
        vector = model.encode(row["information"]).astype(np.float32).tobytes()
        r.hset(f"doc:00{i + 1}", mapping={
            "question": row["question"],
            "answer": row["answer"],
            "context": row["context"],
            "embedding": vector
        })
