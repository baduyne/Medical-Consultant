import pandas as pd
import numpy as np
import redis
import json
from transformers import AutoTokenizer, AutoModel
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition
from redis.commands.search.query import Query
import torch

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
model.eval()

r = redis.Redis(host="localhost", port=3107, decode_responses=False)

def search_redis(query_text, top_k=3, similiarity=0.85):
    score_threshold = 1 - similiarity
    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        vec = embedding.detach().cpu().numpy().astype(np.float32).tobytes()

    query_str = f'*=>[KNN {top_k} @embedding $vec AS score]'
    q = Query(query_str)\
        .return_fields("context", "score")\
        .sort_by("score")\
        .dialect(2)

    results = r.ft("doc_index").search(q, query_params={"vec": vec})

    filtered = ""
    for doc in results.docs:
        try:
            score = float(doc.score)
            if score <= score_threshold:
                filtered += doc.context
        except:
            pass

    return filtered
