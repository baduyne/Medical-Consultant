import pandas as pd
import numpy as np
import redis
import json
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition
from redis.commands.search.query import Query

model = SentenceTransformer("all-MiniLM-L6-v2")
r = redis.Redis(host="localhost", port=3107, decode_responses=False)

def search_redis(query_text, top_k=3, similiarity=0.85):
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

