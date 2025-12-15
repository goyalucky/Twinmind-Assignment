import numpy as np

EMBED_DIM = 1536
USE_MOCK = True

class Retriever:
    def __init__(self, db_engine, faiss_store):
        self.engine = db_engine
        self.store = faiss_store

    def embed_query(self, query: str):
        return np.zeros((1, EMBED_DIM), dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 6, start_date=None, end_date=None):
        vec = self.embed_query(query)

        results = self.store.search(vec, top_k)

        if not results or not results[0]:
            return []

        contexts = []

        for r in results[0]:
            if not isinstance(r, dict):
                continue

            md = r.get("metadata") or {}

            text = (
                md.get("chunk_text")
                or md.get("text")
                or "No content found"
            )

            contexts.append({
                "text": text,
                "source": md.get("source", "unknown"),
                "score": float(r.get("score", 0.0))
            })

        return contexts
