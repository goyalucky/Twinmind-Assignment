import numpy as np
import faiss
import os
import pickle
from typing import List

class FaissStore:
    def __init__(self, dim: int, path: str = "faiss_index"):
        self.dim = dim
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        self.index_file = os.path.join(self.path, "index.faiss")
        self.meta_file = os.path.join(self.path, "meta.pkl")

        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self._load()
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadatas = []
            self._save()

    def add(self, vectors: np.ndarray, metadatas: List[dict]):
        self.index.add(vectors.astype(np.float32))
        start_id = len(self.metadatas)
        self.metadatas.extend(metadatas)
        self._save()
        return list(range(start_id, start_id + len(metadatas)))

    def search(self, vector: np.ndarray, top_k: int = 8):
        if self.index.ntotal == 0:
            return [[]]

        D, I = self.index.search(vector.astype(np.float32), top_k)

        results = []
        for dists, ids in zip(D, I):
            row = []
            for d, idx in zip(dists, ids):
                if idx < 0 or idx >= len(self.metadatas):
                    continue
                row.append({
                    "id": idx,
                    "score": float(d),
                    "metadata": self.metadatas[idx]
                })
            results.append(row)
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadatas, f)

    def _load(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "rb") as f:
            self.metadatas = pickle.load(f)
