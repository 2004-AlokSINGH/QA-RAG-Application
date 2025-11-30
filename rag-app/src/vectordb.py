# vectordb.py

import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []  # to return sources

    def add(self, embeddings, chunks):
        """Add embeddings + text chunks."""
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_emb, top_k=3):
        """Return top_k most similar chunks."""
        D, I = self.index.search(np.array([query_emb]).astype("float32"), top_k)
        results = [(self.chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
        return results
