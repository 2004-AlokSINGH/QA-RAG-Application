# embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en")

    def embed(self, texts):
        """Return list of embeddings."""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
