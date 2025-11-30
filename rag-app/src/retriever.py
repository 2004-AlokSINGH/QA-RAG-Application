
class Retriever:
    def __init__(self, embedder, db):
        self.embedder = embedder
        self.db = db

    def retrieve(self, query, top_k=3):
        query_emb = self.embedder.embed([query])[0]
        return self.db.search(query_emb, top_k)
