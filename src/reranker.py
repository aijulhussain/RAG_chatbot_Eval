from sentence_transformers import CrossEncoder
from src.retriever import load_store

# small, fast, CPU-friendly reranker. Downloads once (~80MB) on first run.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        self.store = load_store()
        self.reranker = CrossEncoder(CROSS_ENCODER)
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK: score each (query, chunk) pair TOGETHER with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]