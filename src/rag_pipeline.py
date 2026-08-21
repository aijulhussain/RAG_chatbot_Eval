from src.reranker import RerankingRetriever
from src.generator import generate


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    
    rag = RagPipeline()
    result = rag.invoke("what is drift and why does it matter after deployment?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")