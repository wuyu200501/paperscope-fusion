"""Convert Qwen retrieval results to PaperScope's evidence format."""

from pathlib import Path

from .retrieve import DEFAULT_HOST, DEFAULT_MODEL, retrieve

STRATEGIES = {"A": "Structured sections", "B": "Sliding window"}


class QwenRetriever:
    def __init__(self, database: Path, host=DEFAULT_HOST, model=DEFAULT_MODEL):
        if not database.is_dir():
            raise FileNotFoundError(f"Qwen index directory is missing: {database}")
        import chromadb

        self.client = chromadb.PersistentClient(path=str(database))
        self.host = host
        self.model = model

    def retrieve(self, paper_id, question, strategy, top_k=5, rerank_n=0):
        collection = self.client.get_collection(f"qasper_{strategy}")
        hits = retrieve(
            collection, question, k=top_k, model=self.model, host=self.host,
            paper_id=paper_id, rerank_n=rerank_n,
        )
        return [
            {
                "id": hit["chunk_id"],
                "paper_id": hit["meta"]["paper_id"],
                "section": hit["meta"]["section_name"],
                "text": hit["text"],
                "similarity": hit["score"],
                "score_label": "Reranker relevance" if "bi_score" in hit else "Cosine similarity",
            }
            for hit in hits
        ]
