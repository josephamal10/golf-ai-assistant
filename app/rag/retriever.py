from __future__ import annotations

from typing import Any

from app.config import settings
from app.rag.embeddings import EmbeddingClient
from app.rag.vectorstore import VectorStore


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingClient()
        self.store = store or VectorStore()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k if top_k is not None else settings.retrieve_top_k
        min_score = min_score if min_score is not None else settings.retrieve_min_score
        query_vector = self.embeddings.embed_query(question)
        metadata_filter = {'category': {'$eq': category}} if category else None
        matches = self.store.query(
            vector=query_vector,
            top_k=top_k,
            filter=metadata_filter,
        )
        return [m for m in matches if m['score'] >= min_score]
