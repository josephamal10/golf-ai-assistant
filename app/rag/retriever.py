from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import settings
from app.rag.embeddings import EmbeddingClient
from app.rag.reranker import Reranker
from app.rag.vectorstore import VectorStore


logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingClient | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingClient()
        self.store = store or VectorStore()
        self.reranker = reranker or (Reranker() if settings.rerank_enabled else None)

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
            top_k=max(top_k, settings.rerank_candidates) if self.reranker else top_k,
            filter=metadata_filter,
        )
        matches = [m for m in matches if m['score'] >= min_score]
        if self.reranker and len(matches) > 1:
            try:
                reranked = self.reranker.rerank(question, matches, top_n=top_k)
            except (httpx.HTTPError, RuntimeError) as exc:
                # Reranking only improves the order; vector order still answers the question.
                logger.warning('Rerank failed, using vector order: %s', exc)
            else:
                # Gate on the best passage only: multi-part answers need weaker supporting ones.
                if reranked[0].get('rerank_score', 1.0) < settings.rerank_min_score:
                    logger.info(
                        'Best rerank score %.3f < %.2f; no relevant context for %r',
                        reranked[0]['rerank_score'], settings.rerank_min_score, question,
                    )
                    return []
                return reranked
        return matches[:top_k]
