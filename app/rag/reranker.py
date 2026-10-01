from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.rag.chunker import embedding_text
from app.rag.embeddings import _is_retryable


logger = logging.getLogger(__name__)
JINA_RERANK_URL = 'https://api.jina.ai/v1/rerank'


class Reranker:
    """Re-orders vector-search candidates by how well each passage answers the query.

    Vector search compares two embeddings made separately, one for the question and one
    for the passage. A reranker reads the question and passage together, so it copes
    better with paraphrases ("partners take turns hitting the same ball" -> foursomes).
    """

    def __init__(self) -> None:
        if not settings.jina_api_key:
            raise RuntimeError('JINA_API_KEY is missing. Add it to .env')
        self.model = settings.jina_rerank_model
        self._client = httpx.Client(
            timeout=10.0,
            headers={
                'Authorization': f'Bearer {settings.jina_api_key}',
                'Content-Type': 'application/json',
            },
        )

    def rerank(self, query: str, matches: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if len(matches) <= 1:
            return matches[:top_n]
        payload = {
            'model': self.model,
            'query': query,
            # Same "title > section" header the passages were embedded with.
            'documents': [embedding_text(m.get('metadata') or {}) for m in matches],
            'top_n': top_n,
            'return_documents': False,
        }
        results = self._post(payload).get('results') or []
        reranked = []
        for result in results:
            match = dict(matches[result['index']])
            match['rerank_score'] = float(result['relevance_score'])
            reranked.append(match)
        if not reranked:
            raise RuntimeError('Jina rerank returned no results')
        return reranked

    @retry(
        # In the request path, so a short budget (~21 s worst case): a slow or throttled
        # rerank falls back to vector order instead of holding up the answer.
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(JINA_RERANK_URL, json=payload)
        response.raise_for_status()
        return response.json()
