from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.rag.errors import RateLimitedError


logger = logging.getLogger(__name__)
JINA_EMBED_URL = 'https://api.jina.ai/v1/embeddings'


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    return False


class EmbeddingClient:
    def __init__(self) -> None:
        if not settings.jina_api_key:
            raise RuntimeError('JINA_API_KEY is missing. Add it to .env')
        self.model = settings.jina_embed_model
        self.dim = settings.jina_embed_dim
        self._client = httpx.Client(
            timeout=120.0,
            headers={
                'Authorization': f'Bearer {settings.jina_api_key}',
                'Content-Type': 'application/json',
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_or_rate_limited(texts, task='retrieval.passage')

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_or_rate_limited([text], task='retrieval.query')
        return vectors[0]

    def _embed_or_rate_limited(self, texts: list[str], task: str) -> list[list[float]]:
        try:
            return self._embed(texts, task=task)
        except httpx.HTTPStatusError as exc:
            # 429s are retried inside _embed; this is only reached once retries run out.
            if exc.response.status_code == 429:
                raise RateLimitedError('Jina embedding rate limit exceeded. Try again later.') from exc
            raise

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            'model': self.model,
            'task': task,
            'input': texts,
            'dimensions': self.dim,
        }
        response = self._client.post(JINA_EMBED_URL, json=payload)
        response.raise_for_status()
        data = response.json().get('data') or []
        data = sorted(data, key=lambda item: item.get('index', 0))
        vectors = [item['embedding'] for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError(f'Jina returned {len(vectors)} vectors for {len(texts)} texts')
        if vectors and len(vectors[0]) != self.dim:
            raise RuntimeError(
                f'Jina returned dim {len(vectors[0])}, expected {self.dim}'
            )
        logger.debug('Embedded %s texts with %s dim=%s', len(texts), self.model, self.dim)
        return vectors
