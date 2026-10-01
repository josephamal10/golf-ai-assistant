from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.rag.errors import RateLimitedError


logger = logging.getLogger(__name__)
JINA_EMBED_URL = 'https://api.jina.ai/v1/embeddings'
BATCH_TIMEOUT_SECONDS = 120.0                              # ingestion: nobody is waiting
QUERY_TIMEOUT_SECONDS = 20.0                               # a user is waiting on /ask


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
            timeout=BATCH_TIMEOUT_SECONDS,
            headers={
                'Authorization': f'Bearer {settings.jina_api_key}',
                'Content-Type': 'application/json',
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_or_rate_limited(self._embed_batch, texts, task='retrieval.passage')

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_or_rate_limited(self._embed_query, [text], task='retrieval.query')
        return vectors[0]

    def _embed_or_rate_limited(self, embed, texts: list[str], task: str) -> list[list[float]]:
        try:
            return embed(texts, task)
        except httpx.HTTPStatusError as exc:
            # 429s are retried inside embed; this is only reached once retries run out.
            if exc.response.status_code == 429:
                raise RateLimitedError('Jina embedding rate limit exceeded. Try again later.') from exc
            raise

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        return self._embed(texts, task, timeout=BATCH_TIMEOUT_SECONDS)

    @retry(
        # Worst case ~65 s (3 x 20 s + waits), inside the UI's 150 s wait for the whole answer.
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _embed_query(self, texts: list[str], task: str) -> list[list[float]]:
        return self._embed(texts, task, timeout=QUERY_TIMEOUT_SECONDS)

    def _embed(self, texts: list[str], task: str, timeout: float) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            'model': self.model,
            'task': task,
            'input': texts,
            'dimensions': self.dim,
        }
        response = self._client.post(JINA_EMBED_URL, json=payload, timeout=timeout)
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
