from __future__ import annotations

import logging

import voyageai

from app.config import settings


logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self) -> None:
        if not settings.voyage_api_key:
            raise RuntimeError('VOYAGE_API_KEY is missing. Add it to .env')
        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self.model = settings.voyage_embed_model
        self.dim = settings.voyage_embed_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type='document')

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text], input_type='query')
        return vectors[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            output_dimension=self.dim,
        )
        logger.debug('Embedded %s texts with %s', len(texts), self.model)
        return list(result.embeddings)
