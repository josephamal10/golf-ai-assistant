from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.config import settings


logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError('PINECONE_API_KEY is missing. Add it to .env')
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._ensure_index()
        self._index = self._pc.Index(self.index_name)

    def _list_index_names(self) -> set[str]:
        listing = self._pc.list_indexes()
        names = getattr(listing, 'names', None)
        if callable(names):
            return set(names())
        resolved: set[str] = set()
        for item in list(listing):
            if isinstance(item, str):
                resolved.add(item)
            elif isinstance(item, dict):
                resolved.add(item['name'])
            else:
                resolved.add(item.name)
        return resolved

    def _ensure_index(self) -> None:
        if self.index_name in self._list_index_names():
            return
        logger.info('Creating Pinecone index %s', self.index_name)
        self._pc.create_index(
            name=self.index_name,
            dimension=settings.voyage_embed_dim,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    def upsert(self, vectors: list[dict[str, Any]], namespace: str = 'static') -> None:
        if not vectors:
            return
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start + batch_size]
            self._index.upsert(vectors=batch, namespace=namespace)
            logger.info(
                'Upserted %s / %s vectors',
                min(start + batch_size, len(vectors)),
                len(vectors),
            )

    def query(
        self,
        vector: list[float],
        top_k: int,
        namespace: str = 'static',
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        result = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=True,
        )
        matches = []
        for match in result.matches or []:
            matches.append({
                'id': match.id,
                'score': float(match.score or 0.0),
                'metadata': dict(match.metadata or {}),
            })
        return matches
