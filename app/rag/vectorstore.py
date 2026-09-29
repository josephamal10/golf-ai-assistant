from __future__ import annotations

import logging
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.config import settings


logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, recreate: bool = False) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError('PINECONE_API_KEY is missing. Add it to .env')
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        if recreate:
            self._delete_index()
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

    def _delete_index(self) -> None:
        if self.index_name not in self._list_index_names():
            logger.info('Pinecone index %s does not exist; nothing to delete', self.index_name)
            return
        logger.info('Deleting Pinecone index %s', self.index_name)
        self._pc.delete_index(self.index_name)
        for _ in range(60):
            if self.index_name not in self._list_index_names():
                logger.info('Deleted Pinecone index %s', self.index_name)
                return
            time.sleep(2)
        raise RuntimeError(f'Timed out waiting for Pinecone index {self.index_name} to delete')

    def _ensure_index(self) -> None:
        if self.index_name in self._list_index_names():
            return
        logger.info(
            'Creating Pinecone index %s dim=%s',
            self.index_name,
            settings.jina_embed_dim,
        )
        self._pc.create_index(
            name=self.index_name,
            dimension=settings.jina_embed_dim,
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

    def stats(self) -> dict[str, Any]:
        result = self._index.describe_index_stats()
        namespaces = getattr(result, 'namespaces', None) or {}
        static = namespaces.get('static') if isinstance(namespaces, dict) else None
        ns_count = None
        if static is not None:
            ns_count = getattr(static, 'vector_count', None)
            if ns_count is None and isinstance(static, dict):
                ns_count = static.get('vector_count')
        return {
            'dimension': getattr(result, 'dimension', None),
            'total_vector_count': getattr(result, 'total_vector_count', None),
            'static_vector_count': ns_count,
        }

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
