from __future__ import annotations

from app.models.schemas import AskResponse, Source
from app.rag.generator import GroundedGenerator
from app.rag.retriever import Retriever


class StaticRagPipeline:
    """Phase 1 path: always route to static retrieval. Phase 2 adds a classifier."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: GroundedGenerator | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.generator = generator or GroundedGenerator()

    def ask(self, question: str) -> AskResponse:
        matches = self.retriever.retrieve(question)
        answer = self.generator.generate(question, matches)
        sources = _dedupe_sources(matches)
        insufficient = len(matches) == 0
        return AskResponse(
            answer=answer,
            sources=sources,
            route='static',
            retrieved_chunk_ids=[m['id'] for m in matches],
            insufficient_context=insufficient,
        )


def _dedupe_sources(matches: list[dict]) -> list[Source]:
    seen: set[str] = set()
    sources: list[Source] = []
    n = 1
    for match in matches:
        meta = match.get('metadata') or {}
        key = f"{meta.get('title', '')}|{meta.get('url', '')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                n=n,
                title=str(meta.get('title') or 'Untitled'),
                url=str(meta.get('url') or ''),
                category=str(meta.get('category') or ''),
                source=str(meta.get('source') or ''),
            )
        )
        n += 1
    return sources
