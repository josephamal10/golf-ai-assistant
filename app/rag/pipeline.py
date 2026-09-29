from __future__ import annotations

from app.models.schemas import AskResponse, Source
from app.rag.generator import Generator, build_generator, group_by_source
from app.rag.retriever import Retriever


class StaticRagPipeline:
    """Phase 1 path: always route to static retrieval. Phase 2 adds a classifier."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.generator = generator or build_generator()

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
            provider=getattr(self.generator, 'provider', ''),
            model=getattr(self.generator, 'model', ''),
        )


def _dedupe_sources(matches: list[dict]) -> list[Source]:
    # Same grouping as format_context, so source n matches the [n] the model cites.
    sources: list[Source] = []
    for n, chunks in enumerate(group_by_source(matches), start=1):
        meta = chunks[0]
        sources.append(
            Source(
                n=n,
                title=str(meta.get('title') or 'Untitled'),
                url=str(meta.get('url') or ''),
                category=str(meta.get('category') or ''),
                source=str(meta.get('source') or ''),
            )
        )
    return sources
