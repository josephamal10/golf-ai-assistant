from __future__ import annotations

import logging

from app.config import settings
from app.models.schemas import AskResponse, Source
from app.rag.generator import Generator, build_generator, drop_invalid_citations, group_by_source
from app.rag.retriever import Retriever
from app.rag.router import FALLBACK_ROUTE, RouteDecision


logger = logging.getLogger(__name__)

OFF_TOPIC_ANSWER = (
    'I am a golf assistant, so I can only help with golf questions. Try asking about '
    'rules, players, tournaments, courses, equipment, or what is happening on tour.'
)
LIVE_UNAVAILABLE_ANSWER = (
    'That needs current information from the web, and live search is not enabled here '
    '(it needs LLM_PROVIDER=gemini and LIVE_SEARCH_ENABLED=true). My knowledge base '
    'covers rules, history, players, courses, and equipment.'
)


class AssistantPipeline:
    """Routes each question: static → knowledge base RAG, live / prediction → web search,
    off_topic → a fixed reply with no LLM call."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        use_router: bool | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.generator = generator or build_generator()
        self.use_router = settings.router_enabled if use_router is None else use_router

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> AskResponse:
        # Follow-ups like "how many majors did he win?" can't be searched or routed as-is,
        # so with history the question is first rewritten to stand alone. One extra LLM call.
        search_query = self.generator.rewrite_question(question, history) if history else question
        decision = self._route(search_query)
        common = {
            'route': decision.route,
            'route_reason': decision.reason,
            'search_query': search_query,
            'provider': getattr(self.generator, 'provider', ''),
            'model': getattr(self.generator, 'model', ''),
        }
        if decision.route == 'off_topic':
            return AskResponse(answer=OFF_TOPIC_ANSWER, sources=[], retrieved_chunk_ids=[], **common)
        if decision.route in ('live', 'prediction'):
            return self._answer_from_web(search_query, decision, common)
        return self._answer_from_knowledge_base(search_query, common)

    def _route(self, question: str) -> RouteDecision:
        if not self.use_router:
            return RouteDecision(FALLBACK_ROUTE)
        try:
            return self.generator.classify(question)
        except Exception:
            # A router failure must not cost the user an answer: Phase 1's path still works.
            logger.warning('Routing failed; answering from the knowledge base', exc_info=True)
            return RouteDecision(FALLBACK_ROUTE, 'router unavailable')

    def _answer_from_knowledge_base(self, search_query: str, common: dict) -> AskResponse:
        matches = self.retriever.retrieve(search_query)
        sources = _dedupe_sources(matches)
        answer = drop_invalid_citations(self.generator.generate(search_query, matches), len(sources))
        return AskResponse(
            answer=answer,
            sources=sources,
            retrieved_chunk_ids=[m['id'] for m in matches],
            insufficient_context=len(matches) == 0,
            **common,
        )

    def _answer_from_web(self, search_query: str, decision: RouteDecision, common: dict) -> AskResponse:
        if not (settings.live_search_enabled and getattr(self.generator, 'supports_live', False)):
            return AskResponse(
                answer=LIVE_UNAVAILABLE_ANSWER, sources=[], retrieved_chunk_ids=[],
                insufficient_context=True, **common,
            )
        live = self.generator.answer_live(search_query, decision.route)
        # Shown as Google returned it: its terms forbid editing grounded answers.
        sources = [
            Source(n=n, title=s['title'], url=s['url'], category='web', source='google_search')
            for n, s in enumerate(live.sources, start=1)
        ]
        return AskResponse(
            answer=live.text,
            sources=sources,
            retrieved_chunk_ids=[],
            insufficient_context=not sources,
            search_suggestions_html=live.search_suggestions_html,
            **common,
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
