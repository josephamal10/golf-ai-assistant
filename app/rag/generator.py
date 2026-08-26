from __future__ import annotations

from typing import Any

import anthropic

from app.config import settings


SYSTEM_PROMPT = """You are a golf knowledge assistant for a production RAG system.

Answer using ONLY the retrieved context below. Do not use outside knowledge.
If the context is missing, incomplete, or contradictory, say you do not have
enough information in the knowledge base. Never invent rules, scores, dates,
player records, or course facts.

Citations:
- Cite supporting passages with bracket numbers that match the context items,
  e.g. [1] or [1][3].
- Every factual sentence should have at least one citation.
- Do not cite a source you did not use.

Style:
- Be precise and concise.
- Prefer official terminology (R&A / USGA wording when present).
- If the question is about live events, current rankings, or future outcomes,
  say that this static knowledge base cannot answer that and the live/prediction
  layers are not enabled yet.

Never produce your own tournament predictions or betting advice.
"""


def format_context(matches: list[dict[str, Any]]) -> str:
    blocks = []
    for i, match in enumerate(matches, start=1):
        meta = match.get('metadata') or {}
        title = meta.get('title', 'Untitled')
        url = meta.get('url', '')
        category = meta.get('category', '')
        text = meta.get('text', '')
        blocks.append(
            f'[{i}] title={title} | category={category} | url={url}\n{text}'
        )
    return '\n\n'.join(blocks)


class GroundedGenerator:
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError('ANTHROPIC_API_KEY is missing. Add it to .env')
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_generation_model

    def generate(self, question: str, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return (
                'I do not have enough information in the golf knowledge base to '
                'answer that. Try a question about rules, history, courses, '
                'tournaments, players, equipment, or terminology.'
            )

        context = format_context(matches)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f'Question:\n{question}\n\n'
                        f'Retrieved context:\n{context}'
                    ),
                }
            ],
        )
        parts = []
        for block in message.content:
            if getattr(block, 'type', None) == 'text':
                parts.append(block.text)
        return '\n'.join(parts).strip()
