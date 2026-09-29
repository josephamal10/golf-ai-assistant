from __future__ import annotations

from typing import Any, Protocol
import logging

import anthropic
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.rag.errors import RateLimitedError


logger = logging.getLogger(__name__)


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

GEMINI_URL = (
    'https://generativelanguage.googleapis.com/v1beta/models/'
    '{model}:generateContent'
)


def _should_retry_gemini(exc: BaseException) -> bool:
    """Retry brief transport/5xx failures. Skip timeouts and 429s — those already
    consumed the Streamlit wait budget or will keep failing under quota."""
    if isinstance(exc, httpx.TimeoutException):
        return False
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 500, 502, 503, 504}
    return False


def _gemini_visible_text(data: dict[str, Any]) -> tuple[str, str | None]:
    candidates = data.get('candidates') or []
    if not candidates:
        raise RuntimeError(f'Gemini returned no candidates: {data}')
    candidate = candidates[0]
    finish = candidate.get('finishReason')
    parts = candidate.get('content', {}).get('parts') or []
    text = ''.join(
        part.get('text', '')
        for part in parts
        if not part.get('thought')
    ).strip()
    return text, finish


def group_by_source(matches: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group retrieved chunk metadata by source document, best-scoring source first.

    A group's 1-based position is its citation number, both in the prompt context
    and in the sources list returned to the user, so the two always agree.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        meta = match.get('metadata') or {}
        key = f"{meta.get('title', '')}|{meta.get('url', '')}"
        groups.setdefault(key, []).append(meta)
    return list(groups.values())


def format_context(matches: list[dict[str, Any]]) -> str:
    blocks = []
    for n, chunks in enumerate(group_by_source(matches), start=1):
        meta = chunks[0]
        title = meta.get('title', 'Untitled')
        url = meta.get('url', '')
        category = meta.get('category', '')
        passages = '\n\n'.join(
            f"Section: {chunk['section']}\n{chunk.get('text', '')}"
            if chunk.get('section') else chunk.get('text', '')
            for chunk in chunks
        )
        blocks.append(
            f'[{n}] title={title} | category={category} | url={url}\n{passages}'
        )
    return '\n\n'.join(blocks)


def empty_answer() -> str:
    return (
        'I do not have enough information in the golf knowledge base to '
        'answer that. Try a question about rules, history, courses, '
        'tournaments, players, equipment, or terminology.'
    )


def user_prompt(question: str, matches: list[dict[str, Any]]) -> str:
    return f'Question:\n{question}\n\nRetrieved context:\n{format_context(matches)}'


class Generator(Protocol):
    provider: str
    model: str

    def generate(self, question: str, matches: list[dict[str, Any]]) -> str:
        ...


class GroundedGenerator:
    """Claude Sonnet grounded generation. Kept intact for LLM_PROVIDER=claude."""

    provider = 'claude'

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError('ANTHROPIC_API_KEY is missing. Add it to .env')
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_generation_model

    def generate(self, question: str, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return empty_answer()

        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=1200,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        'role': 'user',
                        'content': user_prompt(question, matches),
                    }
                ],
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitedError('Claude rate limit exceeded. Try again later.') from exc
        parts = []
        for block in message.content:
            if getattr(block, 'type', None) == 'text':
                parts.append(block.text)
        return '\n'.join(parts).strip()


class GeminiGroundedGenerator:
    """Gemini grounded generation using the same context + citation prompt."""

    provider = 'gemini'

    def __init__(self) -> None:
        if not settings.google_api_key:
            raise RuntimeError('GOOGLE_API_KEY is missing. Add it to .env')
        self.model = settings.gemini_generation_model
        self._client = httpx.Client(timeout=130.0)

    def generate(self, question: str, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return empty_answer()
        url = GEMINI_URL.format(model=self.model)
        payload = {
            'system_instruction': {
                'parts': [{'text': SYSTEM_PROMPT}],
            },
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': user_prompt(question, matches)}],
                }
            ],
            'generationConfig': {
                'maxOutputTokens': 8192,
                'temperature': 0.2,
                # Thinking shares this budget; uncapped thinking truncates answers.
                'thinkingConfig': {
                    'thinkingLevel': 'minimal',
                },
            },
        }
        data = self._post(url, payload)
        text, finish = _gemini_visible_text(data)
        if finish == 'MAX_TOKENS':
            logger.warning(
                'Gemini hit MAX_TOKENS; retrying with a larger output budget. usage=%s',
                data.get('usageMetadata'),
            )
            payload['generationConfig']['maxOutputTokens'] = 16384
            data = self._post(url, payload)
            text, finish = _gemini_visible_text(data)
        if finish and finish not in {'STOP', 'END_TURN'}:
            logger.warning('Gemini finishReason=%s usage=%s', finish, data.get('usageMetadata'))
        if not text:
            raise RuntimeError(f'Gemini returned an empty answer (finishReason={finish})')
        return text

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception(_should_retry_gemini),
        reraise=True,
    )
    def _post(self, url: str, payload: dict) -> dict:
        response = self._client.post(
            url,
            headers={'x-goog-api-key': settings.google_api_key},
            json=payload,
        )
        if response.status_code == 429:
            raise RateLimitedError('Gemini rate limit or quota exceeded. Try again later.')
        response.raise_for_status()
        return response.json()


def build_generator() -> Generator:
    provider = (settings.llm_provider or 'gemini').strip().lower()
    if provider == 'gemini':
        return GeminiGroundedGenerator()
    if provider == 'claude':
        return GroundedGenerator()
    raise RuntimeError(
        f'Unknown LLM_PROVIDER={provider!r}. Use gemini or claude.'
    )
