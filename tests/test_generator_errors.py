import httpx
import pytest
from tenacity import wait_none

from app.rag import generator as gen
from app.rag.errors import RateLimitedError, UpstreamUnavailableError


MATCH = {'id': 'c1', 'score': 0.9, 'metadata': {'title': 'Golf', 'url': 'u', 'text': 'Golf has 18 holes.'}}


def _gemini_returning(monkeypatch, status: int, body: dict) -> tuple[gen.GeminiGroundedGenerator, list]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(gen.settings, 'google_api_key', 'test-key')
    monkeypatch.setattr(gen.GeminiGroundedGenerator._post.retry, 'wait', wait_none())
    generator = gen.GeminiGroundedGenerator()
    generator._client = httpx.Client(transport=httpx.MockTransport(handler))
    return generator, calls


def test_gemini_high_demand_retries_then_raises_clear_error(monkeypatch) -> None:
    body = {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}
    generator, calls = _gemini_returning(monkeypatch, 503, body)
    with pytest.raises(UpstreamUnavailableError, match='high demand'):
        generator.generate('How many holes?', [MATCH])
    assert len(calls) == 4


def test_gemini_quota_is_not_retried(monkeypatch) -> None:
    generator, calls = _gemini_returning(monkeypatch, 429, {'error': {'code': 429}})
    with pytest.raises(RateLimitedError):
        generator.generate('How many holes?', [MATCH])
    assert len(calls) == 1
