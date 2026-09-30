import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app
from app.rag.errors import RateLimitedError, UpstreamUnavailableError


ASK = {'question': 'What is a bogey?'}
# A real internal message: it embeds the raw provider payload, so it must not be returned.
LEAKY_MESSAGE = "Gemini returned no candidates: {'promptFeedback': {'blockReason': 'SAFETY'}}"


class _FailingPipeline:
    """Stands in for StaticRagPipeline so /ask raises without touching any provider."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def ask(self, question: str, history: list[dict] | None = None) -> None:
        raise self._exc


def _client(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> TestClient:
    monkeypatch.setattr(routes, '_pipeline', _FailingPipeline(exc))
    return TestClient(app, raise_server_exceptions=False)


def test_quota_error_maps_to_429(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = RateLimitedError('Gemini rate limit or quota exceeded. Try again later.')
    response = _client(monkeypatch, exc).post('/ask', json=ASK)

    assert response.status_code == 429
    assert response.json()['detail'] == str(exc)


def test_provider_outage_maps_to_503_with_the_provider_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = UpstreamUnavailableError(
        'Gemini is temporarily unavailable: This model is currently experiencing high demand.'
    )
    response = _client(monkeypatch, exc).post('/ask', json=ASK)

    assert response.status_code == 503
    assert 'high demand' in response.json()['detail']


def test_internal_runtime_error_is_a_generic_500(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a blanket `except RuntimeError` returned 503 with this text in `detail`,
    # exposing raw provider payloads and messages like 'GOOGLE_API_KEY is missing'.
    response = _client(monkeypatch, RuntimeError(LEAKY_MESSAGE)).post('/ask', json=ASK)

    assert response.status_code == 500
    assert response.json() == {'detail': 'Failed to answer question.'}
    assert 'blockReason' not in response.text
