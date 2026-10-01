import httpx
import pytest
from tenacity import wait_none

from app.rag import embeddings as emb
from app.rag.errors import RateLimitedError


def _client_returning(monkeypatch, status: int) -> tuple[emb.EmbeddingClient, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={})

    monkeypatch.setattr(emb.settings, 'jina_api_key', 'test-key')
    for method in (emb.EmbeddingClient._embed_query, emb.EmbeddingClient._embed_batch):
        monkeypatch.setattr(method.retry, 'wait', wait_none())
    client = emb.EmbeddingClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


def test_query_embedding_fails_fast_with_a_short_timeout(monkeypatch) -> None:
    # Regression: a throttled Jina once held a single /ask for ~190 s (120 s timeout x 6 tries).
    client, calls = _client_returning(monkeypatch, 429)
    with pytest.raises(RateLimitedError):
        client.embed_query('Who is Rose Zhang?')
    assert len(calls) == 3
    assert calls[0].extensions['timeout']['read'] == emb.QUERY_TIMEOUT_SECONDS


def test_ingestion_keeps_the_patient_retry_budget(monkeypatch) -> None:
    client, calls = _client_returning(monkeypatch, 429)
    with pytest.raises(RateLimitedError):
        client.embed_documents(['passage one', 'passage two'])
    assert len(calls) == 6
    assert calls[0].extensions['timeout']['read'] == emb.BATCH_TIMEOUT_SECONDS
