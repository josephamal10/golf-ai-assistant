import json

import httpx
import pytest

from app.rag import reranker as rr
from app.rag.retriever import Retriever


def _match(i: int, score: float = 0.7) -> dict:
    return {
        'id': f'c{i}',
        'score': score,
        'metadata': {'title': f'Article {i}', 'section': 'History', 'text': f'passage {i}'},
    }


class _FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _FakeStore:
    def __init__(self, matches: list[dict]) -> None:
        self.matches, self.top_k_asked = matches, None

    def query(self, vector: list[float], top_k: int, filter: dict | None = None) -> list[dict]:
        self.top_k_asked = top_k
        return self.matches[:top_k]


class _ReversingReranker:
    def rerank(self, query: str, matches: list[dict], top_n: int) -> list[dict]:
        return list(reversed(matches))[:top_n]


class _BrokenReranker:
    def rerank(self, query: str, matches: list[dict], top_n: int) -> list[dict]:
        raise httpx.ConnectError('down')


def test_reranker_reorders_matches_by_jina_scores(monkeypatch) -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={'results': [
            {'index': 2, 'relevance_score': 0.91},
            {'index': 0, 'relevance_score': 0.40},
        ]})

    monkeypatch.setattr(rr.settings, 'jina_api_key', 'test-key')
    reranker = rr.Reranker()
    reranker._client = httpx.Client(transport=httpx.MockTransport(handler))
    result = reranker.rerank('who won?', [_match(0), _match(1), _match(2)], top_n=2)

    assert [m['id'] for m in result] == ['c2', 'c0']
    assert result[0]['rerank_score'] == 0.91 and result[0]['score'] == 0.7   # vector score kept
    assert sent[0]['query'] == 'who won?' and sent[0]['top_n'] == 2
    assert sent[0]['documents'][0] == 'Article 0 > History\n\npassage 0'


def test_retriever_fetches_extra_candidates_and_keeps_the_reranked_top_k(monkeypatch) -> None:
    monkeypatch.setattr('app.rag.retriever.settings.rerank_candidates', 20)
    store = _FakeStore([_match(i) for i in range(30)])
    retriever = Retriever(_FakeEmbeddings(), store, _ReversingReranker())
    result = retriever.retrieve('q', top_k=6)

    assert store.top_k_asked == 20
    assert [m['id'] for m in result] == ['c19', 'c18', 'c17', 'c16', 'c15', 'c14']


def test_retriever_drops_low_scores_before_reranking() -> None:
    store = _FakeStore([_match(0, 0.8), _match(1, 0.1), _match(2, 0.6)])
    result = Retriever(_FakeEmbeddings(), store, _ReversingReranker()).retrieve('q', top_k=6, min_score=0.25)
    assert [m['id'] for m in result] == ['c2', 'c0']


class _ScoringReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def rerank(self, query: str, matches: list[dict], top_n: int) -> list[dict]:
        return [dict(m, rerank_score=s) for m, s in zip(matches, self.scores)][:top_n]


def test_retriever_returns_nothing_when_even_the_best_passage_is_irrelevant(monkeypatch) -> None:
    monkeypatch.setattr('app.rag.retriever.settings.rerank_min_score', 0.2)
    store = _FakeStore([_match(i) for i in range(3)])
    result = Retriever(_FakeEmbeddings(), store, _ScoringReranker([0.05, 0.03, 0.01])).retrieve('capital of Australia?')
    assert result == []


def test_weak_supporting_passages_are_kept_when_the_best_one_is_relevant(monkeypatch) -> None:
    monkeypatch.setattr('app.rag.retriever.settings.rerank_min_score', 0.2)
    store = _FakeStore([_match(i) for i in range(3)])
    result = Retriever(_FakeEmbeddings(), store, _ScoringReranker([0.6, 0.1, 0.05])).retrieve('four majors?')
    assert [m['id'] for m in result] == ['c0', 'c1', 'c2']


def test_retriever_falls_back_to_vector_order_when_rerank_fails() -> None:
    store = _FakeStore([_match(i) for i in range(10)])
    result = Retriever(_FakeEmbeddings(), store, _BrokenReranker()).retrieve('q', top_k=3)
    assert [m['id'] for m in result] == ['c0', 'c1', 'c2']


def test_retriever_without_reranker_asks_the_store_for_top_k(monkeypatch) -> None:
    monkeypatch.setattr('app.rag.retriever.settings.rerank_enabled', False)
    store = _FakeStore([_match(i) for i in range(10)])
    result = Retriever(_FakeEmbeddings(), store).retrieve('q', top_k=4)
    assert store.top_k_asked == 4 and len(result) == 4
