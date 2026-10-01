import json

import httpx
import pytest

from app.rag import generator as gen
from app.rag import pipeline as pipe
from app.rag.pipeline import AssistantPipeline
from app.rag.router import LiveAnswer, RouteDecision, parse_route


MATCH = {'id': 'c1', 'score': 0.9, 'metadata': {'title': 'Masters Tournament', 'url': 'u', 'text': 'Augusta.'}}


@pytest.mark.parametrize('raw, route', [
    ('{"route": "live", "reason": "asks about this week"}', 'live'),
    ('```json\n{"route": "off-topic"}\n```', 'off_topic'),
    ('Route: prediction', 'prediction'),
    ('{"route": "weather"}', 'static'),
    ('', 'static'),
])
def test_parse_route(raw: str, route: str) -> None:
    assert parse_route(raw).route == route


class _Retriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, question: str) -> list[dict]:
        self.queries.append(question)
        return [MATCH]


class _Generator:
    provider, model, supports_live = 'fake', 'fake-1', True

    def __init__(self, route: str = 'static', fail_routing: bool = False) -> None:
        self.route, self.fail_routing = route, fail_routing
        self.classified: list[str] = []
        self.generated: list[str] = []
        self.live_calls: list[tuple[str, str]] = []

    def classify(self, question: str) -> RouteDecision:
        self.classified.append(question)
        if self.fail_routing:
            raise httpx.ConnectError('down')
        return RouteDecision(self.route, 'because')

    def rewrite_question(self, question: str, history: list[dict]) -> str:
        return 'Who leads the Masters right now?'

    def generate(self, question: str, matches: list[dict]) -> str:
        self.generated.append(question)
        return 'Augusta National [1].'

    def answer_live(self, question: str, route: str) -> LiveAnswer:
        self.live_calls.append((question, route))
        return LiveAnswer(
            text='Scottie Scheffler leads by two [1].',
            sources=[{'title': 'pgatour.com', 'url': 'https://example.test/r'}],
            search_suggestions_html='<div class="chips"></div>',
        )


def test_static_questions_use_the_knowledge_base() -> None:
    retriever, generator = _Retriever(), _Generator('static')
    response = AssistantPipeline(retriever, generator).ask('Where is the Masters played?')

    assert response.route == 'static' and response.route_reason == 'because'
    assert retriever.queries == generator.generated == ['Where is the Masters played?']
    assert generator.live_calls == []


def test_off_topic_questions_skip_retrieval_and_generation() -> None:
    retriever, generator = _Retriever(), _Generator('off_topic')
    response = AssistantPipeline(retriever, generator).ask('What is the capital of Australia?')

    assert response.route == 'off_topic'
    assert response.answer == pipe.OFF_TOPIC_ANSWER and response.sources == []
    assert retriever.queries == [] and generator.generated == [] and generator.live_calls == []


@pytest.mark.parametrize('route', ['live', 'prediction'])
def test_live_and_prediction_questions_go_to_web_search(route: str) -> None:
    retriever, generator = _Retriever(), _Generator(route)
    response = AssistantPipeline(retriever, generator).ask('Who will win the Masters?')

    assert generator.live_calls == [('Who will win the Masters?', route)]
    assert retriever.queries == []
    assert response.route == route
    assert response.answer == 'Scottie Scheffler leads by two [1].'
    assert [(s.n, s.title, s.category) for s in response.sources] == [(1, 'pgatour.com', 'web')]
    assert response.search_suggestions_html == '<div class="chips"></div>'


def test_follow_ups_are_routed_after_the_rewrite() -> None:
    generator = _Generator('live')
    history = [{'role': 'user', 'content': 'Tell me about the Masters'}]
    response = AssistantPipeline(_Retriever(), generator).ask('Who leads it right now?', history=history)

    assert generator.classified == ['Who leads the Masters right now?']
    assert response.search_query == 'Who leads the Masters right now?'


def test_a_router_failure_falls_back_to_the_knowledge_base() -> None:
    retriever, generator = _Retriever(), _Generator(fail_routing=True)
    response = AssistantPipeline(retriever, generator).ask('Where is the Masters played?')

    assert response.route == 'static'
    assert retriever.queries == ['Where is the Masters played?']


def test_router_off_sends_everything_to_the_knowledge_base() -> None:
    generator = _Generator('off_topic')
    response = AssistantPipeline(_Retriever(), generator, use_router=False).ask('Hello there')

    assert response.route == 'static' and generator.classified == []


def test_live_questions_without_web_search_get_a_clear_answer(monkeypatch) -> None:
    generator = _Generator('live')
    generator.supports_live = False                        # e.g. LLM_PROVIDER=claude
    response = AssistantPipeline(_Retriever(), generator).ask('Who leads the Masters right now?')

    assert response.answer == pipe.LIVE_UNAVAILABLE_ANSWER and generator.live_calls == []
    monkeypatch.setattr(pipe.settings, 'live_search_enabled', False)
    generator.supports_live = True
    assert AssistantPipeline(_Retriever(), generator).ask('Who leads?').answer == pipe.LIVE_UNAVAILABLE_ANSWER


def _grounded(text: str, supports: list[dict], chunks: int = 2) -> dict:
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]},
        'groundingMetadata': {
            'webSearchQueries': ['q'],
            'searchEntryPoint': {'renderedContent': '<div>chips</div>'},
            'groundingChunks': [{'web': {'uri': f'https://r/{i}', 'title': f'site{i}.com'}} for i in range(chunks)],
            'groundingSupports': supports,
        }}]}


def test_live_citations_are_placed_by_utf8_byte_offsets() -> None:
    # 'Sörenstam' has a two-byte ö, so character offsets would land one place early.
    text = 'Annika Sörenstam won. Nelly Korda leads.'
    first_end = len('Annika Sörenstam won'.encode('utf-8'))
    data = _grounded(text, [
        {'segment': {'endIndex': first_end}, 'groundingChunkIndices': [1, 0]},
        {'segment': {'startIndex': first_end + 2, 'endIndex': len(text.encode('utf-8')) - 1},
         'groundingChunkIndices': [1]},
    ])
    live = gen.gemini_live_answer(data)

    assert live.text == 'Annika Sörenstam won[1][2]. Nelly Korda leads[2].'
    assert live.sources == [{'title': 'site0.com', 'url': 'https://r/0'}, {'title': 'site1.com', 'url': 'https://r/1'}]
    assert live.search_suggestions_html == '<div>chips</div>'


def test_live_citation_inside_a_character_is_skipped() -> None:
    text = 'Sörenstam'
    live = gen.gemini_live_answer(_grounded(text, [{'segment': {'endIndex': 2}, 'groundingChunkIndices': [0]}]))
    assert live.text == 'Sörenstam'


def test_an_answer_with_no_search_results_is_not_shown() -> None:
    data = _grounded('Rory McIlroy is world number one.', [], chunks=0)
    live = gen.gemini_live_answer(data)
    assert live.text == gen.NO_LIVE_RESULTS and live.sources == []


def _gemini_with(monkeypatch, reply: dict) -> tuple[gen.GeminiGroundedGenerator, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    monkeypatch.setattr(gen.settings, 'google_api_key', 'test-key')
    generator = gen.GeminiGroundedGenerator()
    generator._client = httpx.Client(transport=httpx.MockTransport(handler))
    return generator, sent


def test_gemini_classify_asks_for_one_of_the_four_routes(monkeypatch) -> None:
    reply = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': '{"route": "prediction", "reason": "asks who will win"}'}]}}]}
    generator, sent = _gemini_with(monkeypatch, reply)

    assert generator.classify('Who will win the Open?') == RouteDecision('prediction', 'asks who will win')
    schema = sent[0]['generationConfig']['responseSchema']
    assert schema['properties']['route']['enum'] == ['static', 'live', 'prediction', 'off_topic']


def test_gemini_prediction_search_uses_google_search_and_the_opinion_prompt(monkeypatch) -> None:
    generator, sent = _gemini_with(monkeypatch, _grounded('Experts pick Scheffler.', []))
    live = generator.answer_live('Who will win the Open?', 'prediction')

    assert live.text == 'Experts pick Scheffler.'
    assert sent[0]['tools'] == [{'google_search': {}}]
    assert 'Never make your own prediction' in sent[0]['system_instruction']['parts'][0]['text']
