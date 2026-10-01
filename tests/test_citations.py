import re

import pytest

from app.rag.generator import drop_invalid_citations, format_context, user_prompt
from app.rag.pipeline import StaticRagPipeline, _dedupe_sources


def _match(title: str, text: str, section: str = '') -> dict:
    return {
        'id': f'{title}-{text}',
        'score': 0.5,
        'metadata': {
            'title': title,
            'url': f'https://en.wikipedia.org/wiki/{title}',
            'category': 'glossary',
            'source': 'wikipedia',
            'section': section,
            'text': text,
        },
    }


def test_context_numbers_match_sources_when_chunks_share_a_document() -> None:
    # Regression: chunks used to be numbered 1..k in the prompt but sources were
    # renumbered per document, so the model's [4] pointed at nothing in the UI.
    matches = [
        _match('Par (score)', 'birdie text'),
        _match('Par (score)', 'eagle text', section='Eagle'),
        _match('Glossary of golf', 'glossary text'),
        _match('Golf', 'golf text'),
    ]
    context = format_context(matches)
    sources = _dedupe_sources(matches)

    context_numbers = [int(n) for n in re.findall(r'^\[(\d+)\] title=', context, re.MULTILINE)]
    assert context_numbers == [s.n for s in sources] == [1, 2, 3]
    for source in sources:
        assert f'[{source.n}] title={source.title} |' in context

    par_block = context.split('[2] title=')[0]
    assert 'birdie text' in par_block and 'Section: Eagle\neagle text' in par_block


@pytest.mark.parametrize('answer, n_sources, expected', [
    # Seen live: one source, but the model numbered the passages itself.
    ('He won 18 majors [1][2][3].', 1, 'He won 18 majors [1].'),
    ('Founded in 1860 [2]. Played at Prestwick [4].', 2, 'Founded in 1860 [2]. Played at Prestwick.'),
    ('Valid [1] and [3].', 3, 'Valid [1] and [3].'),
    ('No citations at all.', 0, 'No citations at all.'),
])
def test_drop_invalid_citations(answer: str, n_sources: int, expected: str) -> None:
    assert drop_invalid_citations(answer, n_sources) == expected


def test_prompt_states_which_citation_numbers_exist() -> None:
    one_article = [_match('Jack Nicklaus', 'early life'), _match('Jack Nicklaus', 'majors')]
    assert 'Retrieved context (1 source; cite only [1]):' in user_prompt('q', one_article)
    three = one_article + [_match('Golf', 'x'), _match('Masters Tournament', 'y')]
    assert 'Retrieved context (3 sources; cite only [1] to [3]):' in user_prompt('q', three)


def test_pipeline_never_returns_a_citation_without_a_source() -> None:
    class _OneArticleRetriever:
        def retrieve(self, question: str) -> list[dict]:
            return [_match('Jack Nicklaus', f'passage {i}') for i in range(6)]

    class _OverCitingGenerator:
        provider, model = 'fake', 'fake-1'

        def generate(self, question: str, matches: list[dict]) -> str:
            return 'Jack Nicklaus won 18 majors [1][2][3].'

    response = StaticRagPipeline(_OneArticleRetriever(), _OverCitingGenerator()).ask('How many majors?')
    assert len(response.sources) == 1
    assert response.answer == 'Jack Nicklaus won 18 majors [1].'
