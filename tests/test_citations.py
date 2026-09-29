import re

from app.rag.generator import format_context
from app.rag.pipeline import _dedupe_sources


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
