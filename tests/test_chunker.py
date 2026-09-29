import pytest
from pydantic import ValidationError

from app.rag.chunker import chunk_document, chunk_text, split_sections, word_count
from app.models.schemas import AskRequest


def test_chunk_text_packs_near_target_words() -> None:
    paragraphs = [f'Hole {i} is a par four with bunkers left. ' * 12 for i in range(8)]
    text = '\n\n'.join(paragraphs)
    chunks = chunk_text(text, target_words=80, overlap_words=10)
    assert len(chunks) >= 2
    assert all(word_count(c) >= 20 for c in chunks)


def test_chunk_text_keeps_short_doc_as_one_chunk() -> None:
    text = 'A bogey is one stroke over par. A birdie is one under par.'
    chunks = chunk_text(text, target_words=300, overlap_words=50)
    assert chunks == [text]


def test_chunk_text_handles_empty_text() -> None:
    assert chunk_text('') == []


def test_ask_request_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question='ab')


def _words(n: int, word: str) -> str:
    return ' '.join([word] * n)


def test_split_sections_drops_back_matter_and_empty_sections() -> None:
    text = (
        f'{_words(150, "lead")}\n\n'
        f'== History ==\n{_words(150, "history")}\n\n'
        '== Results ==\nSources:\n\n'
        f'== See also ==\nGolf\nMatch play\n\n'
        f'=== Related lists ===\n{_words(30, "links")}\n\n'
        '== References ==\n\n'
        '== External links ==\nOfficial site'
    )
    sections = split_sections(text)
    assert [heading for heading, _ in sections] == ['', 'History']
    joined = ' '.join(body for _, body in sections)
    assert 'links' not in joined and 'Official site' not in joined


def test_split_sections_folds_subsections_and_merges_short_sections() -> None:
    text = (
        f'{_words(150, "lead")}\n\n'
        '== Career ==\n'
        f'=== Amateur ===\n{_words(150, "amateur")}\n\n'
        f'== Personal life ==\n{_words(30, "family")}\n\n'
        f'== Awards ==\n{_words(30, "award")}'
    )
    sections = split_sections(text)
    assert [heading for heading, _ in sections] == ['', 'Career', 'Personal life; Awards']
    assert sections[1][1].startswith('Amateur\namateur')


def test_split_sections_without_headings_is_one_lead_section() -> None:
    assert split_sections('Plain PDF text.') == [('', 'Plain PDF text.')]


def test_chunk_document_records_section_and_sequential_ids() -> None:
    doc = {
        'id': 'wiki-1-golf', 'title': 'Golf', 'category': 'rules', 'source': 'wikipedia',
        'text': f'{_words(150, "lead")}\n\n== History ==\n{_words(150, "history")}',
    }
    chunks = chunk_document(doc)
    assert [(c['id'], c['section']) for c in chunks] == [
        ('wiki-1-golf-0000', ''),
        ('wiki-1-golf-0001', 'History'),
    ]
