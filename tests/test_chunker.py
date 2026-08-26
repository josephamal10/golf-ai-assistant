from app.rag.chunker import chunk_text, word_count
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


def test_ask_request_rejects_empty() -> None:
    try:
        AskRequest(question='ab')
        assert False, 'expected validation error'
    except Exception:
        pass
