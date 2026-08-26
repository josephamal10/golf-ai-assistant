from __future__ import annotations

import re
from typing import Any


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
PARA_SPLIT_RE = re.compile(r'\n\s*\n')


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def pack_units(units: list[str], target_words: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        chunks.append(' '.join(current).strip())
        if overlap_words > 0 and chunks[-1]:
            overlap = chunks[-1].split()[-overlap_words:]
            current = [' '.join(overlap)] if overlap else []
            current_words = len(overlap)
        else:
            current = []
            current_words = 0

    for unit in units:
        unit_words = word_count(unit)
        if unit_words == 0:
            continue
        if unit_words > target_words * 1.5:
            flush()
            sentences = split_sentences(unit)
            if len(sentences) <= 1:
                words = unit.split()
                step = max(target_words - overlap_words, 1)
                for i in range(0, len(words), step):
                    window = words[i:i + target_words]
                    if window:
                        chunks.append(' '.join(window))
                current = []
                current_words = 0
                continue
            for sentence in sentences:
                if current_words + word_count(sentence) > target_words and current:
                    flush()
                current.append(sentence)
                current_words = word_count(' '.join(current))
            continue

        if current_words + unit_words > target_words and current:
            flush()
        current.append(unit)
        current_words = word_count(' '.join(current))

    if current and (not chunks or ' '.join(current).strip() != chunks[-1]):
        leftover = ' '.join(current).strip()
        if leftover:
            chunks.append(leftover)

    kept = [c for c in chunks if word_count(c) >= 20]
    if kept:
        return kept
    stripped = ' '.join(units).strip() or text.strip()
    return [stripped] if stripped else []


def chunk_text(text: str, target_words: int = 300, overlap_words: int = 50) -> list[str]:
    paragraphs = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    units = paragraphs if paragraphs else [text]
    return pack_units(units, target_words=target_words, overlap_words=overlap_words)


def chunk_document(
    doc: dict[str, Any],
    target_words: int = 300,
    overlap_words: int = 50,
) -> list[dict[str, Any]]:
    pieces = chunk_text(doc['text'], target_words=target_words, overlap_words=overlap_words)
    chunks: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        chunks.append({
            'id': f"{doc['id']}-{index:04d}",
            'doc_id': doc['id'],
            'title': doc['title'],
            'category': doc['category'],
            'source': doc['source'],
            'url': doc.get('url', ''),
            'license': doc.get('license', ''),
            'chunk_index': index,
            'word_count': word_count(piece),
            'text': piece,
        })
    return chunks
