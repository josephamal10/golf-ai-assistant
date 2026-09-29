from __future__ import annotations

import re
from typing import Any


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
PARA_SPLIT_RE = re.compile(r'\n\s*\n')
# MediaWiki plain-text headings: "== History ==", "=== Early years ===".
HEADING_RE = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)
# Wikipedia back matter: link lists and emptied reference stubs, no golf content.
BACK_MATTER_SECTIONS = {
    'see also', 'references', 'external links', 'further reading', 'notes',
    'bibliography', 'sources', 'citations', 'footnotes', 'notes and references',
    'references and notes',
}
# "Source:" captions left behind when the plain-text export drops a table.
TABLE_SOURCE_LINE_RE = re.compile(r'^[ \t]*Sources?:?[ \t]*$', re.MULTILINE)
# Sections shorter than this are merged with the next one to avoid tiny chunks.
SECTION_MIN_WORDS = 100


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
    stripped = ' '.join(units).strip()
    return [stripped] if stripped else []


def chunk_text(text: str, target_words: int = 300, overlap_words: int = 50) -> list[str]:
    paragraphs = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    units = paragraphs if paragraphs else [text]
    return pack_units(units, target_words=target_words, overlap_words=overlap_words)


def split_sections(text: str, min_words: int = SECTION_MIN_WORDS) -> list[tuple[str, str]]:
    """Split MediaWiki plain text into (section heading, body) pairs.

    The lead (text before the first heading) has heading ''. Subsections are folded
    into their top-level section with the subheading kept as a line, back matter and
    empty sections are dropped, and short sections are merged into the next one.
    Text without headings (e.g. PDFs) comes back as a single ('', text) pair.
    """
    parts = HEADING_RE.split(TABLE_SOURCE_LINE_RE.sub('', text))
    sections: list[list[str]] = [['', parts[0].strip()]]
    top_heading = ''
    for i in range(1, len(parts), 3):
        level, heading, body = len(parts[i]), parts[i + 1], parts[i + 2].strip()
        if level == 2:
            top_heading = heading
            if heading.lower() not in BACK_MATTER_SECTIONS:
                sections.append([heading, body])
        elif body and top_heading.lower() not in BACK_MATTER_SECTIONS:
            sections[-1][1] = f'{sections[-1][1]}\n\n{heading}\n{body}'.strip()

    merged: list[list[str]] = []
    for heading, body in sections:
        if not body:
            continue
        prev = merged[-1] if merged else None
        if prev and prev[0] and word_count(prev[1]) < min_words:
            prev[0] = f'{prev[0]}; {heading}'
            prev[1] = f'{prev[1]}\n\n{heading}\n{body}'
        else:
            merged.append([heading, body])
    return [(heading, body) for heading, body in merged]


def embedding_text(chunk: dict[str, Any]) -> str:
    """Text sent to the embedder: the title and section give each passage its context."""
    header = ' > '.join(part for part in (chunk['title'], chunk.get('section')) if part)
    return f"{header}\n\n{chunk['text']}"


def chunk_document(
    doc: dict[str, Any],
    target_words: int = 300,
    overlap_words: int = 50,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section, body in split_sections(doc['text']):
        for piece in chunk_text(body, target_words=target_words, overlap_words=overlap_words):
            index = len(chunks)
            chunks.append({
                'id': f"{doc['id']}-{index:04d}",
                'doc_id': doc['id'],
                'title': doc['title'],
                'section': section,
                'category': doc['category'],
                'source': doc['source'],
                'url': doc.get('url', ''),
                'license': doc.get('license', ''),
                'chunk_index': index,
                'word_count': word_count(piece),
                'text': piece,
            })
    return chunks
