from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, settings
from app.rag.chunker import chunk_document


logger = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / 'processed'


def iter_raw_documents() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for folder in (DATA_DIR / 'raw' / 'wikipedia', DATA_DIR / 'raw' / 'pdf_json'):
        if not folder.exists():
            continue
        for path in sorted(folder.glob('*.json')):
            docs.append(json.loads(path.read_text(encoding='utf-8')))
    return docs


def write_chunks(chunks: list[dict[str, Any]]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / 'chunks.jsonl'
    with out.open('w', encoding='utf-8') as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    return out


def run_chunking() -> Path:
    docs = iter_raw_documents()
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                target_words=settings.chunk_target_words,
                overlap_words=settings.chunk_overlap_words,
            )
        )
    path = write_chunks(chunks)
    logger.info('Wrote %s chunks from %s documents to %s', len(chunks), len(docs), path)
    return path
