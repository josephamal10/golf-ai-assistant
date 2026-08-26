from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.logging_config import configure_logging
from app.rag.embeddings import EmbeddingClient
from app.rag.vectorstore import VectorStore


logger = logging.getLogger(__name__)
CHUNKS_PATH = DATA_DIR / 'processed' / 'chunks.jsonl'
EMBED_BATCH = 32


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f'Run chunk_data.py first. Missing {CHUNKS_PATH}')
    chunks = []
    with CHUNKS_PATH.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def to_vector_record(chunk: dict, values: list[float]) -> dict:
    metadata = {
        'title': chunk['title'],
        'category': chunk['category'],
        'source': chunk['source'],
        'url': chunk.get('url', ''),
        'doc_id': chunk['doc_id'],
        'chunk_index': int(chunk['chunk_index']),
        'word_count': int(chunk['word_count']),
        'text': chunk['text'],
    }
    return {'id': chunk['id'], 'values': values, 'metadata': metadata}


def main() -> None:
    configure_logging()
    chunks = load_chunks()
    embeddings = EmbeddingClient()
    store = VectorStore()
    records: list[dict] = []

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors = embeddings.embed_documents([c['text'] for c in batch])
        for chunk, values in zip(batch, vectors):
            records.append(to_vector_record(chunk, values))
        logger.info('Embedded %s / %s chunks', min(start + EMBED_BATCH, len(chunks)), len(chunks))

    store.upsert(records, namespace='static')
    print(f'Upserted {len(records)} chunks to Pinecone index.')


if __name__ == '__main__':
    main()
