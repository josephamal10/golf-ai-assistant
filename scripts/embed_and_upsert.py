from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.logging_config import configure_logging
from app.rag.chunker import embedding_text
from app.rag.embeddings import EmbeddingClient
from app.rag.vectorstore import VectorStore


logger = logging.getLogger(__name__)
CHUNKS_PATH = DATA_DIR / 'processed' / 'chunks.jsonl'
EMBED_BATCH = 8


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
        'section': chunk.get('section', ''),
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
    parser = argparse.ArgumentParser(description='Embed chunks.jsonl and upsert to Pinecone.')
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Delete and rebuild the index first. Use after re-chunking or changing the '
        'embedding model, so stale vectors do not linger.',
    )
    args = parser.parse_args()

    chunks = load_chunks()
    embeddings = EmbeddingClient()
    store = VectorStore(recreate=args.recreate)
    processed = 0

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors = embeddings.embed_documents([embedding_text(c) for c in batch])
        records = [to_vector_record(chunk, values) for chunk, values in zip(batch, vectors)]
        store.upsert(records, namespace='static')
        processed += len(records)
        logger.info('Embedded and upserted %s / %s chunks', processed, len(chunks))

    stats = store.stats()
    print(f'Upserted {processed} chunks to Pinecone index.')
    print(
        f"Pinecone stats: dimension={stats.get('dimension')} "
        f"total={stats.get('total_vector_count')} "
        f"static={stats.get('static_vector_count')}"
    )


if __name__ == '__main__':
    main()
