from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from app.config import DATA_DIR
from app.ingestion.wikipedia import slugify


logger = logging.getLogger(__name__)

PDF_DIR = DATA_DIR / 'raw' / 'pdfs'
RAW_PDF_JSON_DIR = DATA_DIR / 'raw' / 'pdf_json'
WHITESPACE_RE = re.compile(r'[ \t]+')


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ''
        pages.append(text)
    joined = '\n\n'.join(pages)
    joined = WHITESPACE_RE.sub(' ', joined)
    return joined.strip()


def collect_pdfs(category: str = 'rules') -> list[Path]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PDF_JSON_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for pdf_path in sorted(PDF_DIR.glob('*.pdf')):
        text = extract_pdf_text(pdf_path)
        if len(text.split()) < 80:
            logger.warning('Skipping thin PDF extract: %s', pdf_path.name)
            continue

        doc = {
            'id': f'pdf-{slugify(pdf_path.stem)}',
            'title': pdf_path.stem.replace('_', ' ').replace('-', ' '),
            'category': category,
            'source': 'pdf',
            'license': 'user-provided',
            'url': str(pdf_path),
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'text': text,
        }
        out = RAW_PDF_JSON_DIR / f"{doc['id']}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        written.append(out)
        logger.info('Saved PDF extract %s (%s words)', doc['id'], len(text.split()))

    return written
