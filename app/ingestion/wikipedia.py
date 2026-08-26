from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from app.config import DATA_DIR
from app.ingestion.catalog import WIKIPEDIA_SOURCES, WikiSource


logger = logging.getLogger(__name__)

WIKI_API = 'https://en.wikipedia.org/w/api.php'
# Wikimedia requires a descriptive UA with a URL and contact, or requests get 403.
USER_AGENT = (
    'GolfAIAssistant/0.1 '
    '(https://en.wikipedia.org/wiki/User:GolfAIIntern; golf-ai-assistant@example.com)'
)
RAW_DIR = DATA_DIR / 'raw' / 'wikipedia'
REF_MARK_RE = re.compile(r'\[\d+\]')
MULTI_SPACE_RE = re.compile(r'[ \t]+')
MULTI_NL_RE = re.compile(r'\n{3,}')


def slugify(title: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    return slug or 'untitled'


def clean_extract(text: str) -> str:
    text = REF_MARK_RE.sub('', text)
    text = MULTI_SPACE_RE.sub(' ', text)
    text = MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_extract(client: httpx.Client, wiki_title: str) -> dict[str, Any] | None:
    params = {
        'action': 'query',
        'prop': 'extracts|info',
        'explaintext': '1',
        'inprop': 'url',
        'redirects': '1',
        'format': 'json',
        'titles': wiki_title,
    }
    response = client.get(WIKI_API, params=params, timeout=30.0)
    response.raise_for_status()
    pages = response.json().get('query', {}).get('pages', {})
    page = next(iter(pages.values()), None)
    if not page or page.get('missing') is not None or not page.get('extract'):
        return None
    return page


def page_to_document(source: WikiSource, page: dict[str, Any]) -> dict[str, Any]:
    title = page.get('title') or source['title']
    page_id = page.get('pageid')
    return {
        'id': f"wiki-{page_id}-{slugify(title)}",
        'title': title,
        'requested_title': source['wiki_title'],
        'category': source['category'],
        'source': 'wikipedia',
        'license': 'CC BY-SA 4.0',
        'url': page.get('fullurl') or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        'wiki_page_id': page_id,
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'text': clean_extract(page['extract']),
    }


def save_document(doc: dict[str, Any]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{doc['id']}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def collect_wikipedia(delay_s: float = 0.35) -> list[Path]:
    written: list[Path] = []
    skipped: list[str] = []
    seen_page_ids: set[int] = set()
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json',
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for stale in RAW_DIR.glob('*.json'):
        stale.unlink()

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for source in WIKIPEDIA_SOURCES:
            try:
                page = fetch_extract(client, source['wiki_title'])
            except (httpx.HTTPError, RetryError):
                logger.exception('Failed to fetch %s', source['wiki_title'])
                skipped.append(source['wiki_title'])
                continue

            if page is None:
                logger.warning('Missing or empty Wikipedia page: %s', source['wiki_title'])
                skipped.append(source['wiki_title'])
                time.sleep(delay_s)
                continue

            page_id = page.get('pageid')
            if page_id in seen_page_ids:
                logger.info('Skip duplicate page %s (%s)', page.get('title'), source['wiki_title'])
                time.sleep(delay_s)
                continue
            seen_page_ids.add(page_id)

            doc = page_to_document(source, page)
            written.append(save_document(doc))
            logger.info('Saved %s (%s words)', doc['id'], len(doc['text'].split()))
            time.sleep(delay_s)

    logger.info('Wikipedia collect done: %s saved, %s skipped', len(written), len(skipped))
    if skipped:
        logger.info('Skipped titles: %s', ', '.join(skipped))
    return written
