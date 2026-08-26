from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.pdf_rules import collect_pdfs
from app.ingestion.wikipedia import collect_wikipedia
from app.logging_config import configure_logging


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description='Collect Phase 1 golf knowledge sources.')
    parser.add_argument('--skip-wikipedia', action='store_true')
    parser.add_argument('--skip-pdfs', action='store_true')
    args = parser.parse_args()

    if not args.skip_wikipedia:
        wiki_paths = collect_wikipedia()
        print(f'Wikipedia documents: {len(wiki_paths)}')
    if not args.skip_pdfs:
        pdf_paths = collect_pdfs()
        print(f'PDF documents: {len(pdf_paths)}')


if __name__ == '__main__':
    main()
