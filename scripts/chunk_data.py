from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.chunk_store import run_chunking
from app.logging_config import configure_logging


def main() -> None:
    configure_logging()
    path = run_chunking()
    print(f'Wrote chunks to {path}')


if __name__ == '__main__':
    main()
