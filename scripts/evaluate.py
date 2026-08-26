from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / 'data' / 'eval' / 'golf_eval_set.json'
OUT_PATH = ROOT / 'data' / 'eval' / 'last_run.json'


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the golf eval set against /ask.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--limit', type=int, default=0, help='0 = all questions')
    args = parser.parse_args()

    items = json.loads(EVAL_PATH.read_text(encoding='utf-8'))
    if args.limit:
        items = items[: args.limit]

    results = []
    with httpx.Client(timeout=90.0) as client:
        health = client.get(f'{args.base_url}/health')
        health.raise_for_status()
        for item in items:
            response = client.post(
                f'{args.base_url}/ask',
                json={'question': item['question']},
            )
            payload = response.json()
            results.append({
                'id': item['id'],
                'category': item['category'],
                'question': item['question'],
                'expected_contains': item.get('expected_contains', []),
                'status_code': response.status_code,
                'answer': payload.get('answer'),
                'sources': payload.get('sources', []),
                'insufficient_context': payload.get('insufficient_context'),
            })
            print(f"{item['id']} [{response.status_code}] {item['question'][:70]}")

    report = {
        'ran_at': datetime.now(timezone.utc).isoformat(),
        'base_url': args.base_url,
        'n': len(results),
        'results': results,
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    try:
        main()
    except httpx.ConnectError:
        print('Could not reach the API. Start it with: uvicorn app.main:app --reload --port 8000')
        sys.exit(1)
