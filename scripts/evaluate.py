from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import time

import httpx


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / 'data' / 'eval' / 'golf_eval_set.json'
OUT_PATH = ROOT / 'data' / 'eval' / 'last_run.json'
EVAL_GAP_SECONDS = 28


def _norm(text: str) -> str:
    t = (text or '').lower().replace('\u2019', "'").replace('\u2018', "'")
    t = t.replace('\u2014', ' ').replace('\u2013', ' ').replace('\u2212', '-')
    t = re.sub(r'(?<=[a-z])[.\-](?=[a-z])', ' ', t)
    return ' '.join(t.split())


def keyword_verdict(answer: str, expected: list) -> tuple[bool, list[str]]:
    """Each expected item is a required phrase, or a list of acceptable alternatives."""
    haystack = _norm(answer)
    missing: list[str] = []
    for item in expected or []:
        options = item if isinstance(item, list) else [item]
        if not any(_norm(option) in haystack for option in options):
            missing.append(' | '.join(str(option) for option in options))
    return (len(missing) == 0, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the golf eval set against /ask.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('--eval-file', type=Path, default=EVAL_PATH)
    parser.add_argument('--limit', type=int, default=0, help='0 = all questions')
    args = parser.parse_args()
    out_path = (
        OUT_PATH if args.eval_file.resolve() == EVAL_PATH
        else OUT_PATH.with_name(f'last_run.{args.eval_file.stem}.json')
    )

    items = json.loads(args.eval_file.read_text(encoding='utf-8'))
    if args.limit:
        items = items[: args.limit]

    results = []
    rate_limited = False
    with httpx.Client(timeout=180.0) as client:
        health = client.get(f'{args.base_url}/health')
        health.raise_for_status()
        for index, item in enumerate(items):
            response = client.post(
                f'{args.base_url}/ask',
                json={'question': item['question']},
            )
            payload = response.json()
            answer = payload.get('answer') or ''
            passed, missing = keyword_verdict(answer, item.get('expected_contains', []))
            if response.status_code != 200 or not answer:
                verdict = 'MISSING'
            elif passed:
                verdict = 'PASS'
            else:
                verdict = 'FAIL'
            results.append({
                'id': item['id'],
                'category': item['category'],
                'question': item['question'],
                'expected_contains': item.get('expected_contains', []),
                'status_code': response.status_code,
                'answer': payload.get('answer'),
                'sources': payload.get('sources', []),
                'insufficient_context': payload.get('insufficient_context'),
                'verdict': verdict,
                'missing_keywords': missing,
            })
            print(
                f"{item['id']} [{response.status_code}] {verdict} {item['question'][:70]}",
                flush=True,
            )
            if response.status_code == 429:
                rate_limited = True
                print('Rate limited or out of quota; stopping early and saving partial results.')
                break
            if index < len(items) - 1:
                time.sleep(EVAL_GAP_SECONDS)

    n = len(results)
    n_pass = sum(1 for r in results if r['verdict'] == 'PASS')
    n_fail = sum(1 for r in results if r['verdict'] == 'FAIL')
    n_missing = sum(1 for r in results if r['verdict'] == 'MISSING')
    report = {
        'ran_at': datetime.now(timezone.utc).isoformat(),
        'base_url': args.base_url,
        'eval_file': str(args.eval_file),
        'n': n,
        'n_pass': n_pass,
        'n_fail': n_fail,
        'n_missing': n_missing,
        'stopped_rate_limited': rate_limited,
        'results': results,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PASS={n_pass} FAIL={n_fail} MISSING={n_missing} / {n}')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    try:
        main()
    except httpx.ConnectError:
        print('Could not reach the API. Start it with: uvicorn app.main:app --reload --port 8001')
        sys.exit(1)
