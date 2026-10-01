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
EVAL_GAP_SECONDS = 28                                      # free-tier pacing; paid keys can use --gap 1


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
    parser.add_argument(
        '--start', type=int, default=0,
        help='skip the first N questions, e.g. to resume a run the quota cut short',
    )
    parser.add_argument(
        '--gap', type=float, default=EVAL_GAP_SECONDS,
        help=f'seconds between questions (default {EVAL_GAP_SECONDS}, for free-tier limits)',
    )
    args = parser.parse_args()
    out_path = (
        OUT_PATH if args.eval_file.resolve() == EVAL_PATH
        else OUT_PATH.with_name(f'last_run.{args.eval_file.stem}.json')
    )

    items = json.loads(args.eval_file.read_text(encoding='utf-8'))[args.start:]
    if args.limit:
        items = items[: args.limit]
    if args.start:
        # A resumed run gets its own file so the earlier part's results are kept.
        out_path = out_path.with_name(f'{out_path.stem}.start{args.start}.json')

    results = []
    rate_limited = False
    with httpx.Client(timeout=180.0) as client:
        health = client.get(f'{args.base_url}/health')
        health.raise_for_status()
        for index, item in enumerate(items):
            try:
                response = client.post(
                    f'{args.base_url}/ask',
                    # Follow-up items carry the earlier conversation the question depends on.
                    json={'question': item['question'], 'history': item.get('history', [])},
                )
                status, payload = response.status_code, response.json()
            except httpx.TimeoutException:
                # One slow question must not lose the whole run: record it and carry on.
                status, payload = 'timeout', {}
            answer = payload.get('answer') or ''
            passed, missing = keyword_verdict(answer, item.get('expected_contains', []))
            # Items without an expected_route are knowledge-base questions.
            expected_route = item.get('expected_route', 'static')
            if status == 200 and payload.get('route') != expected_route:
                passed = False
                missing.append(f"route: expected {expected_route}, got {payload.get('route')}")
            if status != 200 or not answer:
                verdict = 'MISSING'
            elif passed:
                verdict = 'PASS'
            else:
                verdict = 'FAIL'
            results.append({
                'id': item['id'],
                'category': item['category'],
                'question': item['question'],
                'expected_route': expected_route,
                'route': payload.get('route'),
                'route_reason': payload.get('route_reason'),
                'search_query': payload.get('search_query'),
                'model': payload.get('model'),
                'expected_contains': item.get('expected_contains', []),
                'status_code': status,
                'answer': payload.get('answer'),
                'sources': payload.get('sources', []),
                'insufficient_context': payload.get('insufficient_context'),
                'verdict': verdict,
                'missing_keywords': missing,
            })
            print(
                f"{item['id']} [{status}] {verdict} {payload.get('route', '-')}: {item['question'][:70]}",
                flush=True,
            )
            if status == 429:
                rate_limited = True
                print('Rate limited or out of quota; stopping early and saving partial results.')
                break
            if index < len(items) - 1:
                time.sleep(args.gap)

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
