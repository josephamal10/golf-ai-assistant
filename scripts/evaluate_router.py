"""Router eval: classify every eval question and compare with its expected route.

One small LLM call per question, no retrieval and no API server needed. Questions
without an `expected_route` are knowledge-base questions (route `static`).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.errors import RateLimitedError  # noqa: E402
from app.rag.generator import build_generator  # noqa: E402
from app.rag.router import ROUTES  # noqa: E402


EVAL_DIR = ROOT / 'data' / 'eval'
DEFAULT_FILES = [
    EVAL_DIR / 'golf_eval_router.json',
    EVAL_DIR / 'golf_eval_set.json',
    EVAL_DIR / 'golf_eval_hard.json',
]
OUT_PATH = EVAL_DIR / 'last_router_run.json'


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure how often the router picks the expected route.')
    parser.add_argument('--eval-file', type=Path, action='append', help='repeatable; default: router + original + hard sets')
    parser.add_argument('--gap', type=float, default=0.5, help='seconds between questions')
    parser.add_argument('--out', type=Path, default=OUT_PATH)
    args = parser.parse_args()

    items = []
    for path in args.eval_file or DEFAULT_FILES:
        items += [{**item, 'file': path.name} for item in json.loads(path.read_text(encoding='utf-8'))]

    generator = build_generator()
    results, stopped = [], False
    for index, item in enumerate(items):
        expected = item.get('expected_route', 'static')
        started = time.perf_counter()
        try:
            decision = generator.classify(item['question'])
            route, reason = decision.route, decision.reason
        except RateLimitedError:
            print('Rate limited; stopping early and saving partial results.')
            stopped = True
            break
        except Exception as exc:                            # recorded, not fatal: one bad call
            route, reason = 'error', f'{type(exc).__name__}: {exc}'[:200]
        seconds = time.perf_counter() - started
        ok = route == expected
        results.append({
            'id': item['id'], 'file': item['file'], 'category': item['category'],
            'question': item['question'], 'expected_route': expected, 'route': route,
            'reason': reason, 'correct': ok, 'seconds': round(seconds, 2),
        })
        print(f"{item['id']} {'ok  ' if ok else 'MISS'} {expected:>10} -> {route:<10} {item['question'][:60]}", flush=True)
        if index < len(items) - 1:
            time.sleep(args.gap)

    n = len(results)
    correct = sum(r['correct'] for r in results)
    confusion = Counter((r['expected_route'], r['route']) for r in results)
    per_route = {}
    for route in ROUTES:
        expected_n = sum(1 for r in results if r['expected_route'] == route)
        predicted_n = sum(1 for r in results if r['route'] == route)
        hits = confusion[(route, route)]
        per_route[route] = {
            'n': expected_n,
            'recall': round(hits / expected_n, 3) if expected_n else None,       # of these, routed right
            'precision': round(hits / predicted_n, 3) if predicted_n else None,  # of routed here, right
        }
    report = {
        'ran_at': datetime.now(timezone.utc).isoformat(),
        'provider': generator.provider,
        'model': generator.model,
        'n': n,
        'accuracy': round(correct / n, 3) if n else None,
        'per_route': per_route,
        'confusion': [
            {'expected': e, 'routed': r, 'count': c} for (e, r), c in sorted(confusion.items())
        ],
        'median_seconds': sorted(r['seconds'] for r in results)[n // 2] if n else None,
        'stopped_rate_limited': stopped,
        'misroutes': [r for r in results if not r['correct']],
        'results': results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\nAccuracy {correct}/{n}' + (f' ({correct / n:.0%})' if n else ''))
    for route, stats in per_route.items():
        print(f"  {route:<10} n={stats['n']:<3} recall={stats['recall']} precision={stats['precision']}")
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
