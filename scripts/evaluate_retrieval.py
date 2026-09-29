"""Retrieval-only eval: does search return the article and facts each question needs?

Two checks per question: an expected article is among the retrieved chunks, and the
prompt context the LLM would see contains the expected answer keywords (the same
keyword check the end-to-end eval applies to answers). Makes one embedding call and one Pinecone query per question, and no LLM calls,
so it is cheap, fast, and unaffected by generation quotas. No API server needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.rag.generator import format_context
from app.rag.retriever import Retriever
from evaluate import keyword_verdict  # sibling script in scripts/


EVAL_PATH = ROOT / 'data' / 'eval' / 'golf_eval_set.json'
OUT_PATH = ROOT / 'data' / 'eval' / 'last_retrieval_run.json'


def first_hit_rank(retrieved_titles: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk from an expected article."""
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in expected:
            return rank
    return None


def summarize(results: list[dict]) -> dict:
    n = len(results)
    ranks = [r['first_hit_rank'] for r in results]
    return {
        'n': n,
        'facts_in_context_rate': (
            round(sum(1 for r in results if r['facts_in_context']) / n, 3) if n else 0.0
        ),
        'hit_rate': round(sum(1 for r in ranks if r) / n, 3) if n else 0.0,
        'top1_rate': round(sum(1 for r in ranks if r == 1) / n, 3) if n else 0.0,
        'mrr': round(sum(1 / r for r in ranks if r) / n, 3) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--limit', type=int, default=0, help='0 = all questions')
    parser.add_argument('--out', type=Path, default=OUT_PATH)
    args = parser.parse_args()

    items = json.loads(EVAL_PATH.read_text(encoding='utf-8'))
    if args.limit:
        items = items[: args.limit]

    retriever = Retriever()
    results = []
    for item in items:
        expected = item.get('expected_sources') or []
        matches = retriever.retrieve(item['question'])
        titles = [str(m['metadata'].get('title', '')) for m in matches]
        rank = first_hit_rank(titles, expected)
        context = format_context(matches)
        facts_found, missing = keyword_verdict(context, item.get('expected_contains', []))
        results.append({
            'id': item['id'],
            'category': item['category'],
            'question': item['question'],
            'expected_sources': expected,
            'retrieved': [
                {
                    'id': m['id'],
                    'title': title,
                    'section': m['metadata'].get('section', ''),
                    'score': round(m['score'], 4),
                }
                for title, m in zip(titles, matches)
            ],
            'first_hit_rank': rank,
            'facts_in_context': facts_found,
            'missing_keywords': missing,
            'context_words': len(context.split()),
        })
        print(
            f"{item['id']} {'HIT ' if rank else 'MISS'} rank={rank or '-'} "
            f"facts={'yes' if facts_found else 'NO '} {item['question'][:55]}",
            flush=True,
        )

    by_category: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_category[result['category']].append(result)

    report = {
        'ran_at': datetime.now(timezone.utc).isoformat(),
        'index': settings.pinecone_index_name,
        'embed_model': settings.jina_embed_model,
        'top_k': settings.retrieve_top_k,
        'min_score': settings.retrieve_min_score,
        **summarize(results),
        'by_category': {cat: summarize(rs) for cat, rs in sorted(by_category.items())},
        'results': results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\nfacts_in_context={report['facts_in_context_rate']:.0%}  "
          f"hit@{settings.retrieve_top_k}={report['hit_rate']:.0%}  "
          f"top1={report['top1_rate']:.0%}  MRR={report['mrr']:.2f}  (n={report['n']})")
    for cat, stats in report['by_category'].items():
        print(f"  {cat:17} facts={stats['facts_in_context_rate']:.0%}  "
              f"hit={stats['hit_rate']:.0%}  top1={stats['top1_rate']:.0%}  n={stats['n']}")
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
