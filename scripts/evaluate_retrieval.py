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
    parser.add_argument('--eval-file', type=Path, default=EVAL_PATH)
    parser.add_argument('--limit', type=int, default=0, help='0 = all questions')
    parser.add_argument(
        '--out', type=Path, default=None,
        help='default: data/eval/last_retrieval_run.json, or last_retrieval_run.<eval file>.json',
    )
    args = parser.parse_args()
    out = args.out or (
        OUT_PATH if args.eval_file.resolve() == EVAL_PATH
        else OUT_PATH.with_name(f'last_retrieval_run.{args.eval_file.stem}.json')
    )

    items = json.loads(args.eval_file.read_text(encoding='utf-8'))
    if args.limit:
        items = items[: args.limit]

    retriever = Retriever()
    results = []
    for item in items:
        expected = item.get('expected_sources') or []
        matches = retriever.retrieve(item['question'])
        titles = [str(m['metadata'].get('title', '')) for m in matches]
        context = format_context(matches)
        # Out-of-scope items have no right article; their answer check is a refusal, which
        # the context can't contain. Record what was retrieved but leave them unscored.
        scored = bool(expected)
        rank = first_hit_rank(titles, expected) if scored else None
        facts_found, missing = (
            keyword_verdict(context, item.get('expected_contains', [])) if scored else (False, [])
        )
        results.append({
            'id': item['id'],
            'category': item['category'],
            'question': item['question'],
            'scored': scored,
            'expected_sources': expected,
            'retrieved': [
                {
                    'id': m['id'],
                    'title': title,
                    'section': m['metadata'].get('section', ''),
                    'score': round(m['score'], 4),
                    **({'rerank_score': round(m['rerank_score'], 4)} if 'rerank_score' in m else {}),
                }
                for title, m in zip(titles, matches)
            ],
            'first_hit_rank': rank,
            'facts_in_context': facts_found,
            'missing_keywords': missing,
            'context_words': len(context.split()),
        })
        if scored:
            print(
                f"{item['id']} {'HIT ' if rank else 'MISS'} rank={rank or '-'} "
                f"facts={'yes' if facts_found else 'NO '} {item['question'][:55]}",
                flush=True,
            )
        else:
            top = f'{matches[0]["score"]:.2f}' if matches else '-'
            print(
                f"{item['id']} n/a  retrieved={len(matches)} top_score={top} {item['question'][:48]}",
                flush=True,
            )

    scored_results = [r for r in results if r['scored']]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for result in scored_results:
        by_category[result['category']].append(result)

    report = {
        'ran_at': datetime.now(timezone.utc).isoformat(),
        'index': settings.pinecone_index_name,
        'embed_model': settings.jina_embed_model,
        'top_k': settings.retrieve_top_k,
        'min_score': settings.retrieve_min_score,
        'rerank': (
            {'model': settings.jina_rerank_model, 'candidates': settings.rerank_candidates}
            if settings.rerank_enabled else None
        ),
        'eval_file': str(args.eval_file),
        **summarize(scored_results),
        'n_unscored': len(results) - len(scored_results),
        'by_category': {cat: summarize(rs) for cat, rs in sorted(by_category.items())},
        'results': results,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\nrerank={'on (' + settings.jina_rerank_model + ')' if settings.rerank_enabled else 'off'}")
    print(f"facts_in_context={report['facts_in_context_rate']:.0%}  "
          f"hit@{settings.retrieve_top_k}={report['hit_rate']:.0%}  "
          f"top1={report['top1_rate']:.0%}  MRR={report['mrr']:.2f}  (n={report['n']})")
    for cat, stats in report['by_category'].items():
        print(f"  {cat:17} facts={stats['facts_in_context_rate']:.0%}  "
              f"hit={stats['hit_rate']:.0%}  top1={stats['top1_rate']:.0%}  n={stats['n']}")
    if report['n_unscored']:
        print(f"  ({report['n_unscored']} out-of-scope questions not scored; see 'retrieved' in the report)")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
