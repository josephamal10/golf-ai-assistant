# Current status — Golf AI Assistant

Update this file at the end of each working session. Last updated: 2026-09-30.

## Phase

Phase 1 — Static RAG pipeline (in progress)

## Done

- Project scaffold at `C:\Nanonino files-RAG\golf-ai-assistant`
- Wikipedia collector (full articles, duplicate-page skip, disambiguation-page skip,
  Wikimedia-compliant User-Agent)
- Section-aware chunker (~300 words, 50-word overlap; drops See also / References /
  External links; records each chunk's section; merges short sections)
- 143 Wikipedia documents collected → **1,802 chunks** in `data/processed/chunks.jsonl`
  (2026-09-30: removed two disambiguation pages, 'Fairway' and 'Four-ball', that had
  been collected instead of articles; the real content is in Golf course / Four-ball golf)
- Jina embeddings (`jina-embeddings-v4`, 1024-d) + Pinecone upsert script (`--recreate` to rebuild)
- Grounded generation with Gemini (default) or Claude, “answer only from context + citations”
- Citation numbers in answers match the returned sources list (context grouped by article)
- Rate limits / quota errors return HTTP 429 with a clear message
- Gemini/Claude server and connection errors (e.g. 503 "high demand") get up to 4 attempts
  (Gemini via tenacity, Claude via the SDK's `max_retries`), then return HTTP 503 with the
  provider's reason. Retries are capped at 45s so they stay inside the UI's 150s wait
- Only provider-authored messages reach the client; other errors return a generic 500
  and are logged, so raw payloads and config hints are not exposed
- FastAPI `GET /health` and `POST /ask`
- Streamlit chat UI: themed, example questions, provider/model + latency per answer,
  numbered source cards matching the `[n]` citations, distinct 429/503/timeout states
- 50-question eval set with expected answer keywords and expected source articles
- Hard eval set `data/eval/golf_eval_hard.json` (25): 10 paraphrased, 10 specific-fact,
  5 out-of-scope; every expected fact verified to exist in the chunks. Both eval
  scripts take `--eval-file`
- Retrieval eval (`scripts/evaluate_retrieval.py`) after the 2026-09-29 re-chunk + rebuild:
  expected article retrieved for 50/50 questions, ranked first for 48/50 (was 50/50),
  answer keywords in context for 50/50; reference/link-list chunks in retrieved context
  dropped from 21 of 300 to 0. The two rank changes (q025, q050) still have the right
  article at ranks 2–3 and the facts in context.
- Retrieval eval on the hard set (2026-09-30): expected article retrieved for 20/20
  answerable questions, ranked first for 18/20; paraphrased questions are the weak spot
  (8/10 first; h005 "partners take turns hitting the same ball" only at rank 5)
- Finding: similarity scores cannot separate out-of-scope questions from answerable ones
  (out-of-scope top scores 0.59–0.76; answerable lowest 0.61, median 0.79), so the 0.25 `min_score`
  never filters anything and refusals rely on the prompt. Detecting live/prediction/
  off-topic intent is the Phase 2 router's job
- End-to-end eval (2026-08-31): 19 of the 20 answered questions passed; the other 30
  failed once the Gemini quota ran out
- End-to-end eval on the hard set (2026-09-30): 15/15 answered questions passed (all 10
  paraphrased, h011–h015 specific facts); the quota ran out at h016 and the run stopped
  cleanly. h016–h025, including all 5 out-of-scope refusals, are still untested
- Follow-up questions (2026-09-30): `/ask` accepts `history`; with history the LLM first
  rewrites the question to stand alone (`search_query` in the response, shown in the UI).
  6-item follow-up set `data/eval/golf_eval_followups.json`. Searching the raw follow-ups
  without the rewrite: right article first for only 2/6 (hit 4/6); the with-rewrite
  number needs an end-to-end run
- Reranking (2026-09-30): Pinecone top 20 → `jina-reranker-v3.5` → best 6, falling back to
  vector order if the reranker fails. Right article ranked first, off → on: original set
  96% → 100%, hard set 90% → 100% (paraphrased 80% → 100%), raw follow-ups 33% → 50%.
  Costs about 1.5–3.5 s per question
- Relevance cutoff: if the best rerank score is below `RERANK_MIN_SCORE` (0.2) the answer
  is "not enough information" with no LLM call. Lowest best score among 70 answerable eval
  questions: 0.34; off-topic/unsupported: 0.05 (capital of Australia), 0.13 (yesterday's
  scores). Golf prediction/ranking questions score 0.36–0.47, so they still need the
  Phase 2 router
- Unit tests: 32 passing (chunker, citations, schemas, generator errors, API error mapping,
  disambiguation detection, follow-up rewriting, reranking)

## Pending (end of 2026-09-30 session)

- `master` now holds all the work (fast-forwarded from `phase1-rag-improvements`), and
  every commit is authored by Joseph Amal: the first commit's placeholder author was
  rewritten on 2026-09-30, which changed all commit IDs. The pre-rewrite history is kept
  in the tags `backup/branch-before-author-fix` and `backup/master-before-author-fix`;
  delete them once you're happy, before publishing
- Restart the API after pulling in code changes unless it runs with `--reload`: a server
  started at 11:27 without it kept serving the old code all afternoon
- Before publishing as a portfolio project: rotate the API keys kept in plain text in
  `work notes.txt` (outside the repo) and get written OK from Nanonino

## Next

1. Finish the end-to-end evals within the daily Gemini quota (20 calls; each follow-up
   uses 2), e.g. one per day. Both now run on the reranked pipeline:
   - `python scripts/evaluate.py --eval-file data/eval/golf_eval_hard.json --start 15`
     (10 calls; includes the 5 out-of-scope refusals)
   - `python scripts/evaluate.py --eval-file data/eval/golf_eval_followups.json` (12 calls)
2. ~~Harder eval questions~~ done 2026-09-30
3. ~~Follow-up questions~~ built 2026-09-30; verify with the follow-up eval above
4. ~~Reranking~~ done 2026-09-30. Hybrid keyword + vector search only if a harder eval
   shows reranking isn't enough
5. Optional: licensed Rules of Golf PDF in `data/raw/pdfs/`

## Blockers

- Gemini free-tier quota: 20 requests per day **per model**
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, confirmed from the 429 details),
  so end-to-end evals run in daily slices. A second Gemini model for rewrites would
  double the budget
- No official Rules PDF yet (Wikipedia only)
