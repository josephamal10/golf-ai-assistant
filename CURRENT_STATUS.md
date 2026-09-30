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
- Unit tests: 16 passing (chunker, citations, schemas, generator errors, API error mapping,
  disambiguation detection)

## Pending (end of 2026-09-30 session)

- Work is on branch `phase1-rag-improvements`; not merged into `master` yet
- First commit (`c670f39`) is still authored by the placeholder `AMAL <amal@local>`;
  optionally rewrite it to Joseph Amal before publishing anywhere
- Before publishing as a portfolio project: get written OK from Nanonino

## Next

1. End-to-end eval within the Gemini quota. Most informative: the hard set
   (`python scripts/evaluate.py --eval-file data/eval/golf_eval_hard.json`), which checks
   refusals on the 5 out-of-scope questions; or the original `--limit 20`
2. ~~Harder eval questions~~ done 2026-09-30
3. Follow-up questions: send chat history and rewrite follow-ups into standalone questions
4. Hybrid (keyword + vector) search and/or reranking
5. Optional: licensed Rules of Golf PDF in `data/raw/pdfs/`

## Blockers

- Gemini free-tier quota limits end-to-end eval runs (the 2026-08-31 run was cut off after 20 questions)
- No official Rules PDF yet (Wikipedia only)
