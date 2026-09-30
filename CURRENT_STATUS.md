# Current status — Golf AI Assistant

Update this file at the end of each working session. Last updated: 2026-09-29.

## Phase

Phase 1 — Static RAG pipeline (in progress)

## Done

- Project scaffold at `C:\Nanonino files-RAG\golf-ai-assistant`
- Wikipedia collector (full articles, duplicate-page skip, Wikimedia-compliant User-Agent)
- Section-aware chunker (~300 words, 50-word overlap; drops See also / References /
  External links; records each chunk's section; merges short sections)
- 145 Wikipedia documents collected → **1,804 chunks** in `data/processed/chunks.jsonl`
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
- Streamlit chat UI
- 50-question eval set with expected answer keywords and expected source articles
- Retrieval eval (`scripts/evaluate_retrieval.py`) after the 2026-09-29 re-chunk + rebuild:
  expected article retrieved for 50/50 questions, ranked first for 48/50 (was 50/50),
  answer keywords in context for 50/50; reference/link-list chunks in retrieved context
  dropped from 21 of 300 to 0. The two rank changes (q025, q050) still have the right
  article at ranks 2–3 and the facts in context.
- End-to-end eval (2026-08-31): 19 of the 20 answered questions passed; the other 30
  failed once the Gemini quota ran out
- Unit tests: 14 passing (chunker, citations, schemas, generator errors, API error mapping)

## Pending (end of 2026-09-30 session)

- Work is on branch `phase1-rag-improvements`; not merged into `master` yet
- First commit (`c670f39`) is still authored by the placeholder `AMAL <amal@local>`;
  optionally rewrite it to Joseph Amal before publishing anywhere
- Before publishing as a portfolio project: get written OK from Nanonino

## Next

1. Re-run the end-to-end eval within the Gemini quota (`python scripts/evaluate.py --limit 20`)
2. Harder eval questions (paraphrased terms, specific facts, out-of-scope questions);
   the current set is too easy to show retrieval differences
3. Follow-up questions: send chat history and rewrite follow-ups into standalone questions
4. Hybrid (keyword + vector) search and/or reranking
5. Optional: licensed Rules of Golf PDF in `data/raw/pdfs/`

## Blockers

- Gemini free-tier quota limits end-to-end eval runs (the 2026-08-31 run was cut off after 20 questions)
- No official Rules PDF yet (Wikipedia only)
