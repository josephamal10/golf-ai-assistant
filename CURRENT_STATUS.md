# Current status — CaddieAI

Update this file at the end of each working session. Last updated: 2026-10-06.

## Phase

Phase 2 — Query router + live web search (built 2026-10-01). Phase 1 (static RAG) is done.

## Phase 2 — done 2026-10-01

- Router: one LLM call (Gemini, JSON with a fixed list of routes, temperature 0, today's
  date in the prompt) labels each question `static`, `live`, `prediction` or `off_topic`.
  Follow-ups are rewritten first, then routed. If the router call fails the question
  takes the static path. `ROUTER_ENABLED=false` restores Phase 1 behaviour
- static → the Phase 1 RAG pipeline, unchanged
- live → Gemini with Google Search grounding: dated answers with the web pages used as
  `[n]` sources. Citation marks are placed from Google's byte offsets (tested with
  "Sörenstam"/"Åberg"). An answer with no search results behind it is replaced by "could
  not find current information" rather than shown from model memory
- prediction → the same search, but the prompt forbids the model's own picks: it reports
  published predictions and odds as other people's opinions, says it gives no betting
  advice, and does not pass on betting tips (a first version relayed "try Top 5 bets to
  manage risk"; the prompt now forbids staking tips and two re-runs were clean)
- off_topic → fixed "golf only" reply, no retrieval and no LLM answer (~3 s, the router call)
- Google's terms for grounded results: answers are shown unedited and Google's search
  suggestions are displayed with each one (the UI renders `search_suggestions_html`,
  links open in a new tab)
- UI: a label above each answer says where it came from (knowledge base / live web
  search / published predictions / outside golf); web answers list "Web sources"; two
  new example questions (world number one, Masters favourites)
- `/health` reports `router` and `live_search`; `LIVE_SEARCH_ENABLED=false` or
  `LLM_PROVIDER=claude` gives live/prediction questions a short "not enabled" reply
- Router eval (`scripts/evaluate_router.py`, new set `golf_eval_router.json` with 37
  live/prediction/off-topic/tricky-static questions, plus the original 50 and hard 25):
  **111/112 routed correctly (99%)**; live, prediction and off-topic recall 100%. The one
  miss, q032 "Name the current LPGA major championships", went to web search because of
  "current"; the web answers it fine, so the prompt was not tuned to this one question.
  Median 1.7 s per routing call
- End-to-end with the router on (paid key): hard set 25/25 (the 5 formerly out-of-scope
  questions now route to prediction/live/off_topic and are checked for route, opinion
  label and no betting advice), follow-ups 6/6. The answers to h022 and h024 were read by
  hand: h022 gave the current OWGR number one with the ranking week; h024 said no PGA
  Tour event finished yesterday and named the one starting today instead of inventing
  scores
- Cost: each question now makes one extra small routing call; each live/prediction answer
  is billed per Google search the model runs. All of today's Phase 2 testing (~112
  routing calls, ~15 web answers) cost well under a dollar
- Unit tests: 58 passing (18 new: route parsing, pipeline routing incl. fallbacks, live
  citations by byte offset, no-results replacement, Gemini request shapes)
- Original 50-question end-to-end eval with the router on (2026-10-06): 48/50 by the
  keyword check, 50/50 correct when the two FAILs were read: q020 (known, see Pending)
  and q037, whose answer says "Swedish" where the check wants "Sweden". q032 routed to
  static this time (it went to live in the router eval)

## Publishing (2026-10-06)

- Nanonino approved publishing the project and keeping it in Joseph's portfolio
- Published: https://github.com/josephamal10/CaddieAI (public, MIT license); CI passed on the first push
- Before publishing: every commit, including the old pre-author-fix history, was searched
  for Gemini, Claude, Jina and Pinecone key patterns (none found); `.env` is ignored; the
  collected Wikipedia text (`data/raw`, `data/processed`) is not in the repo. The two
  `backup/...` tags from the author fix were deleted
- GitHub Actions (`.github/workflows/tests.yml`) runs the 58 unit tests on every push.
  They use fakes and mock HTTP, so no secrets are configured and nothing is billed
  (checked by running them in a clean checkout with no `.env`)
- README: pipeline diagrams (Mermaid, checked to render), and a quick start for
  colleagues with a warning never to run `embed_and_upsert.py --recreate` on the shared
  index

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
  number needs an end-to-end run (done 2026-10-01, see below)
- Reranking (2026-09-30): Pinecone top 20 → `jina-reranker-v3.5` → best 6, falling back to
  vector order if the reranker fails. Right article ranked first, off → on: original set
  96% → 100%, hard set 90% → 100% (paraphrased 80% → 100%), raw follow-ups 33% → 50%.
  Costs about 1.5–3.5 s per question
- Relevance cutoff: if the best rerank score is below `RERANK_MIN_SCORE` (0.2) the answer
  is "not enough information" with no LLM call. Lowest best score among 70 answerable eval
  questions: 0.34; off-topic/unsupported: 0.05 (capital of Australia), 0.13 (yesterday's
  scores). Golf prediction/ranking questions score 0.36–0.47, so they still need the
  Phase 2 router
- **End-to-end on the full pipeline with the company's paid Gemini key (2026-09-30/10-01):**
  - Follow-ups 6/6: every rewrite resolved its reference (he → Jack Nicklaus, it → the
    Masters, the first official one → the Ryder Cup, she → Annika Sörenstam, they → links
    courses) and the topic change ("What is a birdie?") was left unchanged
  - Hard set 25/25, including all 5 out-of-scope questions: the 3 golf prediction/ranking/
    betting questions were refused by the prompt with no prediction; the 2 off-topic ones
    were stopped by the relevance cutoff with no LLM call
  - Original set 49/50. The one FAIL (q020) is a grader false negative seen since August:
    the answer is correct but doesn't contain the expected keyword "Scotland"
  - Key verified as paid before use: it can call the paid-only `gemini-3.1-pro-preview`
    and responses report `serviceTier: standard`
- Request-path timeouts (2026-10-01): a throttled Jina once held one /ask for ~190 s
  (query embedding used the ingestion budget of 120 s x 6 tries). Query embeddings now
  use 20 s x 3 tries (~65 s worst case), reranking 10 s x 2 (~21 s) before falling back
  to vector order. evaluate.py records a timed-out question and carries on, and takes
  `--gap` (5 s is enough on a paid key; 1 s tripped Jina's rate limit)
- Citation numbers beyond the sources (2026-10-01): found by hand in the UI ("[1][2][3]"
  with one source) and in 2 of 81 saved eval answers (q035, q036). All were single-article
  contexts, where the model numbered the passages itself. The prompt now states which
  numbers exist ("cite only [1]") and that passages in one item share its number: 6/6 raw
  answers on those questions were then clean. As a safety net the pipeline also drops any
  [n] with no matching source
- `start.bat`: double-click to start the API (with `--reload`) and the chat page
- Unit tests: 40 passing (chunker, citations, schemas, generator errors, API error mapping,
  disambiguation detection, follow-up rewriting, reranking, embedding timeouts)

## Pending (2026-10-06)

- Restart the API after pulling in code changes unless it runs with `--reload` (which
  `start.bat` uses): a server started without it keeps serving the old code
- Polish (now rare, since live/prediction questions no longer reach the static path): a
  static-path refusal can still cite "[1]" and show the retrieved source cards
- Optional: two eval keywords are stricter than the questions need, so correct answers
  fail the keyword check: q020 ("Scotland") and q037 ("Sweden", answer says "Swedish")

## Next

- Optional: if a static question gets "not enough information", try the web as a
  fallback (would blur the "knowledge base only" promise, so it should be labelled)
- Optional: one LLM call for rewrite + route on follow-ups (saves ~1.7 s per follow-up)
- Optional: live search for `LLM_PROVIDER=claude` (Anthropic's web search tool)
- Portfolio write-up and a short screen recording
- Refresh the Wikipedia collection now and then (collected Aug–Sep 2026; player records
  go stale). Re-collect, re-chunk, then rebuild the index with `--recreate`

Phase 1 list:

1. ~~End-to-end evals on the full pipeline~~ done 2026-10-01 with the paid key
2. ~~Harder eval questions~~ done 2026-09-30
3. ~~Follow-up questions~~ done 2026-09-30, verified 6/6 on 2026-10-01
4. ~~Reranking~~ done 2026-09-30. Hybrid keyword + vector search only if a harder eval
   shows reranking isn't enough
5. Optional: licensed Rules of Golf PDF in `data/raw/pdfs/`

## Blockers

- ~~Gemini free-tier quota~~ (20 requests/day per model): resolved by the company's paid
  key. Free keys still hit it, and Jina's rate limit is now the first one a fast eval hits
- No official Rules PDF yet (Wikipedia only)
