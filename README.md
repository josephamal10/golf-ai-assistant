# Golf AI Assistant

Production-oriented golf knowledge assistant for internship delivery. Phase 1 is a
static RAG pipeline: retrieve from a vector index, then generate answers with an LLM
(Gemini by default, Claude optional) **only from retrieved context**, with citations.

Later phases add a query router (live news / rankings) and retrieved expert
predictions that are labeled as opinion, never model-generated forecasts.

## Architecture (Phase 1)

```
Question
   → Jina embedding
   → Pinecone top-20 chunks → Jina reranker keeps the best 6 (grouped by source article)
   → Gemini or Claude (grounded generation + citations)
   → FastAPI /ask  →  Streamlit UI
```

## Stack

| Layer        | Choice                                              |
|--------------|-----------------------------------------------------|
| API          | Python, FastAPI                                     |
| Embeddings   | Jina AI (`jina-embeddings-v4`, 1024-d)              |
| Reranking    | Jina AI (`jina-reranker-v3.5`)                      |
| Vector DB    | Pinecone serverless                                 |
| Generation   | Google Gemini (default) or Anthropic Claude, via `LLM_PROVIDER` |
| UI           | Streamlit (temporary)                               |
| Sources      | Wikipedia (CC BY-SA) + optional official PDFs       |

## Setup

```powershell
cd "C:\Nanonino files-RAG\golf-ai-assistant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill in `JINA_API_KEY`, `PINECONE_API_KEY`, and `GOOGLE_API_KEY` in `.env`. To generate
with Claude instead, set `LLM_PROVIDER=claude` and fill in `ANTHROPIC_API_KEY`.

## Ingest static knowledge

Wikipedia collection does **not** need API keys:

```powershell
python scripts/collect_data.py
python scripts/chunk_data.py
```

Chunking is section-aware: Wikipedia back matter (See also, References, External
links, ...) is dropped, each chunk records its section, and short sections are merged.

Embed and upsert (needs Jina + Pinecone):

```powershell
python scripts/embed_and_upsert.py --recreate
```

`--recreate` deletes and rebuilds the index. Use it after re-chunking or changing the
embedding model, otherwise vectors from the old chunks stay in the index.

Optional: drop a licensed Rules of Golf PDF in `data/raw/pdfs/` and re-run collect
(the PDF collector picks up `*.pdf` in that folder).

## Run

```powershell
# API
uvicorn app.main:app --reload --port 8001

# UI (separate terminal)
streamlit run frontend/streamlit_app.py
```

- Health: `GET http://127.0.0.1:8001/health`
- Ask: `POST http://127.0.0.1:8001/ask` with `{"question": "What is a bogey?"}`

Run Streamlit from the repo root so it picks up the theme in `.streamlit/config.toml`.
The UI shows the active provider and model, disables the composer while the API is
unreachable, and lists the source articles behind every answer, numbered to match the
`[n]` citations. Point it at another API with `GOLF_API_URL`.

Retrieval reranks: Pinecone returns the top `RERANK_CANDIDATES` (20) passages, Jina's
reranker (`jina-reranker-v3.5`) reads the question with each one, and the best
`RETRIEVE_TOP_K` (6) go to the LLM. If even the best passage scores below
`RERANK_MIN_SCORE` (0.2), the answer is "not enough information" with no LLM call. If the
reranker is down, the vector order is used. Set `RERANK_ENABLED=false` to turn it off.

Follow-up questions work: the UI sends the recent conversation as `history`, and when
there is history the API first asks the LLM to rewrite the question to stand alone
("How many majors did he win?" → "How many majors did Jack Nicklaus win?"), then searches
and answers with that. The response's `search_query` shows the rewrite and the UI displays
it. A follow-up therefore costs two LLM calls instead of one.

If the embedding or LLM provider is rate limited or out of quota, `/ask` returns
HTTP 429 with a message instead of a generic 500. Provider outages return HTTP 503 with
the provider's reason; the UI shows both as distinct, actionable messages.

## Eval

Eval set: `data/eval/golf_eval_set.json` (50 questions across categories). Each item has
`expected_contains` (answer keywords) and `expected_sources` (acceptable articles).

Hard set: `data/eval/golf_eval_hard.json` (25 questions): paraphrased terms that don't
name the article, specific facts buried in articles, and out-of-scope questions (live
results, predictions, betting, non-golf) whose expected answer is a refusal. Run either
script on it with `--eval-file data/eval/golf_eval_hard.json`; results go to
`last_*.golf_eval_hard.json`. Out-of-scope items have no expected article, so the
retrieval eval reports them without scoring them.

Retrieval only: no LLM calls, no API server needed, runs in 1–2 minutes:

```powershell
python scripts/evaluate_retrieval.py
```

It reports whether an expected article was retrieved (hit rate, top-1, MRR) and whether
the prompt context contains the expected keywords. Output: `data/eval/last_retrieval_run.json`.

End to end (answers from `/ask`; the API must be running):

```powershell
python scripts/evaluate.py
```

It waits 28 seconds between questions to respect free-tier limits, stops early at the
first HTTP 429, and writes `data/eval/last_run.json`. Use `--limit N` to run fewer and
`--start N` to resume after the first N (a resumed run writes its own `.startN` file).

Follow-ups: `data/eval/golf_eval_followups.json` (6 items) gives each question the earlier
conversation it depends on. The end-to-end eval sends that history; the retrieval eval
searches the raw follow-up, which shows how badly it does without the rewrite.

## Tests

```powershell
python -m pytest
```

## Source and license notes

- Wikipedia text is collected via the MediaWiki API and is CC BY-SA 4.0. Downstream
  answers that quote it should keep attribution (the API already returns source URLs).
- The official Rules of Golf are copyright USGA / R&A. Do not scrape their sites.
  Use Wikipedia for a public summary, or ingest a copy you are licensed to use.

## Project layout

```
app/            FastAPI app, RAG, ingestion
data/catalogs/  Curated Wikipedia title list
data/eval/      Hand-built Q&A eval set
frontend/       Streamlit UI
scripts/        collect → chunk → embed → evaluate
tests/          Unit tests (chunker, citations, schemas)
```
