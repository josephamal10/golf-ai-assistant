# Golf AI Assistant

Production-oriented golf knowledge assistant for internship delivery. Phase 1 is a
static RAG pipeline: retrieve from a vector index, then generate answers with Claude
**only from retrieved context**, with citations.

Later phases add a query router (live news / rankings) and retrieved expert
predictions that are labeled as opinion, never model-generated forecasts.

## Architecture (Phase 1)

```
Question
   → Voyage embedding
   → Pinecone top-k chunks
   → Claude Sonnet (grounded generation + citations)
   → FastAPI /ask  →  Streamlit UI
```

## Stack

| Layer        | Choice                          |
|--------------|---------------------------------|
| API          | Python, FastAPI                 |
| Embeddings   | Voyage AI (`voyage-3.5`)        |
| Vector DB    | Pinecone serverless             |
| Generation   | Anthropic Claude Sonnet         |
| UI           | Streamlit (temporary)           |
| Sources      | Wikipedia (CC BY-SA) + optional official PDFs |

## Setup

```powershell
cd "C:\Nanonino files\golf-ai-assistant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill in `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, and `PINECONE_API_KEY` in `.env`.

## Ingest static knowledge

Wikipedia collection does **not** need API keys:

```powershell
python scripts/collect_data.py
python scripts/chunk_data.py
```

Embed and upsert (needs Voyage + Pinecone):

```powershell
python scripts/embed_and_upsert.py
```

Optional: drop a licensed Rules of Golf PDF in `data/raw/pdfs/` and re-run collect
(the PDF collector picks up `*.pdf` in that folder).

## Run

```powershell
# API
uvicorn app.main:app --reload --port 8000

# UI (separate terminal)
streamlit run frontend/streamlit_app.py
```

- Health: `GET http://127.0.0.1:8000/health`
- Ask: `POST http://127.0.0.1:8000/ask` with `{"question": "What is a bogey?"}`

## Eval

Starter set: `data/eval/golf_eval_set.json` (50 questions across categories).

```powershell
python scripts/evaluate.py
```

That hits `/ask` and writes `data/eval/last_run.json` for manual scoring. The API
must be running first.

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
tests/          Unit tests (chunker, schemas)
```
