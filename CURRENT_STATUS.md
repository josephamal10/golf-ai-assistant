# Current status — Golf AI Assistant

Update this file at the end of each working session.

## Phase

Phase 1 — Static RAG pipeline (in progress)

## Done

- Project scaffold at `C:\Nanonino files\golf-ai-assistant`
- Wikipedia collector (full articles, duplicate-page skip, Wikimedia-compliant User-Agent)
- Chunker (~300 words, 50-word overlap, category/title/source metadata)
- 145 Wikipedia documents collected → **1,791 chunks** in `data/processed/chunks.jsonl`
- Voyage embeddings + Pinecone upsert script
- Claude Sonnet generation with “answer only from context + citations”
- FastAPI `GET /health` and `POST /ask`
- Streamlit chat UI
- 50-question eval set in `data/eval/golf_eval_set.json`
- Unit tests for chunker (3 passing)

## Next

1. Put Anthropic, Voyage, and Pinecone keys in `.env`
2. Run `python scripts/embed_and_upsert.py`
3. Start API: `uvicorn app.main:app --reload --port 8000`
4. Start UI: `streamlit run frontend/streamlit_app.py`
5. Run `python scripts/evaluate.py` and manually score answers
6. Optional: drop a licensed Rules of Golf PDF in `data/raw/pdfs/` and re-run collect + chunk + upsert

## Blockers

- API keys not in `.env` yet (Voyage, Pinecone, Anthropic) — `/ask` cannot run until upsert + keys
- Git is not installed on this machine; version control is pending
- Three Wikipedia titles were missing: Etiquette in golf, Stymie (golf), Shamble (golf)
