from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / '.env'),
        env_file_encoding='utf-8',
        extra='ignore',
    )

    app_env: str = 'development'
    log_level: str = 'INFO'

    anthropic_api_key: str = ''
    anthropic_generation_model: str = 'claude-sonnet-4-20250514'
    anthropic_classifier_model: str = 'claude-haiku-4-5-20251001'

    jina_api_key: str = ''
    jina_embed_model: str = 'jina-embeddings-v4'
    jina_embed_dim: int = 1024

    pinecone_api_key: str = ''
    pinecone_index_name: str = 'golf-ai-assistant'
    pinecone_cloud: str = 'aws'
    pinecone_region: str = 'us-east-1'

    google_api_key: str = ''
    gemini_generation_model: str = 'gemini-3.6-flash'
    llm_provider: str = 'gemini'

    retrieve_top_k: int = 6
    retrieve_min_score: float = 0.25

    # Rerank: fetch more vector-search candidates, keep the top_k a reranker scores best.
    rerank_enabled: bool = True
    jina_rerank_model: str = 'jina-reranker-v3.5'
    rerank_candidates: int = 20
    # If even the best passage scores below this, nothing relevant was found: answer
    # "not enough information" without an LLM call. On 2026-09-30 the lowest top score
    # among 70 answerable eval questions was 0.34; off-topic ones scored 0.05-0.13.
    rerank_min_score: float = 0.2

    # Phase 2: an LLM call sorts each question into static / live / prediction / off_topic.
    # Off: every question takes the static path, as in Phase 1.
    router_enabled: bool = True
    # Live and prediction questions are answered with Gemini + Google Search. Off (or with
    # LLM_PROVIDER=claude) they get a short "not available" answer instead.
    live_search_enabled: bool = True

    chunk_target_words: int = 300
    chunk_overlap_words: int = 50


settings = Settings()
