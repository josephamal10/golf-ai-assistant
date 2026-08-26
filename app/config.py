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

    voyage_api_key: str = ''
    voyage_embed_model: str = 'voyage-3.5'
    voyage_embed_dim: int = 1024

    pinecone_api_key: str = ''
    pinecone_index_name: str = 'golf-ai-assistant'
    pinecone_cloud: str = 'aws'
    pinecone_region: str = 'us-east-1'

    retrieve_top_k: int = 6
    retrieve_min_score: float = 0.25

    chunk_target_words: int = 300
    chunk_overlap_words: int = 50


settings = Settings()
