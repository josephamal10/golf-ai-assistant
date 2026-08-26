from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.logging_config import configure_logging


configure_logging()

app = FastAPI(
    title='Golf AI Assistant',
    version='0.1.0',
    description='Phase 1 static RAG golf knowledge assistant.',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(router)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'phase': '1-static-rag'}
