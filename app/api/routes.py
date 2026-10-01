from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import AskRequest, AskResponse
from app.rag.errors import RateLimitedError, UpstreamUnavailableError
from app.rag.pipeline import AssistantPipeline


logger = logging.getLogger(__name__)
router = APIRouter()
_pipeline: AssistantPipeline | None = None


def get_pipeline() -> AssistantPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AssistantPipeline()
    return _pipeline


@router.post('/ask', response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail='Question cannot be empty.')
    try:
        history = [turn.model_dump() for turn in payload.history]
        return get_pipeline().ask(question, history=history)
    except RateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except UpstreamUnavailableError as exc:
        # Provider-authored reason, safe to show; other RuntimeErrors carry internals.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Ask failed')
        raise HTTPException(status_code=500, detail='Failed to answer question.') from exc
