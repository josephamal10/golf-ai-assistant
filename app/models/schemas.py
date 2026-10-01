from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=20000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Earlier turns, oldest first. Only the last few are used to rewrite follow-ups.
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)


class Source(BaseModel):
    n: int
    title: str
    url: str
    category: str
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    # static: knowledge base · live / prediction: Google Search · off_topic: not golf
    route: Literal['static', 'live', 'prediction', 'off_topic']
    route_reason: str = ''
    # The standalone question used for search and generation; equals the question
    # unless a follow-up was rewritten using the history.
    search_query: str = ''
    # Google's search-suggestion chips; its terms require showing them with a web answer.
    search_suggestions_html: str = ''
    retrieved_chunk_ids: list[str]
    insufficient_context: bool = False
    provider: str = ''
    model: str = ''
