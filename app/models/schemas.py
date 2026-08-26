from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(BaseModel):
    n: int
    title: str
    url: str
    category: str
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    route: str
    retrieved_chunk_ids: list[str]
    insufficient_context: bool = False
