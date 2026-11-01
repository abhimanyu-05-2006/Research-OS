from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    document_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class Citation(BaseModel):
    citation_id: str
    document_name: str
    page_number: int
    chunk_id: str
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    limitations: list[str]