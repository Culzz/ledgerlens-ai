"""Request and response shapes for the API.

Kept in one place so the routes read as routing and the frontend has a single
file to match against.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str | None
    documents: int
    # True when the documents are on disk. False means the split has not been
    # downloaded yet, which is a setup step rather than a fault.
    documents_loaded: bool


class DocumentSummary(BaseModel):
    id: str
    filename: str
    questions: list[str]
    table_rows: int
    context_lines: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    # How many matched, so the UI can say what it is not showing.
    total: int
    limit: int | None = None


class DocumentDetail(BaseModel):
    id: str
    filename: str
    table: list[list[str]]
    context: list[str]
    questions: list[str]


class EvidenceOut(BaseModel):
    """One retrieved unit, and whether the answer rested on it."""

    label: str
    kind: str
    text: str
    cited: bool


class AskRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=600)


class AskResponse(BaseModel):
    question: str
    # The engine's result, or null. Never a number taken from the model's reply.
    answer: str | None
    program: str | None
    steps: list[str]
    failure: str | None
    evidence: list[EvidenceOut]
    # The unit ids the program's steps were checked against, for the UI to link.
    retrieval_fallback: bool
    provider: str
    model: str | None
    total_tokens: int
    cost_usd: float | None
    latency_seconds: float
    # Looked up after the answer is computed, for display only. The answering path
    # never sees it.
    reference_answer: str | None
