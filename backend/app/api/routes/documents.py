"""Health, and the documents the demo can be asked about."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import get_library, get_provider, require_library
from app.schemas.api import (
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    HealthResponse,
)

router = APIRouter(tags=["documents"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    library = get_library()
    provider = get_provider()
    documents = library.summaries()
    return HealthResponse(
        status="ok",
        provider=provider.name,
        model=provider.model or None,
        documents=len(documents),
        documents_loaded=library.available,
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(q: str | None = None, limit: int | None = 60) -> DocumentListResponse:
    library = require_library()
    if limit is not None:
        limit = max(1, min(limit, 500))
    matches = library.summaries(query=q)
    return DocumentListResponse(
        documents=[DocumentSummary(**summary.to_dict()) for summary in matches[:limit]],
        total=len(matches),
        limit=limit,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str) -> DocumentDetail:
    library = require_library()
    examples = library.get(document_id)
    if not examples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no document {document_id!r}"
        )

    first = examples[0]
    return DocumentDetail(
        id=document_id,
        filename=first.id.rsplit("-", 1)[0],
        table=[list(row) for row in first.table],
        context=list(first.context_text),
        questions=[example.question for example in examples],
    )
