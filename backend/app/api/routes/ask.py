"""Asking a question about a document.

The order of operations here is the point of the whole project. The question and
the document go to the orchestrator, which retrieves evidence, asks the model for
a program, and runs that program. Only once the answer exists is the reference
looked up, and it is looked up to display, never to compute. ``exe_ans`` does not
enter anything upstream of the answer.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import get_orchestrator, require_library
from app.schemas.api import AskRequest, AskResponse, EvidenceOut, GroundingOut
from app.services.llm import BudgetExceeded, LLMError, ProviderNotConfigured, ProviderRateLimited
from app.services.reasoning import Document

router = APIRouter(tags=["questions"])

# Provider trouble the caller can act on: not configured, out of quota, or over
# budget. Everything else from the adapter is a bad upstream call.
UNAVAILABLE = (ProviderNotConfigured, ProviderRateLimited, BudgetExceeded)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    library = require_library()
    examples = library.get(request.document_id)
    if not examples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no document {request.document_id!r}",
        )

    document = Document.from_example(examples[0])
    try:
        grounded = get_orchestrator().answer(document, request.question)
    except UNAVAILABLE as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    cited = {citation.label for citation in grounded.citations}
    evidence = [
        EvidenceOut(
            label=citation.label,
            kind=citation.kind,
            text=citation.text,
            cited=citation.label in cited,
        )
        for citation in grounded.offered
    ]

    # Display only, and after the fact. A hand-typed question has no reference.
    record = library.example_for(request.document_id, request.question)
    completion = grounded.completion

    # The wire shape is the grounding module's own, validated rather than rebuilt
    # here, so the trail a reader sees and the trail the check reasoned over cannot
    # drift apart.
    audit = (
        GroundingOut.model_validate(grounded.grounding.to_dict()) if grounded.grounding else None
    )

    return AskResponse(
        question=grounded.question,
        answer=grounded.answer,
        program=grounded.program,
        audit=audit,
        failure=grounded.failure,
        evidence=evidence,
        repair_count=len(grounded.repair_notes),
        retrieval_fallback=grounded.retrieval_fallback,
        provider=completion.provider if completion else "unknown",
        model=(completion.model if completion else None) or None,
        total_tokens=completion.usage.total_tokens if completion else 0,
        cost_usd=completion.cost_usd if completion else None,
        latency_seconds=completion.latency_seconds if completion else 0.0,
        reference_answer=str(record.exe_ans) if record is not None else None,
    )
