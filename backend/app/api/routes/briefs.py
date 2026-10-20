"""Saving a brief and handing it back by link.

The request carries questions and programs. It does not carry figures, verdicts or
anything else that came back from the model, and that is the design of the module in
one sentence: every line of a saved brief is produced by running the stored program
again against the stored report and auditing the result here. A saved page built from
what the browser had on screen would be a photograph of an answer, and the two drift.

That is also why a share link is safe to hand out. It cannot contain a figure the
server did not compute, and reopening it re-derives the whole page from the report
rather than from whatever was true when it was saved.

No model is called. The replay path is the model-free half of the answering path, so
this is a pair of ordinary database calls behind a program run.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, status

from app.api.deps import get_orchestrator, require_library
from app.db.repositories import briefs as store
from app.schemas.api import BriefOut, BriefRequest, BriefRowOut
from app.services.briefs import replay
from app.services.reasoning import Document

router = APIRouter(tags=["briefs"])

# Ids are minted here and are URL-safe, so anything else is a 404 before it reaches
# the database.
BRIEF_ID = r"^[A-Za-z0-9_-]{8,64}$"


def _out(record: dict) -> BriefOut:
    return BriefOut(
        id=record["id"],
        document_id=record["document_id"],
        filename=record["filename"],
        created_at=record["created_at"],
        rows=[BriefRowOut(**row) for row in record["rows"]],
    )


@router.post("/briefs", response_model=BriefOut, status_code=status.HTTP_201_CREATED)
def save_brief(request: BriefRequest) -> BriefOut:
    library = require_library()
    examples = library.get(request.document_id)
    if not examples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no document {request.document_id!r}",
        )

    document = Document.from_example(examples[0])
    orchestrator = get_orchestrator()
    rows = [replay(orchestrator, document, row.question, row.program) for row in request.rows]

    try:
        brief_id = store.save(request.document_id, examples[0].report_filename, rows)
        # Read back rather than echo. What the caller is shown is then what was
        # written, and a store that dropped a row cannot come back looking complete.
        record = store.load(brief_id)
    except store.BriefsUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the brief store did not return the brief it had just written",
        )
    return _out(record)


@router.get("/briefs/{brief_id}", response_model=BriefOut)
def read_brief(brief_id: str = Path(pattern=BRIEF_ID)) -> BriefOut:
    try:
        record = store.load(brief_id)
    except store.BriefsUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such brief")
    return _out(record)
