"""Running the same program with one of its figures changed.

This is the other half of what it means for the answer to be a program rather than
a sentence. A sentence cannot be asked a follow-up without asking the model again.
A program can, and the engine answers in the time it takes to do the arithmetic -
no provider, no tokens, no waiting, and the same figure every time.

No model is called here, and there is nothing in this module that could call one.

The grounding verdict comes back with the result and is deliberately *not* used to
refuse it. A what-if has changed a figure the report states, so the new figure is
by definition not in the report and the verdict will say so. That is the honest
reading and the reason a what-if is not an answer: the arithmetic is the engine's,
and the input is the reader's.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import require_library
from app.schemas.api import (
    GroundingOut,
    RecomputeRequest,
    RecomputeResponse,
)
from app.services.calculations import CalculationError, execute_program, substitute_figures
from app.services.reasoning import Document, audit, read_report

router = APIRouter(tags=["questions"])


@router.post("/recompute", response_model=RecomputeResponse)
def recompute(request: RecomputeRequest) -> RecomputeResponse:
    library = require_library()
    examples = library.get(request.document_id)
    if not examples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no document {request.document_id!r}",
        )

    document = Document.from_example(examples[0])
    edits = [(edit.step, edit.operand, edit.value) for edit in request.edits]

    try:
        program = substitute_figures(request.program, edits)
    except CalculationError as exc:
        # The request cannot be turned into a program at all: a figure that is not a
        # figure, a position the program does not have, or a program that will not
        # parse. That is the caller's mistake and it is answered as one.
        raise HTTPException(
            # 422 rather than a named constant: Starlette has deprecated the name for
            # it, and the numeric code is what FastAPI's own request validation
            # already returns. One number for one kind of mistake.
            status_code=422,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        trace = execute_program(program, table=document.table)
    except CalculationError as exc:
        # The program is well formed and the arithmetic has no answer - someone set a
        # divisor to zero. That is a real thing to have asked, so it comes back with
        # the engine's reason against the program that was actually run, rather than
        # as a failure of the request.
        return RecomputeResponse(
            program=program,
            answer=None,
            failure=f"{type(exc).__name__}: {exc}",
        )

    grounding = audit(trace, read_report(document))

    return RecomputeResponse(
        program=program,
        answer=str(trace.result),
        audit=GroundingOut.model_validate(grounding.to_dict()),
        failure=None,
    )
