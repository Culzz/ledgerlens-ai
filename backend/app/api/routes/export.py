"""Handing back an answer as a workbook.

The file is rebuilt rather than forwarded, and that is the whole design of this
module. The caller sends the question and the program, not the figures and not the
trail; the program is executed again against the same document here, the check is
run again here, and the workbook holds what came out of that. A file assembled from
what the browser had on screen would be a picture of an answer, and pictures and
answers drift apart.

No model is called, and nothing here could call one. Retrieval and the engine are
both model-free, which is why this is a POST that returns immediately rather than a
second round trip through the provider.

The evidence sheet is the retrieval this process just ran, with the same retriever
the answer was served from. The column saying which rows the answer rested on is the
check's own finding - the rows a figure in the program was found in - rather than the
reply's list of citations. That is the reading that can be recomputed here, since the
reply is not part of the request, and it is the one the audit exists to produce;
where it differs from what the reply claimed is exactly the case the check is for.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.api.deps import get_orchestrator, require_library
from app.schemas.api import ExportRequest
from app.services.calculations import CalculationError, execute_program
from app.services.reasoning import Document, audit, read_report
from app.services.reporting import AnswerReport, EvidenceRow, filename_for, workbook_bytes

# The type Excel and LibreOffice both expect for this file.
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

router = APIRouter(tags=["questions"])


@router.post("/export/xlsx")
def export_xlsx(request: ExportRequest) -> Response:
    library = require_library()
    examples = library.get(request.document_id)
    if not examples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no document {request.document_id!r}",
        )

    document = Document.from_example(examples[0])

    try:
        trace = execute_program(request.program, table=document.table)
    except CalculationError as exc:
        # A program that will not run has no workbook to make. Same answer as
        # /api/recompute gives for the same reason, and the same number.
        raise HTTPException(
            status_code=422,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    grounding = audit(trace, read_report(document))
    retrieval = get_orchestrator().retrieve(document, request.question)

    rested_on = set(grounding.labels)
    report = AnswerReport(
        question=request.question,
        document_id=request.document_id,
        filename=examples[0].report_filename,
        answer=str(trace.result),
        program=request.program,
        failure=None,
        verdict=grounding.verdict,
        unsourced=grounding.unsourced,
        sourced_count=grounding.sourced_count,
        literal_count=grounding.literal_count,
        report_unit=grounding.report_unit,
        steps=grounding.steps,
        evidence=tuple(
            EvidenceRow(
                label=citation.label,
                kind=citation.kind,
                cited=citation.label in rested_on,
                text=citation.text,
            )
            for citation in retrieval.citations
        ),
        provider=request.provider or "not recorded",
        model=request.model,
    )

    return Response(
        content=workbook_bytes(report),
        media_type=XLSX,
        headers={
            "Content-Disposition": f'attachment; filename="{filename_for(request.document_id)}"',
        },
    )
