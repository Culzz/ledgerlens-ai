"""The answer, the trail and the evidence, as a workbook.

An answer is a number. The reason it can be trusted is a set of facts about that
number - which report it came from, which rows it read, which of its figures the
report actually states - and on screen those facts live in a card that is pleasant
to read and useless to re-use. A spreadsheet is what an analyst keeps, sorts and
pastes into something else.

Three sheets, and the split is the point:

``Answer``
    One line per fact: the question, the figure, the program, and what the check
    made of it.

``Trail``
    The audit, a row per step and per operand, so a figure that came from nowhere is
    a row in a table rather than a sentence in a paragraph.

``Evidence``
    What retrieval put in front of the model, and which of it the answer rested on.
    The two columns differ, and that difference is worth being able to see: a row can
    be offered and not used, and a row can be used without the reply saying so.

Nothing here decides anything or computes anything. Every value arrives already
computed, by the same code that answered the question, and this module only lays it
out - which is why it imports no model, no retriever and no engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.worksheet import Worksheet

from app.services.reasoning.grounding import (
    CONSTANT,
    LABEL,
    NESTED,
    REFERENCE,
    SOURCED,
    UNSOURCED,
    AuditStep,
)

# What a figure's origin is called in a spreadsheet. The card on screen says the
# same things in the same words, because two vocabularies for one idea is how a
# reader ends up believing they are two ideas.
ORIGINS = {
    SOURCED: "from the report",
    CONSTANT: "arithmetic",
    UNSOURCED: "not in the report",
    LABEL: "a row or column name",
    REFERENCE: "an earlier step",
    NESTED: "a calculation",
}

# The worksheet palette, borrowed from the card so the two do not disagree about
# what this system looks like. Same ochre, same ink.
HEADER_FILL = PatternFill("solid", fgColor="FBF3D6")
HEADER_FONT = Font(bold=True, color="1B1814")
FIELD_FONT = Font(bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")

ANSWER_WIDTHS = (30, 96)
TRAIL_WIDTHS = (6, 14, 26, 60, 30, 26, 18)
TRAIL_HEADERS = (
    "Step",
    "Operation",
    "Result",
    "Step as written",
    "Operand",
    "Where the figure came from",
    "Rows",
)
EVIDENCE_HEADERS = ("Row", "Kind", "Used in the answer", "Text")
EVIDENCE_WIDTHS = (10, 8, 18, 110)


@dataclass(frozen=True)
class EvidenceRow:
    """One unit that was put in front of the model."""

    label: str
    kind: str
    cited: bool
    text: str


@dataclass(frozen=True)
class AnswerReport:
    """Everything a workbook needs, and nothing that needs deciding."""

    question: str
    document_id: str
    filename: str
    answer: str | None
    program: str | None
    failure: str | None
    # The check's verdict and what it objected to.
    verdict: str
    unsourced: tuple[str, ...]
    sourced_count: int
    literal_count: int
    report_unit: str | None
    steps: tuple[AuditStep, ...]
    evidence: tuple[EvidenceRow, ...]
    provider: str
    model: str | None
    # A factory rather than a default: a plain default is evaluated once, when the
    # class is defined, so every report would carry the time the module was imported.
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def as_text(value: Decimal | str) -> str:
    """A computed value as it should be written down.

    As text, deliberately. The engine works in exact decimals - a ratio here is 27
    significant digits - and a spreadsheet's own number type holds about 15, so
    writing one as a number would quietly round the answer this whole system exists
    to get right. Text is what the engine said.
    """
    return str(value)


def _wide(sheet: Worksheet, widths: tuple[int, ...]) -> None:
    for index, width in enumerate(widths):
        sheet.column_dimensions[chr(ord("A") + index)].width = width


def _header(sheet: Worksheet, headers: tuple[str, ...]) -> None:
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
    sheet.freeze_panes = "A2"


def _answer_sheet(sheet: Worksheet, report: AnswerReport) -> None:
    sheet.title = "Answer"
    _wide(sheet, ANSWER_WIDTHS)

    rows: list[tuple[str, str | None]] = [
        ("Question", report.question),
        ("Report", f"{report.filename}  ({report.document_id})"),
        ("Answer", report.answer if report.answer is not None else "no answer"),
        ("Program", report.program),
        ("Check", report.verdict.replace("_", " ")),
        (
            "Figures",
            f"{report.sourced_count} read from the report, "
            f"{len(report.unsourced)} not found"
            + (f", {report.report_unit}" if report.report_unit else ""),
        ),
    ]
    if report.unsourced:
        rows.append(("Not in the report", ", ".join(report.unsourced)))
    if report.failure:
        rows.append(("Why there is no answer", report.failure))
    rows.extend(
        [
            ("Model", f"{report.provider} / {report.model or 'no model set'}"),
            ("Exported", report.generated_at.isoformat(timespec="seconds")),
        ]
    )

    for label, text in rows:
        sheet.append([label, text])
    for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row):
        row[0].font = FIELD_FONT
        row[1].alignment = WRAP

    sheet.append([])
    for note in (
        (
            "The figure above was computed by the engine from this report. The program "
            "is the model's; where its figures came from is on the Trail sheet."
        ),
        (
            "Figures are written as text, at the full precision they were computed at, "
            "which a spreadsheet's number type would round."
        ),
    ):
        sheet.append([note])
    for row in sheet.iter_rows(min_row=sheet.max_row - 1, max_row=sheet.max_row):
        row[0].alignment = WRAP


def _trail_sheet(sheet: Worksheet, report: AnswerReport) -> None:
    sheet.title = "Trail"
    _wide(sheet, TRAIL_WIDTHS)
    _header(sheet, TRAIL_HEADERS)

    if not report.steps:
        sheet.append(["", "", "no program ran", "", "", ""])
        return

    for step in report.steps:
        # The step's own columns are written once, on its first row, and left blank
        # beside the remaining operands. A step with three figures is one step, not
        # three, and repeating its result on every line of it would read as three.
        lead = [step.index, step.operation, as_text(step.result), step.render()]
        for offset, operand in enumerate(step.operands):
            sheet.append(
                [
                    *(lead if offset == 0 else ["", "", "", ""]),
                    operand.text,
                    ORIGINS.get(operand.role, operand.role),
                    ", ".join(operand.labels),
                ]
            )
        if not step.operands:
            sheet.append([*lead, "", "", ""])

    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        for cell in row:
            cell.alignment = WRAP


def _evidence_sheet(sheet: Worksheet, report: AnswerReport) -> None:
    sheet.title = "Evidence"
    _wide(sheet, EVIDENCE_WIDTHS)
    _header(sheet, EVIDENCE_HEADERS)

    if not report.evidence:
        sheet.append(["", "", "", "nothing was retrieved for this question"])
        return

    for row in report.evidence:
        sheet.append([row.label, row.kind, "used" if row.cited else "offered", row.text])
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        row[3].alignment = WRAP


def workbook_bytes(report: AnswerReport) -> bytes:
    """The report as the bytes of an .xlsx file."""
    workbook = Workbook()
    _answer_sheet(workbook.active, report)
    _trail_sheet(workbook.create_sheet(), report)
    _evidence_sheet(workbook.create_sheet(), report)

    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def filename_for(document_id: str, generated_at: datetime | None = None) -> str:
    """A download name that is safe to put in a header.

    Built from the document id, which comes from a filename in the corpus and is
    therefore not trusted to be a filename itself. Everything outside a conservative
    set is dropped rather than escaped, because a header is the wrong place to find
    out that an escape was wrong.
    """
    kept = [character if character.isalnum() or character in "-_." else "-" for character in document_id]
    stem = "".join(kept).strip("-.") or "answer"
    stamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"ledgerlens-{stem}-{stamp}.xlsx"
