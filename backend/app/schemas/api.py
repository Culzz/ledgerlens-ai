"""Request and response shapes for the API.

Kept in one place so the routes read as routing and the frontend has a single
file to match against.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# The engine's own limit, imported rather than restated so a request cannot be
# accepted here that the engine would refuse for length.
from app.services.calculations import MAX_STEPS


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


class OperandOut(BaseModel):
    """One argument of a step, and where its number came from."""

    text: str
    # reference | label | sourced | constant | unsourced | nested. The UI says
    # different things for each: a sourced figure cites a row, an unsourced one is
    # the reason an answer was refused.
    role: str
    value: str | None = None
    # The units of the report that state this figure. Empty for anything that is
    # not a sourced figure.
    labels: list[str] = []


class AuditStepOut(BaseModel):
    index: int
    operation: str
    result: str
    # The step as one line, with each figure annotated. The same text the engine
    # prints, plus citations.
    render: str
    operands: list[OperandOut] = []


class GroundingOut(BaseModel):
    """The answer to "did these numbers come from the report"."""

    # grounded | partly_grounded | ungrounded. Anything but the first is worth
    # showing; the third is why an answer was refused.
    verdict: str
    labels: list[str] = []
    # Figures the report does not state, as the program wrote them.
    unsourced: list[str] = []
    sourced_count: int = 0
    literal_count: int = 0
    constant_only: bool = False
    # What the report said its figures were in, when it said. Shown so a reader can
    # see why a figure written out in full still matched.
    report_unit: str | None = None
    steps: list[AuditStepOut] = []


class AskRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=600)


class AskResponse(BaseModel):
    question: str
    # The engine's result, or null. Never a number taken from the model's reply.
    answer: str | None
    program: str | None
    # Where every figure in that program came from. Present whenever a program ran,
    # including when this is the reason there is no answer.
    audit: GroundingOut | None
    failure: str | None
    evidence: list[EvidenceOut]
    # How many times a failed program was sent back before this answer. Zero is the
    # common case; anything else means the first attempt did not run.
    repair_count: int
    retrieval_fallback: bool
    provider: str
    model: str | None
    total_tokens: int
    cost_usd: float | None
    latency_seconds: float
    # Looked up after the answer is computed, for display only. The answering path
    # never sees it.
    reference_answer: str | None


class FigureEdit(BaseModel):
    """One figure to write differently, named by where it sits in the program.

    A position rather than a value: a program that uses 150 in two places has two
    places to change, and changing "the 150" would have to guess which.
    """

    step: int = Field(ge=0, le=MAX_STEPS - 1)
    operand: int = Field(ge=0, le=MAX_STEPS - 1)
    # A number only. The engine parses it and prints it back, so nothing sent here
    # can arrive as program syntax.
    value: str = Field(min_length=1, max_length=40)


class RecomputeRequest(BaseModel):
    """The same program with some of its figures changed. No model is called."""

    document_id: str = Field(min_length=1, max_length=200)
    program: str = Field(min_length=1, max_length=2000)
    edits: list[FigureEdit] = Field(min_length=1, max_length=32)


class RecomputeResponse(BaseModel):
    program: str
    # The engine's result. It stands even when the verdict below says the figures are
    # no longer the report's - a changed figure is a question, not an answer, and a
    # what-if has to be allowed to leave the report.
    answer: str | None
    # Absent only when the arithmetic itself has no answer, such as a division by the
    # zero someone just typed. That is a legitimate hypothetical rather than a bad
    # request, so it comes back as a failure on a 200 the way a program that will not
    # run does on /api/ask.
    audit: GroundingOut | None = None
    failure: str | None = None
