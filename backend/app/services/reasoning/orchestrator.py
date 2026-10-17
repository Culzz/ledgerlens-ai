"""Answering a question from a document, with the arithmetic left to the engine.

This is the difference between the two tracks. Baseline A hands the model the whole
document and asks for a number, so a right figure and a wrong one come back the
same way and only the scoring step tells them apart. Here the model is asked for a
*program* in the DSL the calculation engine already replays against the document's
own cells, and the number the caller receives is the engine's.

Two things follow, and they are the reason for the design.

An answer is traceable. Every step of the arithmetic is kept, so a figure can be
shown as ``subtract(1200, 900) = 300`` instead of asserted, and each citation is a
row or a line that was actually in the evidence.

Failure is visible. A program that will not parse, or that names a row the table
does not have, returns no answer rather than a plausible one. That is the common
failure of this shape of system, and the point of returning nothing is that it can
be counted instead of dissolving into the wrong-answer pile.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from app.services.calculations import CalculationError, execute_program
from app.services.dataset import FinQAExample
from app.services.llm import Completion, LLMProvider, Message, assistant, system, user
from app.services.retrieval import TABLE, BM25Index, DocumentLike, EvidenceUnit, build_units

from .answers import extract_json_object
from .grounding import Grounding, audit, read_report

PROMPT_VERSION = "program-v3"

# How many evidence units reach the prompt. Ten is not a guess: the retrieval
# measurement on dev put complete@10 at 0.666, so a tenth unit is still buying
# something. The table's header row is added on top and does not count against it.
DEFAULT_TOP_K = 10

# Room for a short reasoning sentence and a program. Nothing here is a long-form
# answer, so this is generous rather than tight.
DEFAULT_MAX_TOKENS = 1024

# One retry by default. A second program is one more model call per question, and
# on a rate-limited free tier that is a real cost; the measured effect of the
# first retry is reported in the evaluation rather than assumed.
DEFAULT_MAX_REPAIRS = 1

PROGRAM_SYSTEM = """You answer questions about a financial report by writing a short program.

You are given part of the report: some of its table rows and some of its narrative lines. Each piece of evidence has a name, like table_3 or text_7.

A program is a comma-separated list of steps. #0, #1 and so on refer to the result of an earlier step.

Numbers:
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)
A comparison, giving yes or no:
  greater(a, b)

Every step takes plain numbers, taken from the report. Most questions are answered this way:

  subtract(11503, 10815)
  divide(637, 5)

Write a figure as digits and a decimal point, nothing else. Drop the currency symbol, drop thousands separators, and do not round: if the arithmetic comes to 127.4 then the answer is 127.4, not 127.

If the answer is a single figure the report already states, the program is that figure on its own:

  189

When one row has to be aggregated across its columns, use a table call. The row is named by its first cell, written as the table writes it, and the column is a header from the table, or the word none for the whole row:

  table_sum(total obligations, none)
  table_average(settlements, none)

The names of the evidence - table_3, text_7 - are for citing in the evidence list. They are never part of the program:

  wrong:  divide(table_1, table_2)
  right:  divide(193.5, 100)

Reply with one JSON object and nothing else:

{"reasoning": "<one short sentence>",
 "program": "<the program>",
 "evidence": ["table_3", "text_7"]}

The evidence list names the rows you read. Do not state the answer anywhere else: the last step of the program is the answer."""

# Sent back when a program would not run. The engine's own error text goes in the
# middle of it, because the errors are specific - a name the table does not have, a
# reference to a step that does not exist - and a model that is told which one it
# made fixes it far more often than a model that is told to try again.
REPAIR_INSTRUCTION = """That program did not run:

  {error}

Write it again, using only figures the report states. Do not do the arithmetic yourself, and do not put an evidence name in the program. Reply with one JSON object and nothing else."""


@dataclass(frozen=True)
class Document:
    """A table and the narrative around it, ready to be retrieved over.

    FinQA records already arrive in this shape. A report uploaded through the
    product path is turned into it during ingestion, which is what lets one
    retriever sit behind both and makes the benchmark say something about the
    product.
    """

    id: str
    table: tuple[tuple[str, ...], ...]
    context_text: tuple[str, ...]

    @classmethod
    def from_example(cls, example: FinQAExample) -> Document:
        return cls(
            id=example.id,
            table=tuple(tuple(row) for row in example.table),
            context_text=tuple(example.context_text),
        )


@dataclass(frozen=True)
class Citation:
    """One piece of evidence, named the way it was shown to the model."""

    label: str
    unit_id: str
    kind: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "unit_id": self.unit_id, "kind": self.kind, "text": self.text}


@dataclass(frozen=True)
class GroundedAnswer:
    """What came back, including the parts that are easier to hide than to report."""

    question: str
    answer: str | None
    program: str | None
    steps: tuple[str, ...]
    citations: tuple[Citation, ...]
    failure: str | None
    reply: str
    # The model call behind it, kept so a caller can show what a question cost.
    # Absent when a stored reply is replayed through ``resolve``.
    completion: Completion | None = None
    # Everything retrieval put in the prompt. The citations above are the subset
    # the reply said it used; keeping both is what lets a UI show the evidence and
    # mark what the answer rested on.
    offered: tuple[Citation, ...] = ()
    # Retrieval found nothing and the whole document went into the prompt.
    retrieval_fallback: bool = False
    # Where each figure in the program came from, and whether all of them came from
    # somewhere. Present whenever a program ran, including when the check refused it.
    grounding: Grounding | None = None
    # What the model got wrong on the attempts that did not take, oldest first. The
    # failure the caller sees is the last one; this is how many tries it took to get
    # past the others.
    repair_notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.answer is not None

    def unsourced_labels(self, offered: tuple[Citation, ...] | None = None) -> tuple[str, ...]:
        """Evidence a figure was found in that the prompt never contained.

        Not an error - the check runs against the whole report, and retrieval does
        not promise to surface the row a figure lives in. It is a useful thing to be
        able to count, because it is the only visible trace of a model answering
        from something other than what it was shown.
        """
        if self.grounding is None:
            return ()
        shown = {citation.label for citation in (offered if offered is not None else self.offered)}
        return tuple(label for label in self.grounding.labels if label not in shown)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "program": self.program,
            "steps": list(self.steps),
            "citations": [citation.to_dict() for citation in self.citations],
            "failure": self.failure,
            "retrieval_fallback": self.retrieval_fallback,
        }
        if self.grounding is not None:
            payload["grounding"] = self.grounding.to_dict()
        if self.repair_notes:
            payload["repair_notes"] = list(self.repair_notes)
        if self.completion is not None:
            payload.update(
                {
                    "provider": self.completion.provider,
                    "model": self.completion.model,
                    "total_tokens": self.completion.usage.total_tokens,
                    "cost_usd": self.completion.cost_usd,
                    "latency_seconds": self.completion.latency_seconds,
                }
            )
        return payload


@dataclass(frozen=True)
class Retrieval:
    """The evidence for one question, and whether retrieval actually found it."""

    citations: tuple[Citation, ...]
    by_label: dict[str, Citation]
    # True when nothing matched and the whole document was used instead. Worth
    # carrying, because it is the difference between a retrieval failure and a
    # reasoning failure and the two look identical in an accuracy number.
    fallback: bool = False


def label_for(unit: EvidenceUnit) -> str:
    """The short name a model sees and cites, matching FinQA's own encoding."""
    return f"{unit.kind}_{unit.index}"


def _index_of(label: str) -> int | None:
    """The row or line number behind a ``table_3`` or ``text_7`` label."""
    _, _, raw = label.partition("_")
    return int(raw) if raw.isdigit() else None


def _render_evidence(document: DocumentLike, evidence: tuple[Citation, ...]) -> list[str]:
    """Lay the evidence out the way it sits in the report.

    Table rows are printed as whole rows, pipes between cells, so a value stays
    under its own header on the way down. The first version of this handed the
    model the rows as flattened lines of text, and it read the labels as arguments
    - ``table_sum(table_3, payments volume)`` - which the engine rejected. Showing
    the row and naming it on the left is one less way to misread the table.
    """
    table: list[str] = []
    narrative: list[str] = []

    for citation in evidence:
        if citation.kind != TABLE:
            narrative.append(f"  {citation.label}: {citation.text}")
            continue

        index = _index_of(citation.label)
        row = document.table[index] if index is not None and index < len(document.table) else None
        cells = " | ".join(str(cell).strip() for cell in row) if row else citation.text
        table.append(f"  {citation.label}: {cells}")

    lines: list[str] = []
    if table:
        lines.append("Table rows:")
        lines.extend(table)
        lines.append("")
    if narrative:
        lines.append("Narrative:")
        lines.extend(narrative)
        lines.append("")
    return lines


def build_program_messages(
    document: DocumentLike,
    evidence: tuple[Citation, ...],
    question: str,
) -> list[Message]:
    """The prompt: the retrieved evidence, then the question, and nothing else."""
    lines = [f"Report: {document.id}", ""]
    lines.extend(_render_evidence(document, evidence))
    lines.append(f"Question: {question}")
    return [system(PROGRAM_SYSTEM), user("\n".join(lines))]


def _combine(completions: list[Completion]) -> Completion | None:
    """Roll the calls behind one answer into a single record.

    A repaired answer took more than one call, and reporting only the last one
    would understate what the question cost and how much quota it used. Cost is
    summed only when every call had a known price: a partial sum reported as a
    total would be the silent zero this project refuses elsewhere.
    """
    if not completions:
        return None
    if len(completions) == 1:
        return completions[0]

    usage = completions[0].usage
    for completion in completions[1:]:
        usage = usage + completion.usage

    costs = [completion.cost_usd for completion in completions]
    return replace(
        completions[-1],
        usage=usage,
        latency_seconds=sum(completion.latency_seconds for completion in completions),
        attempts=sum(completion.attempts for completion in completions),
        cost_usd=None if any(cost is None for cost in costs) else sum(costs),  # type: ignore[arg-type]
    )


class Orchestrator:
    """Retrieve, write a program, run it, and keep the receipts."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        prompt_version: str = PROMPT_VERSION,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
        enforce_grounding: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        self._provider = provider
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
        self.max_repairs = max_repairs
        self.enforce_grounding = enforce_grounding

    @property
    def name(self) -> str:
        return f"orchestrator@{self._provider.name}"

    def retrieve(self, document: DocumentLike, question: str) -> Retrieval:
        """The evidence for one question, header row first.

        The header always goes in. Reading a table means knowing what its columns
        are called, and a system that has the table has them - withholding the
        header would measure whether BM25 happened to surface it, which is not
        the question anyone is asking.
        """
        units = build_units(document)
        index = BM25Index()
        index.add_many((unit.id, unit.text) for unit in units)
        index.finalize()

        by_id = {unit.id: unit for unit in units}
        ranked = [by_id[doc_id] for doc_id, _ in index.search(question, top_k=self.top_k)]

        # Nothing matched at all. Answering from an empty evidence block fails for
        # a reason that says nothing about the system, so fall back to the whole
        # document - the same view baseline A gets. It is only reached when no
        # token in the question appears anywhere in the document, and the flag
        # goes back to the caller so those questions can be counted rather than
        # quietly mixed in.
        fallback = not ranked
        if fallback:
            ranked = list(units)

        if units and units[0].is_table:
            header = units[0]
            ranked = [header, *(unit for unit in ranked if unit.id != header.id)]

        citations = tuple(
            Citation(label=label_for(unit), unit_id=unit.id, kind=unit.kind, text=unit.text)
            for unit in ranked
        )
        return Retrieval(
            citations=citations,
            by_label={citation.label: citation for citation in citations},
            fallback=fallback,
        )

    def answer(self, document: DocumentLike, question: str) -> GroundedAnswer:
        """Answer one question about one document, retrying a program that fails.

        A program that will not run is usually nearly right: a row named the way the
        narrative writes it instead of the way the table does, or a figure rounded
        off. The engine's error says which, so it goes back to the model with the
        prompt intact. The retry budget is small on purpose - every repair is another
        call against a free quota - and the measured effect is in the evaluation
        rather than assumed.
        """
        retrieval = self.retrieve(document, question)
        messages = build_program_messages(document, retrieval.citations, question)

        completions: list[Completion] = []
        notes: list[str] = []
        grounded: GroundedAnswer | None = None

        for _ in range(self.max_repairs + 1):
            completion = self._provider.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            completions.append(completion)
            grounded = self.resolve(question, completion.text, document, retrieval.by_label)

            if grounded.ok:
                break
            notes.append(grounded.failure or "the program did not run")
            messages = [
                *messages,
                assistant(completion.text),
                user(REPAIR_INSTRUCTION.format(error=notes[-1])),
            ]

        assert grounded is not None
        return replace(
            grounded,
            # Every failure along the way, oldest first. On a recovered answer these
            # are the ones that were got past; on an exhausted one the last entry is
            # also the failure the caller sees.
            repair_notes=tuple(notes),
            completion=_combine(completions),
            offered=retrieval.citations,
            retrieval_fallback=retrieval.fallback,
        )

    def resolve(
        self,
        question: str,
        reply: str,
        document: DocumentLike,
        by_label: dict[str, Citation],
    ) -> GroundedAnswer:
        """Run the program in a reply against the document, or say why not."""
        payload = extract_json_object(reply)
        program = payload.get("program") if isinstance(payload, dict) else None

        citations = self._citations(payload, by_label)
        # Everything the caller put in the prompt, in the order it was offered, so
        # ``unsourced_labels`` can tell a figure read from a shown row apart from one
        # found elsewhere in the report without the caller passing it back in.
        offered = tuple(by_label.values())

        def failed(reason: str, grounding: Grounding | None = None) -> GroundedAnswer:
            return GroundedAnswer(
                question=question,
                answer=None,
                program=str(program).strip() if program else None,
                steps=(),
                citations=citations,
                failure=reason,
                reply=reply,
                grounding=grounding,
                offered=offered,
            )

        if not program or not str(program).strip():
            return failed("the reply carried no program")

        try:
            trace = execute_program(str(program).strip(), table=document.table)
        except CalculationError as exc:
            # Named and kept. A plausible number here would be the exact failure
            # this architecture exists to make impossible.
            return failed(f"{type(exc).__name__}: {exc}")

        grounding = audit(trace, read_report(document))

        if self.enforce_grounding and grounding.blocks:
            # Every figure the program used is one the report does not state, so the
            # answer is arithmetic on numbers of the model's own making. It is the
            # one case where a program that *ran* is still not an answer.
            unsourced = ", ".join(grounding.unsourced) or "no figure"
            return failed(f"UngroundedAnswer: the report states none of {unsourced}", grounding)

        return GroundedAnswer(
            question=question,
            answer=str(trace.result),
            program=str(program).strip(),
            # The engine's own rendering. The citations on the operands are in
            # ``grounding``, kept apart so that a step line keeps meaning the same
            # thing in every run this project has recorded.
            steps=tuple(step.render() for step in trace.steps),
            citations=citations,
            failure=None,
            reply=reply,
            grounding=grounding,
            offered=offered,
        )

    def _citations(self, payload: Any, by_label: dict[str, Citation]) -> tuple[Citation, ...]:
        """The units the reply says it read, minus any label that was never offered.

        A citation to a label that was not in the prompt is dropped rather than
        recorded. Trusting it would let the model invent a source, and an invented
        source is worse than no source because it looks like one.
        """
        if not isinstance(payload, dict):
            return ()
        claimed = payload.get("evidence")
        if not isinstance(claimed, list):
            return ()

        picked: list[Citation] = []
        seen: set[str] = set()
        for item in claimed:
            label = str(item)
            citation = by_label.get(label)
            if citation is not None and label not in seen:
                seen.add(label)
                picked.append(citation)
        return tuple(picked)
