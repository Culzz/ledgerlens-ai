"""What a system under test produces, and the shape every system shares.

A system takes one FinQA example and returns a prediction. Keeping that
interface this small is what lets a new system be added without the runner
knowing anything about it, and lets the runner be tested against a fake model
with no network at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.services.dataset import FinQAExample
from app.services.reasoning import ParsedAnswer


@dataclass
class Prediction:
    """One system's answer to one question, with what it cost to get it."""

    system: str
    question_id: str
    parsed: ParsedAnswer
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Total as the provider counted it, which for reasoning models is more than
    # prompt + completion, plus the difference stated outright.
    total_tokens: int = 0
    reasoning_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float | None = None
    attempts: int = 1
    call_error: str | None = None
    # "length" means the provider cut the answer off. Worth keeping, because a
    # truncated answer looks like a wrong answer and nothing else would say so.
    finish_reason: str | None = None
    # Whatever else the system thinks belongs in the record. A program-shaped
    # system puts the program and its steps here; one that answers directly leaves
    # it empty. Keeping it generic means the runner stays ignorant of both.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.parsed.raw

    def to_record(self) -> dict[str, Any]:
        """A flat, JSON-safe form. The runner writes this to the predictions file."""
        record = asdict(self)
        record["parsed"] = {
            "answer": self.parsed.answer,
            "format": self.parsed.format,
            "raw": self.parsed.raw,
            "reasoning": self.parsed.reasoning,
            "evidence": list(self.parsed.evidence),
        }
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Prediction:
        parsed = record.get("parsed") or {}
        return cls(
            system=record.get("system", ""),
            question_id=record.get("question_id", ""),
            parsed=ParsedAnswer(
                answer=parsed.get("answer"),
                format=parsed.get("format", "unparsed"),
                raw=parsed.get("raw", ""),
                reasoning=parsed.get("reasoning", ""),
                evidence=tuple(parsed.get("evidence") or ()),
            ),
            provider=record.get("provider", ""),
            model=record.get("model", ""),
            prompt_tokens=record.get("prompt_tokens", 0),
            completion_tokens=record.get("completion_tokens", 0),
            total_tokens=record.get("total_tokens", 0),
            reasoning_tokens=record.get("reasoning_tokens", 0),
            latency_seconds=record.get("latency_seconds", 0.0),
            cost_usd=record.get("cost_usd"),
            attempts=record.get("attempts", 1),
            call_error=record.get("call_error"),
            finish_reason=record.get("finish_reason"),
            detail=dict(record.get("detail") or {}),
        )


@runtime_checkable
class System(Protocol):
    """A system under test."""

    name: str
    prompt_version: str

    def answer(self, example: FinQAExample) -> Prediction: ...


def render_table(table: Sequence[Sequence[str]]) -> str:
    """Render a FinQA table as a pipe table.

    Padding short rows keeps the rendered grid rectangular, because ragged pipe
    tables are genuinely hard for a model to read and FinQA tables are sometimes
    ragged.
    """
    if not table:
        return "(the document has no table)"

    rows = [[str(cell).strip() for cell in row] for row in table]
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]

    header, *body = padded
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def render_narrative(example: FinQAExample) -> str:
    """Narrative lines, numbered from zero across pre_text then post_text.

    The numbering matches the ``text_N`` evidence ids the retrieval layer uses,
    so a system that cites evidence is citing the same indices everything else
    in the project counts in.
    """
    lines = [f"[{i}] {line}" for i, line in enumerate(example.context_text)]
    return "\n".join(lines) if lines else "(the document has no narrative text)"
