"""Turning a session's questions back into a brief the server can stand behind.

The brief a reader saves is a list of questions and the programs that answered
them. Replaying one is a call to ``Orchestrator.resolve`` with that program where
the model's reply would go, which is the same function that ran it the first time:
the same parser, the same engine, the same check, and the same sentence on the
failure path. A second implementation of those four steps would be a second place
for them to disagree, and the disagreement would show up as a shared page that
disagrees with the page it was shared from.

No model is called. ``resolve`` is the model-free half of the answering path, and
the only thing it needs from a reply is the program.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.reasoning import GroundedAnswer, Orchestrator
from app.services.reasoning.orchestrator import DocumentLike

NO_PROGRAM = "the model returned no program for this question"


def replay(
    orchestrator: Orchestrator,
    document: DocumentLike,
    question: str,
    program: str | None,
) -> dict[str, Any]:
    """One brief row, recomputed here rather than taken from the caller."""
    if not program or not program.strip():
        return _row(question, None, None, NO_PROGRAM)

    # The shape ``resolve`` parses: an object with a program in it. Evidence is
    # absent because there is no reply to quote and no citation to claim.
    reply = json.dumps({"program": program})
    grounded = orchestrator.resolve(question, reply, document, {})
    return _row(question, grounded.program, grounded, grounded.failure)


def _row(
    question: str,
    program: str | None,
    answer: GroundedAnswer | None,
    note: str | None,
) -> dict[str, Any]:
    grounding = answer.grounding if answer is not None else None
    return {
        "question": question,
        "program": program,
        "answer": answer.answer if answer is not None else None,
        "verdict": grounding.verdict if grounding is not None else None,
        "unsourced": list(grounding.unsourced) if grounding is not None else [],
        "sourced_count": grounding.sourced_count if grounding is not None else 0,
        "literal_count": grounding.literal_count if grounding is not None else 0,
        "labels": list(grounding.labels) if grounding is not None else [],
        "note": note,
    }
