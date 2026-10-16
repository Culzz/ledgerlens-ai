"""Prompt text, versioned.

The evaluation protocol requires every run to record which prompt produced it.
That makes the wording data rather than something scattered through the systems,
so it can be diffed and so a version bump is an explicit act. Two runs sharing a
prompt version that disagree are then a real disagreement.
"""

from __future__ import annotations

from app.services.dataset import FinQAExample
from app.services.llm import Message, system, user

from .base import render_narrative, render_table

DIRECT_PROMPT_VERSION = "direct-v1"

DIRECT_SYSTEM = """You answer questions about a financial document.

You are given the document's table and its narrative text. Both are supplied in
full. Use only figures that appear in the document.

Work the answer out and reply with one short line of reasoning, then a final
line in exactly this form:

ANSWER: <value>

The value is a single number, or yes or no. No units, no thousands separators,
and nothing after it."""


def build_direct_messages(example: FinQAExample) -> list[Message]:
    """Baseline A: the whole document, no retrieval, the model does the arithmetic."""
    document = "\n\n".join(
        [
            "Narrative text:",
            render_narrative(example),
            "Table:",
            render_table(example.table),
        ]
    )
    return [
        system(DIRECT_SYSTEM),
        user(f"{document}\n\nQuestion: {example.question}"),
    ]


PROMPTS = {"direct": DIRECT_PROMPT_VERSION}
