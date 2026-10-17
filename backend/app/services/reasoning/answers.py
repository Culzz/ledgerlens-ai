"""Turning a model reply into something the rest of the pipeline can use.

A reply is free text. Scoring, the verifier and the UI all need a value, and
sometimes the evidence behind it. One module does that conversion, and it is
written to fail loudly rather than guess.

Three formats are recognised, in order:

1. a JSON object with an ``answer`` field
2. a line marked ``ANSWER: <value>``
3. otherwise the last number that appears anywhere in the reply

Which one was used is recorded on the result rather than thrown away, because
how often the fallbacks fire is itself a number worth reporting. A parser that
quietly rescues a number out of a paragraph is how a wrong answer gets filed as
a right one.

Numbers are compared under two conventions, both reported. See ``scoring``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# The marker is matched greedily from the last occurrence, because models like to
# restate the answer at the end.
MARKER_RE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)
# Covers 100, 1,234.5, .455 (which does occur in FinQA tables) and 1e3.
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?")
BOOLEANS = {"yes", "no", "true", "false"}

FORMAT_JSON = "json"
FORMAT_MARKER = "marked"
FORMAT_FALLBACK = "last_number"
FORMAT_UNPARSED = "unparsed"
# The reply parsed but the program it carried would not run. Kept apart from
# ``unparsed`` because they are different failures: one is a model that did not
# answer in a readable shape, the other is a model that answered clearly and was
# wrong about the document. Counting them together hides the second.
FORMAT_UNEXECUTED = "unexecuted"


@dataclass(frozen=True)
class ParsedAnswer:
    """A model reply, reduced to a value plus whatever context came with it."""

    answer: str | None
    format: str
    raw: str
    reasoning: str = ""
    evidence: tuple[str, ...] = ()

    @property
    def parsed(self) -> bool:
        return self.answer is not None

    @property
    def value(self) -> Decimal | None:
        return to_decimal(self.answer) if self.answer is not None else None


def to_decimal(value: object) -> Decimal | None:
    """Read a number out of a short string, or return ``None``.

    Handles the shapes financial text actually uses: currency symbols, thousands
    separators, a trailing percent sign, a leading plus, and the accounting
    convention where a figure in brackets is negative.

    A percent sign is stripped, not applied. Whether ``93.5%`` should equal
    ``0.935`` is a scoring question, not a parsing one, and it is answered
    explicitly in ``scoring`` so both readings can be reported.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Various things that look like a minus sign.
    for dash in ("\u2212", "\u2013", "\u2014"):
        text = text.replace(dash, "-")

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    for symbol in ("$", "\u20ac", "\u00a3", "\u00a5"):
        text = text.replace(symbol, "")

    text = text.strip()
    text = text.removesuffix("%")

    text = text.replace(",", "").replace(" ", "").strip()
    if not NUMBER_RE.fullmatch(text):
        return None

    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return -number if negative else number


def normalise_text(value: object) -> str:
    """Lowercase and collapse whitespace, for comparing short answers."""
    return " ".join(str(value).strip().lower().split())


def last_number_in(text: str) -> str | None:
    """The last numeric token in a string, or ``None``."""
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_json_object(text: str) -> dict | None:
    """Find the first JSON object in a reply that actually parses.

    Tried in order: the whole reply, any fenced code block, then the span from
    the first brace to the last. Models wrap JSON in prose and in fences, and
    occasionally in both.
    """
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _json_candidates(text: str):
    stripped = text.strip()
    if stripped:
        yield stripped
    for match in FENCE_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            yield block
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        yield text[first : last + 1]


def _stringify(value: object) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _evidence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = _stringify(item)
            if text:
                out.append(text)
        return tuple(out)
    return ()


def _value_from(segment: str) -> str | None:
    """Pick the value out of the text following an ``ANSWER:`` marker."""
    cleaned = segment.strip().strip("*_`").strip().rstrip(".").strip()
    if not cleaned:
        return None
    if to_decimal(cleaned) is not None or normalise_text(cleaned) in BOOLEANS:
        return cleaned
    # "127.4 million" and similar: keep the number, drop the prose.
    return last_number_in(cleaned) or cleaned


def parse_answer(text: str | None) -> ParsedAnswer:
    """Reduce a model reply to an answer value, recording how it was found."""
    raw = text or ""

    obj = extract_json_object(raw)
    if obj is not None and "answer" in obj:
        return ParsedAnswer(
            answer=_stringify(obj.get("answer")),
            format=FORMAT_JSON,
            raw=raw,
            reasoning=_stringify(obj.get("reasoning")) or "",
            evidence=_evidence(obj.get("evidence")),
        )

    markers = list(MARKER_RE.finditer(raw))
    if markers:
        answer = _value_from(markers[-1].group(1))
        if answer is not None:
            return ParsedAnswer(answer=answer, format=FORMAT_MARKER, raw=raw, reasoning=raw)

    number = last_number_in(raw)
    if number is not None:
        return ParsedAnswer(answer=number, format=FORMAT_FALLBACK, raw=raw, reasoning=raw)

    return ParsedAnswer(answer=None, format=FORMAT_UNPARSED, raw=raw, reasoning=raw)
