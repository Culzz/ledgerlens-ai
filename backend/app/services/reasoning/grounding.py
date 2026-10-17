"""Where the numbers in a program came from.

The engine can run a program, but a program on its own does not say whether its
figures were in the report. ``divide(24800, 22900)`` is arithmetic; it is not
evidence. Two of them were read off the AAL table, and the engine has no way to
know that.

This closes the loop by looking each figure back up in the report. A figure that
appears in a row or a line is *sourced* and gets a citation. A figure that appears
nowhere is either a structural constant - 100 to turn a ratio into a percent - or it
is a number the model made up, and those are worth knowing apart.

It is deliberately the same mechanism for two jobs. The citation on an operand in
the audit trail and the decision to refuse an answer both come from "was this
number in the report", and answering that question twice in two places would let
them disagree.

Three ways a right figure reads as missing
-----------------------------------------
Reading a report for figures is harder than it looks, and every one of these was
found by replaying a measured run and looking at what the check objected to:

* a cell carrying its own unit - ``$ 739 million`` states the figure 739
* a table declaring a unit - ``( in millions )`` under a column of 240.0
* a count of periods - ``divide(48259, 3)`` averages three years

The first two are the report saying a number a different way and are handled by
reading the cells and the declaration. The third is arithmetic rather than data,
and is carved out by name. A check that misses any of them throws away correct
answers, which is a worse failure than the one it is built to prevent.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from app.services.calculations import (
    TABLE_OPERATIONS,
    CalculationTrace,
    Call,
    InvalidArgument,
    Literal,
    parse_number,
    parse_program,
)
from app.services.retrieval import TABLE, DocumentLike, build_units

# The roles a single argument can play in a step.
REFERENCE = "reference"  # a #n back-reference to an earlier step
LABEL = "label"  # a row or column name in a table operation
SOURCED = "sourced"  # a figure the report states
CONSTANT = "constant"  # a figure that is part of the arithmetic
UNSOURCED = "unsourced"  # a figure the report does not state
NESTED = "nested"  # a call inside another call

# The verdict for a program as a whole.
GROUNDED = "grounded"
PARTLY_GROUNDED = "partly_grounded"
UNGROUNDED = "ungrounded"

# Unit words and what they multiply a figure by, longest unit first so that a
# prefix test cannot match "million" in "millions" or "illion" in "trillion".
UNIT_FACTORS: tuple[tuple[str, Decimal], ...] = (
    ("trillion", Decimal(10) ** 12),
    ("billion", Decimal(10) ** 9),
    ("million", Decimal(10) ** 6),
    ("thousand", Decimal(10) ** 3),
)

UNIT_NAMES: dict[Decimal, str] = {
    Decimal(10) ** 12: "trillions",
    Decimal(10) ** 9: "billions",
    Decimal(10) ** 6: "millions",
    Decimal(10) ** 3: "thousands",
}

# How a report writes a unit, which is plural, against the words a cell uses, which
# can be either ("$ 739 million").
_FACTOR_BY_WORD: dict[str, Decimal] = {word: factor for word, factor in UNIT_FACTORS} | {
    f"{word}s": factor for word, factor in UNIT_FACTORS
}

# Figures a program may contain that are not in the report, because they are part
# of the arithmetic rather than the data. Two kinds, both narrow on purpose: every
# entry is a licence not to look for a figure, and a long list would be the same as
# having no check at all.
#
#   scales  100 to turn a ratio into a percent, a thousand to change a unit
#   counts  how many periods are being averaged: three years, four quarters
#
# Counts are the integers. A report rarely states 3 as a figure and very often has
# three of something, so refusing 3 costs correct answers and buys almost nothing.
# This is measured rather than assumed - see ``evaluation/audit_grounding.py``.
STRUCTURAL_CONSTANTS = frozenset(
    [Decimal(text) for text in ("-1", "0.01", "0.001", "100", "-100", "1000", "10000", "100000")]
    + [Decimal(value) for value in range(1, 13)]
    + [Decimal(10) ** exponent for exponent in (6, 9, 12)]
)

# A figure: 1,234 / .455 / 12.5% / -13. The `$` is left out because the parser
# strips it, and so are the thousands separators, which the parser puts back.
_FIGURE = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?")

# A table saying what its figures are in. "( in millions )", "( dollars in
# thousands , except per share data )".
_DECLARATION = re.compile(r"\bin\s+(thousands|millions|billions|trillions)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Stated:
    """One figure a piece of report text states, and how it was written."""

    value: Decimal
    # The unit word the figure carried, if it carried one. "$ 4.6 billion" is
    # 4.6 with a billion attached; a bare table cell is 240.0 with nothing.
    unit: Decimal | None = None
    # Written with a percent sign, which makes it a ratio rather than a quantity.
    percent: bool = False


def readings(text: str) -> list[Stated]:
    """Every figure in a line of prose, and how it was written.

    Prose holds several figures with words between them, so this scans rather than
    reading the line whole.

    A unit word is only read when it sits just after the figure, which is where a
    report puts it. Looking further would let a line mentioning "3 million" and
    later "4 thousand" scale the wrong one.
    """
    found: list[Stated] = []
    text = text or ""

    for match in _FIGURE.finditer(text):
        raw = match.group(0)
        try:
            value = parse_number(raw)
        except InvalidArgument:
            continue

        if raw.endswith("%"):
            found.append(Stated(value=value, percent=True))
            continue

        window = text[match.end() : match.end() + 16].lower().lstrip()
        unit = next((factor for word, factor in UNIT_FACTORS if window.startswith(word)), None)
        found.append(Stated(value=value, unit=unit))

    return found


def cell_readings(text: str) -> list[Stated]:
    """Every figure in one table cell, and how it was written.

    The engine's own reading comes first, and it is the only one when it works: one
    cell, one figure. Scanning a cell for figures with a pattern would read
    ``-13 ( 13 )`` as both -13 and 13, and a program that lost a sign would then
    look like it had a source.

    A cell the parser cannot read whole gets a second look for a figure carrying
    its own unit, which is how a balance sheet writes ``$ 739 million``.

    A cell with no digit in it is skipped rather than parsed. The parser reads a
    lone dash as zero, which is right for the arithmetic and wrong here: it would
    make every row in the report a source for the figure zero.
    """
    text = str(text)
    if not any(character.isdigit() for character in text):
        return []

    try:
        value = parse_number(text)
    except InvalidArgument:
        pass
    else:
        if "%" in text:
            return [Stated(value=value, percent=True)]
        return [Stated(value=value)]

    match = _FIGURE.search(text)
    if match is None:
        return []
    try:
        bare = parse_number(match.group(0))
    except InvalidArgument:
        return []

    if match.group(0).endswith("%"):
        return [Stated(value=bare, percent=True)]

    lowered = text.lower()
    unit = next((factor for word, factor in UNIT_FACTORS if word in lowered), None)
    return [Stated(value=bare, unit=unit)]


def readings_in_unit(stated: Sequence[Stated], unit: Decimal | None) -> set[Decimal]:
    """Every way a figure reads, in the units the report makes available.

    Two kinds of figure, and they need different treatment.

    A figure that named its own unit - "6.8 billion" - is an amount, and an amount
    can be written in any unit. The report says 6.8 billion and the program writes
    6800, which is the same amount in millions, so both readings count. Which one
    the report "uses" is a presentation choice, and refusing an answer over it is
    refusing a right answer.

    A bare figure - 240.0 in a table cell - is not an amount yet. It is 240.0 of
    whatever the table said it was in, so the declaration is what turns it into one:
    in a table of millions it is also 240,000,000.

    A percent is neither. It is a ratio rather than an amount, so 12.5% reads as
    0.125 and as 12.5 and is not converted into anything.

    Every one of these existed because a measured run said so. See
    ``evaluation/audit_grounding.py``, which replays a finished run through the
    check and counts the answers it refuses that were right.
    """
    found: set[Decimal] = set()

    for item in stated:
        found.add(item.value)

        if item.percent:
            found.add(item.value * 100)
            continue

        if item.unit is not None:
            amount = item.value * item.unit
            found.add(amount)
            for _, factor in UNIT_FACTORS:
                found.add(amount / factor)
            continue

        if unit is not None:
            found.add(item.value * unit)

    return found


def numbers_in(text: str, unit: Decimal | None = None) -> set[Decimal]:
    """Every figure that can be read out of a line of prose."""
    return readings_in_unit(readings(text), unit)


def cell_numbers(text: str, unit: Decimal | None = None) -> set[Decimal]:
    """Every figure that can be read out of one table cell."""
    return readings_in_unit(cell_readings(text), unit)


def row_numbers(row: Sequence[str], unit: Decimal | None = None) -> set[Decimal]:
    """Every figure in a table row, read cell by cell."""
    found: set[Decimal] = set()
    for cell in row:
        found |= cell_numbers(cell, unit)
    return found


@dataclass(frozen=True)
class Report:
    """The figures a report states, in every unit it makes available."""

    figures: dict[str, set[Decimal]]
    # The factor the report's tables say their figures are in, or None when the
    # report does not say. Carried for the audit trail, so a reader can be told the
    # figures were compared in the report's own unit.
    unit: Decimal | None = None

    @property
    def unit_name(self) -> str | None:
        return UNIT_NAMES.get(self.unit) if self.unit is not None else None


def declared_unit(document: DocumentLike) -> Decimal | None:
    """The unit the report says its figures are in, if it says.

    A table headed "( in millions )" states 240 where the number is 240,000,000 and
    means the second.

    One unit is kept for the whole report, the most commonly declared one. That is
    right for the filings here, where a single note covers a set of statements, and
    wrong for a report that declares different units in different tables. Recorded
    here so the limitation is on the record rather than found later.
    """
    lines: list[str] = list(document.context_text)
    if document.table:
        lines.extend(document.table[0])

    counts: Counter[Decimal] = Counter()
    for line in lines:
        for match in _DECLARATION.finditer(str(line)):
            counts[_FACTOR_BY_WORD[match.group(1).lower()]] += 1

    if not counts:
        return None
    return counts.most_common(1)[0][0]


def read_report(document: DocumentLike) -> Report:
    """Every figure in the report, in every unit the report makes available.

    The whole document, not just the retrieved part of it. The claim being tested
    is "this figure is in the report", and retrieval missing the row a figure lives
    in is not the same as the figure not existing - checking only what was
    retrieved would turn a retrieval miss into an accusation of invention.
    """
    unit = declared_unit(document)
    figures: dict[str, set[Decimal]] = {}

    for piece in build_units(document):
        label = f"{piece.kind}_{piece.index}"
        if piece.kind == TABLE and 0 <= piece.index < len(document.table):
            figures[label] = row_numbers(document.table[piece.index], unit)
            continue
        figures[label] = numbers_in(piece.text, unit)

    return Report(figures=figures, unit=unit)


@dataclass(frozen=True)
class Operand:
    """One argument of a step, and what its numbers are."""

    text: str
    role: str
    value: Decimal | None = None
    labels: tuple[str, ...] = ()

    @property
    def is_bare(self) -> bool:
        """A figure written out in the program rather than referred to."""
        return self.role in (SOURCED, CONSTANT, UNSOURCED)


@dataclass(frozen=True)
class AuditStep:
    """A step as executed, with each figure tied back to the report."""

    index: int
    operation: str
    operands: tuple[Operand, ...]
    result: Decimal | str

    def render(self) -> str:
        if not self.operation:
            return f"#{self.index} = {self.result}"
        inner = ", ".join(operand.text for operand in self.operands)
        return f"#{self.index} = {self.operation}({inner}) = {self.result}"


@dataclass(frozen=True)
class Grounding:
    """The answer to "did these numbers come from the report"."""

    verdict: str
    steps: tuple[AuditStep, ...]
    # Every label a figure was found in, whether or not the reply cited it. The
    # difference between this and the reply's own list is a model that used a row
    # without saying so.
    labels: tuple[str, ...] = ()
    # Figures with no home in the report, as written in the program.
    unsourced: tuple[str, ...] = ()
    sourced_count: int = 0
    literal_count: int = 0
    # Every figure was a constant. Not an error, but the answer is arithmetic rather
    # than a reading of the document, and that is worth being able to count.
    constant_only: bool = False
    # The unit the report said its figures were in, when it said. A figure can match
    # exactly and still be worth explaining: 240000000 against a table of 240.0 is
    # the same amount, and a reader of the trail should be told why it matched.
    report_unit: str | None = None

    @property
    def blocks(self) -> bool:
        return self.verdict == UNGROUNDED

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "labels": list(self.labels),
            "unsourced": list(self.unsourced),
            "sourced_count": self.sourced_count,
            "literal_count": self.literal_count,
            "constant_only": self.constant_only,
            "report_unit": self.report_unit,
            "steps": [
                {
                    "index": step.index,
                    "operation": step.operation,
                    "result": str(step.result),
                    "render": step.render(),
                    "operands": [
                        {
                            "text": operand.text,
                            "role": operand.role,
                            "labels": list(operand.labels),
                            "value": None if operand.value is None else str(operand.value),
                        }
                        for operand in step.operands
                    ],
                }
                for step in self.steps
            ],
        }


def _holding(value: Decimal, figures: Mapping[str, set[Decimal]]) -> tuple[str, ...]:
    """Every unit of the report that states this figure."""
    return tuple(sorted(label for label, stated in figures.items() if value in stated))


def _classify(literal: Literal, report: Report) -> Operand:
    """Say where one figure came from."""
    text = literal.text
    try:
        value = parse_number(text)
    except InvalidArgument:
        # It parsed as part of the program, so this should not happen. Treating it
        # as unsourced is the safe reading.
        return Operand(text=f"{text} (unverified)", role=UNSOURCED)

    labels = _holding(value, report.figures)
    if labels:
        return Operand(
            text=f"{text} [{', '.join(labels)}]", role=SOURCED, value=value, labels=labels
        )

    if value in STRUCTURAL_CONSTANTS:
        return Operand(text=text, role=CONSTANT, value=value)
    return Operand(text=f"{text} (unverified)", role=UNSOURCED, value=value)


def _table_operands(node: Call) -> tuple[Operand, ...]:
    """A table call's arguments, which are row and column names rather than figures.

    They have to be kept out of the figure walk entirely. "revenue" is a label, and
    looking it up as a number would either find nothing and mark the step
    unverified, or find the figure 1200 somewhere else and cite the wrong unit.
    """
    return tuple(Operand(text=getattr(arg, "text", str(arg)), role=LABEL) for arg in node.args)


def _annotate(node: object, report: Report) -> Operand:
    """An argument, rendered with its citations."""
    if isinstance(node, Literal):
        if node.is_reference:
            return Operand(text=node.text, role=REFERENCE)
        return _classify(node, report)

    if not isinstance(node, Call):
        return Operand(text=str(node), role=UNSOURCED)

    if node.name in TABLE_OPERATIONS:
        args = _table_operands(node)
        inner = ", ".join(operand.text for operand in args)
        return Operand(text=f"{node.name}({inner})", role=LABEL)

    inner = ", ".join(_annotate(arg, report).text for arg in node.args)
    return Operand(text=f"{node.name}({inner})", role=NESTED)


def _literals_in(node: object) -> list[Literal]:
    """Every bare figure in a program, however deeply it is nested.

    Nested calls matter: in ``divide(subtract(150, 120), 120)`` the engine records
    one step, so walking the *steps* would see only the outer 120 and let 150 and
    120 through unexamined.
    """
    if isinstance(node, Literal):
        return [] if node.is_reference else [node]
    if not isinstance(node, Call):
        return []
    if node.name in TABLE_OPERATIONS:
        return []
    found: list[Literal] = []
    for arg in node.args:
        found.extend(_literals_in(arg))
    return found


def audit(trace: CalculationTrace, report: Report) -> Grounding:
    """Tie every figure in an executed program back to the report."""
    parsed = parse_program(trace.program)

    steps: list[AuditStep] = []
    literals: list[Literal] = []
    for node, step in zip(parsed, trace.steps, strict=True):
        literals.extend(_literals_in(node))
        if isinstance(node, Literal):
            operands = (_annotate(node, report),)
        elif isinstance(node, Call) and node.name in TABLE_OPERATIONS:
            # A whole step that reads a row names its labels directly, so it is not
            # an argument to be classified but a call whose operands are names.
            operands = _table_operands(node)
        else:
            operands = tuple(_annotate(arg, report) for arg in node.args)
        steps.append(AuditStep(step.index, step.operation, operands, step.result))

    classified = [_classify(literal, report) for literal in literals]
    sourced = [operand for operand in classified if operand.role == SOURCED]
    unsourced = [operand for operand in classified if operand.role == UNSOURCED]

    if sourced and not unsourced:
        verdict = GROUNDED
    elif sourced:
        verdict = PARTLY_GROUNDED
    elif unsourced:
        verdict = UNGROUNDED
    else:
        # No bare figures at all: every value came out of a table operation.
        verdict = GROUNDED

    labels: list[str] = []
    for operand in sourced:
        for label in operand.labels:
            if label not in labels:
                labels.append(label)

    return Grounding(
        verdict=verdict,
        steps=tuple(steps),
        labels=tuple(labels),
        unsourced=tuple(operand.text.removesuffix(" (unverified)") for operand in unsourced),
        sourced_count=len(sourced),
        literal_count=len(classified),
        constant_only=bool(classified) and not sourced and not unsourced,
        report_unit=report.unit_name,
    )
