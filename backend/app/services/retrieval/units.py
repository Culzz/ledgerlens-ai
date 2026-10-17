"""Turning a FinQA record into the units a retriever works over.

A question is answered from a single document: a table plus the narrative around
it. Retrieval happens inside that document, which mirrors the product case - one
uploaded report, find the rows that matter.

Two kinds of unit:

* one per table row
* one per narrative line

Table rows are expanded with their column headers before indexing. A raw row
reads as "net sales 100 90", and the year is exactly what questions mention, so
dropping the header would make those rows unfindable.

Text lines keep their published wording, warts and all. They come out of PDF
extraction and carry artefacts like "2019s" for "2019's". Cleaning that up is a
separate problem, and doing it here would change what the measurement means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.services.dataset import FinQAExample

TABLE = "table"
TEXT = "text"


@runtime_checkable
class DocumentLike(Protocol):
    """The part of a document retrieval needs, and nothing else.

    A FinQA record satisfies this, and so does a report uploaded through the
    product path. Both then run the same retriever, which is the only way the
    retrieval measured on the benchmark says anything about the product.
    """

    id: str
    table: Sequence[Sequence[str]]
    context_text: Sequence[str]


@dataclass(frozen=True)
class EvidenceUnit:
    """One retrievable piece of evidence."""

    id: str
    example_id: str
    kind: str
    index: int
    text: str

    @property
    def is_table(self) -> bool:
        return self.kind == TABLE

    @property
    def is_text(self) -> bool:
        return self.kind == TEXT


def unit_id(example_id: str, kind: str, index: int) -> str:
    """Stable id for a unit.

    Built from the same indices the annotations use, so a gold reference and a
    retrieved unit are directly comparable strings.
    """
    return f"{example_id}::{kind}::{index}"


def row_text(table, row_index: int) -> str:
    """Read a table row as a line of text, with its column headers attached."""
    row = table[row_index]
    header = table[0] if table else ()
    parts: list[str] = []

    for column, raw in enumerate(row):
        cell = str(raw).strip()
        if column == 0:
            if cell:
                parts.append(cell)
            continue
        label = str(header[column]).strip() if column < len(header) else ""
        # On the header row the label and the cell are the same string. Repeating
        # it would double every token and add nothing.
        if label and label != cell:
            joined = f"{label} {cell}".strip()
        else:
            joined = cell or label
        if joined:
            parts.append(joined)

    return " ".join(parts)


def build_units(example: DocumentLike) -> tuple[EvidenceUnit, ...]:
    """Every unit in one document, in a stable order.

    The count is always ``len(table) + len(context_text)``. Empty narrative
    lines are kept rather than dropped, so the mapping from annotation index to
    unit stays total and a gold reference can never point at something that was
    silently removed.
    """
    units: list[EvidenceUnit] = []

    for index in range(len(example.table)):
        units.append(
            EvidenceUnit(
                id=unit_id(example.id, TABLE, index),
                example_id=example.id,
                kind=TABLE,
                index=index,
                text=row_text(example.table, index),
            )
        )

    for index, line in enumerate(example.context_text):
        units.append(
            EvidenceUnit(
                id=unit_id(example.id, TEXT, index),
                example_id=example.id,
                kind=TEXT,
                index=index,
                text=line,
            )
        )

    return tuple(units)


def gold_unit_ids(example: FinQAExample) -> set[str]:
    """The units the annotators said the answer needs."""
    ids = {unit_id(example.id, TABLE, index) for index in example.ann_table_rows}
    ids |= {unit_id(example.id, TEXT, index) for index in example.ann_text_rows}
    return ids


def evidence_kinds(example: FinQAExample) -> tuple[str, ...]:
    """Which kinds of unit this question needs, in a fixed order."""
    kinds: list[str] = []
    if example.ann_table_rows:
        kinds.append(TABLE)
    if example.ann_text_rows:
        kinds.append(TEXT)
    return tuple(kinds)
