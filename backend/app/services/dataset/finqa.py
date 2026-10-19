"""FinQA dataset loading, validation and quality checks.

FinQA records are nested, which trips people up. The answer fields live under
``qa``, not at the top level:

    {
      "pre_text":  ["..."],
      "post_text": ["..."],
      "table":     [["..."]],
      "id":        "ETR/2016/page_23.pdf-2",
      "qa": {
        "question":   "...",
        "program":    "subtract(5829, 5735)",
        "gold_inds":  {...},
        "exe_ans":    94,
        "program_re": "..."
      }
    }

This module turns those into ``FinQAExample`` objects and can report on the
quality of a split. Nothing here parses PDFs; FinQA ships its tables already
extracted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Fields we cannot do anything without.
REQUIRED_TOP_LEVEL = ("id", "table", "pre_text", "post_text")
REQUIRED_QA = ("question", "program", "exe_ans")


class FinQASchemaError(ValueError):
    """A record could not be read as a FinQA example."""

    def __init__(
        self,
        message: str,
        *,
        index: int | None = None,
        example_id: str | None = None,
    ) -> None:
        parts = []
        if index is not None:
            parts.append(f"index {index}")
        if example_id is not None:
            parts.append(f"id {example_id!r}")
        suffix = f" [{', '.join(parts)}]" if parts else ""
        super().__init__(f"{message}{suffix}")
        self.index = index
        self.example_id = example_id


@dataclass(frozen=True)
class FinQAExample:
    """One FinQA question, with its table and the evidence behind the answer."""

    id: str
    question: str
    table: tuple[tuple[str, ...], ...]
    pre_text: tuple[str, ...]
    post_text: tuple[str, ...]
    program: str
    exe_ans: Any
    program_re: str | None = None
    gold_inds: Mapping[str, Any] = field(default_factory=dict)
    filename: str | None = None
    # Which table rows and which text lines the annotators marked as the evidence
    # needed to answer. These are what FinQA's own retriever was trained against,
    # so they are the ground truth for measuring retrieval. Note that ``gold_inds``
    # is a separate annotation and disagrees with these on a minority of records.
    ann_table_rows: tuple[int, ...] = ()
    ann_text_rows: tuple[int, ...] = ()

    @property
    def context_text(self) -> tuple[str, ...]:
        """Narrative text around the table, in reading order."""
        return self.pre_text + self.post_text

    @property
    def report_filename(self) -> str:
        """The report this question is about.

        FinQA's records carry the report in the id - "V/2008/page_17.pdf-2" is the
        second question about that page - and leave the ``filename`` field empty.
        The id is therefore the reliable place to read it from.
        """
        return self.filename or self.id.rsplit("-", 1)[0]

    @property
    def table_shape(self) -> tuple[int, int]:
        rows = len(self.table)
        cols = len(self.table[0]) if rows else 0
        return rows, cols


def _text_list(
    value: Any, *, field_name: str, index: int, example_id: str | None
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise FinQASchemaError(
                    f"{field_name} must be a list of strings, got {type(item).__name__}",
                    index=index,
                    example_id=example_id,
                )
            out.append(item)
        return tuple(out)
    raise FinQASchemaError(
        f"{field_name} must be a string or list of strings, got {type(value).__name__}",
        index=index,
        example_id=example_id,
    )


def _int_list(
    value: Any, *, field_name: str, index: int, example_id: str | None
) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FinQASchemaError(
            f"{field_name} must be a list of integers, got {type(value).__name__}",
            index=index,
            example_id=example_id,
        )
    out: list[int] = []
    for item in value:
        # bool is a subclass of int, and True as a row index would be a silent bug.
        if isinstance(item, bool) or not isinstance(item, int):
            raise FinQASchemaError(
                f"{field_name} must contain integers, got {type(item).__name__}",
                index=index,
                example_id=example_id,
            )
        out.append(item)
    return tuple(out)


def _table(
    value: Any, *, index: int, example_id: str | None
) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FinQASchemaError(
            f"table must be a list of rows, got {type(value).__name__}",
            index=index,
            example_id=example_id,
        )
    rows: list[tuple[str, ...]] = []
    for row_no, row in enumerate(value):
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise FinQASchemaError(
                f"table row {row_no} must be a list, got {type(row).__name__}",
                index=index,
                example_id=example_id,
            )
        rows.append(tuple("" if cell is None else str(cell) for cell in row))
    return tuple(rows)


def parse_example(record: Mapping[str, Any], *, index: int | None = None) -> FinQAExample:
    """Turn one raw record into a ``FinQAExample``.

    Raises ``FinQASchemaError`` with the offending record identified.
    """
    if not isinstance(record, Mapping):
        raise FinQASchemaError(
            f"record must be an object, got {type(record).__name__}", index=index
        )

    example_id = record.get("id")
    if not isinstance(example_id, str) or not example_id:
        raise FinQASchemaError("missing or empty id", index=index)

    for name in REQUIRED_TOP_LEVEL:
        if name not in record:
            raise FinQASchemaError(
                f"missing required field {name!r}", index=index, example_id=example_id
            )

    qa = record.get("qa")
    if not isinstance(qa, Mapping):
        raise FinQASchemaError(
            f"'qa' must be an object, got {type(qa).__name__}",
            index=index,
            example_id=example_id,
        )
    for name in REQUIRED_QA:
        if name not in qa:
            raise FinQASchemaError(
                f"missing required field 'qa.{name}'", index=index, example_id=example_id
            )

    question = qa["question"]
    if not isinstance(question, str) or not question.strip():
        raise FinQASchemaError(
            "qa.question must be a non-empty string", index=index, example_id=example_id
        )

    program = qa["program"]
    if not isinstance(program, str):
        raise FinQASchemaError(
            f"qa.program must be a string, got {type(program).__name__}",
            index=index,
            example_id=example_id,
        )

    gold_inds = qa.get("gold_inds") or {}
    if not isinstance(gold_inds, Mapping):
        gold_inds = {}

    return FinQAExample(
        id=example_id,
        question=question,
        table=_table(record["table"], index=index, example_id=example_id),
        pre_text=_text_list(
            record["pre_text"], field_name="pre_text", index=index, example_id=example_id
        ),
        post_text=_text_list(
            record["post_text"], field_name="post_text", index=index, example_id=example_id
        ),
        program=program,
        exe_ans=qa["exe_ans"],
        program_re=qa.get("program_re"),
        gold_inds=gold_inds,
        filename=record.get("filename"),
        ann_table_rows=_int_list(
            qa.get("ann_table_rows"),
            field_name="qa.ann_table_rows",
            index=index,
            example_id=example_id,
        ),
        ann_text_rows=_int_list(
            qa.get("ann_text_rows"),
            field_name="qa.ann_text_rows",
            index=index,
            example_id=example_id,
        ),
    )


def load_examples(path: str | Path) -> list[FinQAExample]:
    """Load a FinQA split file (a JSON list of records)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such FinQA split: {p}")

    with p.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise FinQASchemaError(
            f"{p.name} should contain a list of records, got {type(raw).__name__}"
        )

    return [parse_example(rec, index=i) for i, rec in enumerate(raw)]


# Alias, since "split" is the word the evaluation protocol uses.
load_split = load_examples


@dataclass
class DataQualityReport:
    """What a split actually looks like, so problems surface early."""

    source: str
    total: int
    unique_ids: int = 0
    duplicate_ids: tuple[str, ...] = ()
    empty_questions: int = 0
    empty_tables: int = 0
    ragged_tables: tuple[str, ...] = ()
    with_program_re: int = 0
    with_gold_inds: int = 0
    answer_kinds: Mapping[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"source:            {self.source}",
            f"examples:          {self.total}",
            f"unique ids:        {self.unique_ids}",
            f"duplicate ids:     {len(self.duplicate_ids)}",
            f"empty questions:   {self.empty_questions}",
            f"empty tables:      {self.empty_tables}",
            f"ragged tables:     {len(self.ragged_tables)}",
            f"with program_re:   {self.with_program_re}",
            f"with gold_inds:    {self.with_gold_inds}",
            "answer kinds:      "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.answer_kinds.items())),
        ]
        if self.duplicate_ids:
            shown = ", ".join(self.duplicate_ids[:5])
            more = "" if len(self.duplicate_ids) <= 5 else f" (+{len(self.duplicate_ids) - 5} more)"
            lines.append(f"  duplicates: {shown}{more}")
        if self.ragged_tables:
            shown = ", ".join(self.ragged_tables[:5])
            more = "" if len(self.ragged_tables) <= 5 else f" (+{len(self.ragged_tables) - 5} more)"
            lines.append(f"  ragged: {shown}{more}")
        return "\n".join(lines)


def scan_quality(examples: Sequence[FinQAExample], *, source: str = "<memory>") -> DataQualityReport:
    """Summarise a split. Does not raise; problems are counted and listed."""
    seen: dict[str, int] = {}
    dupes: list[str] = []
    empty_questions = 0
    empty_tables = 0
    ragged: list[str] = []
    answer_kinds: dict[str, int] = {}
    with_program_re = 0
    with_gold_inds = 0

    for ex in examples:
        seen[ex.id] = seen.get(ex.id, 0) + 1
        if not ex.question.strip():
            empty_questions += 1
        rows, cols = ex.table_shape
        if rows == 0:
            empty_tables += 1
        elif any(len(r) != cols for r in ex.table):
            ragged.append(ex.id)
        if ex.program_re:
            with_program_re += 1
        if ex.gold_inds:
            with_gold_inds += 1
        if isinstance(ex.exe_ans, bool):
            kind = "bool"
        elif isinstance(ex.exe_ans, (int, float)):
            kind = "number"
        elif isinstance(ex.exe_ans, str):
            kind = "string"
        else:
            kind = type(ex.exe_ans).__name__
        answer_kinds[kind] = answer_kinds.get(kind, 0) + 1

    dupes = [eid for eid, n in seen.items() if n > 1]

    return DataQualityReport(
        source=source,
        total=len(examples),
        unique_ids=len(seen),
        duplicate_ids=tuple(dupes),
        empty_questions=empty_questions,
        empty_tables=empty_tables,
        ragged_tables=tuple(ragged),
        with_program_re=with_program_re,
        with_gold_inds=with_gold_inds,
        answer_kinds=answer_kinds,
    )
