"""The documents the demo can be asked about.

FinQA is the document set, read from the split on disk rather than committed, so
the repository does not redistribute it. A record in that file is one question
about one report, and several records share a report, so the library groups them
back by filename into something a person would recognise as a document.

All that is handed out is the document itself: the table, the narrative, and the
questions that go with it. Reference answers sit behind a call of their own, taken
only when a response needs to show what the annotation says. The answering path
cannot reach them, which is the property the evaluation protocol needs and the
easiest one to lose by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.services.dataset import FinQAExample, load_split

_SLUG = re.compile(r"[^a-z0-9]+")


def slug_for(filename: str) -> str:
    """A URL-safe name for a report, since the filenames contain slashes."""
    return _SLUG.sub("-", filename.lower()).strip("-") or "document"


@dataclass(frozen=True)
class DocumentSummary:
    """One report, with the questions asked about it."""

    id: str
    filename: str
    questions: tuple[str, ...]
    table_rows: int
    context_lines: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "filename": self.filename,
            "questions": list(self.questions),
            "table_rows": self.table_rows,
            "context_lines": self.context_lines,
        }


class DocumentLibrary:
    """Read-only access to the documents, grouped by report."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._by_id: dict[str, list[FinQAExample]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            # The split has not been downloaded. Not a fault: the app runs, it just
            # has nothing to show yet, and the API says what to run to fix that.
            return

        # One key per report, reused by every question about it. The counter is
        # only for the rare case of two different filenames that slug the same.
        keys: dict[str, str] = {}
        used: set[str] = set()

        for example in load_split(self.path):
            # "V/2008/page_17.pdf-2" is the second question about that report.
            filename = example.id.rsplit("-", 1)[0]
            key = keys.get(filename)
            if key is None:
                base = slug_for(filename)
                key, counter = base, 2
                while key in used:
                    key = f"{base}-{counter}"
                    counter += 1
                used.add(key)
                keys[filename] = key
            self._by_id.setdefault(key, []).append(example)

    @property
    def available(self) -> bool:
        return bool(self._by_id)

    def summaries(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[DocumentSummary]:
        """Reports, optionally filtered by a substring of the filename.

        The whole set is a few hundred reports, which is more than anyone wants to
        scroll, so the filter is how a person finds one rather than a hard cap on
        what is reachable.
        """
        needle = (query or "").strip().lower()
        out: list[DocumentSummary] = []

        for key, examples in self._by_id.items():
            first = examples[0]
            filename = first.id.rsplit("-", 1)[0]
            if needle and needle not in filename.lower():
                continue
            out.append(
                DocumentSummary(
                    id=key,
                    filename=filename,
                    questions=tuple(example.question for example in examples),
                    table_rows=len(first.table),
                    context_lines=len(first.context_text),
                )
            )

        out.sort(key=lambda summary: summary.filename)
        return out[:limit] if limit else out

    def get(self, document_id: str) -> list[FinQAExample] | None:
        return self._by_id.get(document_id)

    def example_for(self, document_id: str, question: str) -> FinQAExample | None:
        """The record behind one question, or ``None`` if it was not in the set.

        A question typed by hand rather than picked from the list has no record
        behind it, and that is fine - it is answered from the document like any
        other, with no reference to compare against afterwards.
        """
        for example in self._by_id.get(document_id, []):
            if example.question == question:
                return example
        return None
