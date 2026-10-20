"""The brief store, against a real Postgres.

Skipped unless DATABASE_URL points at one. The unit tests replace the repository, so
this is the only place the SQL is exercised: the insert, the read, and the row coming
back as a list of dictionaries rather than as a string that still needs decoding.
"""

from __future__ import annotations

import os

import pytest
from app.db.repositories import briefs

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL is not set; the store needs a real database",
)


ROWS = [
    {
        "question": "What was the change in operating income?",
        "program": "subtract(150, 120)",
        "answer": "30",
        "verdict": "grounded",
        "unsourced": [],
        "sourced_count": 2,
        "literal_count": 2,
        "labels": ["table_1"],
        "note": None,
    }
]


def test_a_brief_survives_a_round_trip():
    brief_id = briefs.save("sample-2023-page-10-pdf", "SAMPLE/2023/page_10.pdf", ROWS)

    record = briefs.load(brief_id)

    assert record is not None
    assert record["id"] == brief_id
    assert record["document_id"] == "sample-2023-page-10-pdf"
    assert record["filename"] == "SAMPLE/2023/page_10.pdf"
    # jsonb, decoded: the rows have to come back as rows, not as text holding JSON.
    assert record["rows"] == ROWS
    assert record["created_at"] is not None


def test_an_unknown_id_loads_nothing():
    assert briefs.load("nosuchbrief1") is None
