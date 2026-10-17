"""Tests for the run reviewer.

The classification is the part worth testing: it decides whether a wrong answer
is a scale mismatch, a rounding difference or genuinely wrong, and those three
mean very different things about the system.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.dataset import load_split

from evaluation.review_run import Case, _bucket, collect, decompose

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "finqa_sample.json"


def case(reference: str, predicted: str | None, *, outcome: str = "wrong_number", **kwargs) -> Case:
    return Case(
        question_id="q",
        question="?",
        reference=reference,
        predicted=predicted,
        outcome=outcome,
        correct=kwargs.get("correct", False),
        percent_equivalent=kwargs.get("percent_equivalent", False),
    )


class TestBuckets:
    def test_a_correct_answer(self):
        assert _bucket(case("5", "5", correct=True)) == "correct"

    def test_a_percentage_answered_as_a_percentage(self):
        # FinQA stores 93.5% as 0.935; the model answers 93.5.
        assert _bucket(case("0.935", "93.5", percent_equivalent=True)) == "scale_mismatch"

    def test_a_rounded_answer_is_a_near_miss(self):
        assert _bucket(case("24.69136", "24.7")) == "near_miss"

    def test_a_bigger_gap_is_marginal(self):
        assert _bucket(case("100", "103")) == "marginal"

    def test_plainly_wrong(self):
        assert _bucket(case("100", "250")) == "wrong"

    def test_a_non_numeric_failure_keeps_its_own_outcome(self):
        assert _bucket(case("5", "lots", outcome="wrong_text")) == "wrong_text"

    def test_a_zero_reference_has_no_ratio(self):
        assert _bucket(case("0", "5")) == "wrong_not_comparable"

    def test_an_unparseable_number_has_no_ratio(self):
        assert _bucket(case("5", None)) == "wrong_not_comparable"


class TestDecompose:
    def test_counts_every_bucket(self):
        cases = [
            case("5", "5", correct=True),
            case("0.935", "93.5", percent_equivalent=True),
            case("24.69136", "24.7"),
            case("100", "103"),
            case("100", "250"),
        ]
        assert decompose(cases) == {
            "correct": 1,
            "scale_mismatch": 1,
            "near_miss": 1,
            "marginal": 1,
            "wrong": 1,
        }

    def test_an_empty_run_decomposes_to_nothing(self):
        assert decompose([]) == {}


class TestCollect:
    def test_reads_a_run_against_its_split(self, tmp_path):
        examples = load_split(FIXTURE)
        first = examples[0]

        run = tmp_path / "run"
        run.mkdir()
        records = [
            {
                "question_id": first.id,
                "parsed": {"answer": str(first.exe_ans), "format": "marked", "raw": ""},
            },
            {"question_id": "not-in-the-split", "parsed": {"answer": "1", "format": "marked"}},
        ]
        (run / "predictions.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )

        cases = collect(run, FIXTURE)
        # Only questions in the split are scored, and a correct answer stays correct.
        assert len(cases) == 1
        assert cases[0].question_id == first.id
        assert cases[0].correct
