"""Tests for the side-by-side run comparison."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation import compare_runs

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "finqa_sample.json"

Q1 = "SAMPLE/2023/page_10.pdf-1"  # 30
Q2 = "SAMPLE/2023/page_10.pdf-2"  # 0.25
Q3 = "SAMPLE/2023/page_10.pdf-3"  # 25
Q4 = "SAMPLE/2023/page_11.pdf-1"  # 270


def write_run(tmp_path: Path, name: str, answers: dict[str, str]) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps({"system": name, "prompt_version": "test-v1"}), encoding="utf-8"
    )
    with (run / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for question_id, answer in answers.items():
            handle.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "parsed": {"answer": answer, "format": "json", "raw": ""},
                    }
                )
                + "\n"
            )
    return run


def run_compare(tmp_path: Path, first: dict[str, str], second: dict[str, str], capsys) -> str:
    left = write_run(tmp_path, "first", first)
    right = write_run(tmp_path, "second", second)
    code = compare_runs.main([str(left), str(right), "--split", str(FIXTURE)])
    assert code == 0
    return capsys.readouterr().out


class TestCompareRuns:
    def test_only_the_shared_questions_are_scored(self, tmp_path, capsys):
        # The first run answered three of these, the second two, and only two are
        # in both. Scoring anything else would compare different question sets.
        text = run_compare(
            tmp_path,
            {Q1: "30", Q2: "0.25", Q3: "25"},
            {Q2: "0.25", Q3: "25", Q4: "270"},
            capsys,
        )

        assert "shared: 2 questions" in text

    def test_the_pairs_are_split_four_ways(self, tmp_path, capsys):
        text = run_compare(
            tmp_path,
            {Q1: "30", Q2: "0.25", Q3: "999", Q4: "270"},
            {Q1: "30", Q2: "999", Q3: "25", Q4: "270"},
            capsys,
        )

        assert "both got it          2" in text
        assert "only the first       1" in text
        assert "only the second      1" in text
        assert "neither              0" in text

    def test_the_accuracies_are_over_the_shared_set(self, tmp_path, capsys):
        text = run_compare(
            tmp_path,
            {Q1: "30", Q2: "0.25", Q3: "999", Q4: "270"},
            {Q1: "999", Q2: "999", Q3: "999", Q4: "999"},
            capsys,
        )

        assert "strict               0.7500   0.0000" in text

    def test_a_percentage_stored_as_a_fraction_still_counts_as_percent_equivalent(
        self, tmp_path, capsys
    ):
        # FinQA keeps 0.25 and the run answered 25. Strict says no, the
        # percent-equivalent column says yes, and both are printed.
        text = run_compare(tmp_path, {Q2: "25"}, {Q2: "0.25"}, capsys)

        assert "strict               0.0000   1.0000" in text
        assert "percent-equivalent   1.0000   1.0000" in text

    def test_runs_with_nothing_in_common_say_so(self, tmp_path, capsys):
        text = run_compare(tmp_path, {Q1: "30"}, {Q4: "270"}, capsys)

        assert "no questions in common" in text

    def test_the_questions_each_one_won_are_named(self, tmp_path, capsys):
        text = run_compare(
            tmp_path,
            {Q1: "30", Q2: "0.25", Q3: "999", Q4: "270"},
            {Q1: "30", Q2: "999", Q3: "25", Q4: "270"},
            capsys,
        )

        assert "only the first answered, e.g." in text
        assert f"  {Q2}" in text
        assert "only the second answered, e.g." in text
        assert f"  {Q3}" in text
