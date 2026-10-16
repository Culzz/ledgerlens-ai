"""Tests for the FinQA dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.dataset import (
    FinQASchemaError,
    load_split,
    parse_example,
    scan_quality,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "finqa_sample.json"


@pytest.fixture
def examples():
    return load_split(FIXTURE)


def _record(**overrides):
    """A minimal valid record, with pieces replaced per test."""
    record = {
        "pre_text": ["before"],
        "post_text": ["after"],
        "table": [["", "2023"], ["Revenue", "100"]],
        "id": "X/page_1.pdf-1",
        "qa": {
            "question": "What was revenue?",
            "program": "table_max(1, 1, 1)",
            "gold_inds": {},
            "exe_ans": 100,
            "program_re": "table_max(1, 1, 1)",
        },
    }
    record.update(overrides)
    return record


class TestLoading:
    def test_loads_fixture(self, examples):
        assert len(examples) == 6

    def test_ids_are_preserved(self, examples):
        assert examples[0].id == "SAMPLE/2023/page_10.pdf-1"

    def test_missing_file_is_reported_clearly(self):
        with pytest.raises(FileNotFoundError):
            load_split(FIXTURE.parent / "does_not_exist.json")

    def test_answers_are_nested_under_qa(self, examples):
        # The field that is easy to get wrong: these live under qa, not top level.
        first = examples[0]
        assert first.question == "What was the change in operating income from 2022 to 2023?"
        assert first.exe_ans == 30
        assert first.program == "subtract(150, 120)"

    def test_context_text_is_pre_plus_post_in_order(self, examples):
        first = examples[0]
        assert first.context_text == first.pre_text + first.post_text
        assert first.context_text[0].startswith("The following table")

    def test_table_shape(self, examples):
        assert examples[0].table_shape == (3, 3)


class TestNormalisation:
    def test_string_pre_text_becomes_one_item(self):
        ex = parse_example(_record(pre_text="a single paragraph"))
        assert ex.pre_text == ("a single paragraph",)

    def test_missing_pre_text_becomes_empty(self):
        record = _record()
        record.pop("pre_text")
        with pytest.raises(FinQASchemaError):
            parse_example(record)

    def test_none_cells_become_empty_strings(self):
        ex = parse_example(_record(table=[["", None], ["a", "b"]]))
        assert ex.table[0][1] == ""

    def test_numeric_cells_are_stringified(self):
        ex = parse_example(_record(table=[["h"], [100]]))
        assert ex.table[1][0] == "100"

    def test_gold_inds_defaults_to_empty(self):
        record = _record()
        record["qa"].pop("gold_inds")
        ex = parse_example(record)
        assert ex.gold_inds == {}


class TestAnnotationRows:
    """The annotation rows are the ground truth for retrieval, so they matter."""

    def test_annotation_rows_are_parsed(self):
        record = _record()
        record["qa"]["ann_table_rows"] = [1, 2]
        record["qa"]["ann_text_rows"] = [0]
        ex = parse_example(record)
        assert ex.ann_table_rows == (1, 2)
        assert ex.ann_text_rows == (0,)

    def test_missing_annotation_rows_default_to_empty(self):
        ex = parse_example(_record())
        assert ex.ann_table_rows == ()
        assert ex.ann_text_rows == ()

    def test_non_integer_annotation_rows_are_rejected(self):
        record = _record()
        record["qa"]["ann_table_rows"] = ["1"]
        with pytest.raises(FinQASchemaError, match="must contain integers"):
            parse_example(record)

    def test_boolean_annotation_row_is_rejected(self):
        # bool is a subclass of int, and True as a row index would pass silently.
        record = _record()
        record["qa"]["ann_table_rows"] = [True]
        with pytest.raises(FinQASchemaError, match="must contain integers"):
            parse_example(record)

    def test_non_list_annotation_rows_are_rejected(self):
        record = _record()
        record["qa"]["ann_text_rows"] = 3
        with pytest.raises(FinQASchemaError, match="must be a list of integers"):
            parse_example(record)

    def test_filename_is_captured_when_present(self):
        ex = parse_example(_record(filename="ETR/2016/page_23.pdf"))
        assert ex.filename == "ETR/2016/page_23.pdf"


class TestSchemaErrors:
    def test_missing_qa_is_rejected(self):
        record = _record()
        record.pop("qa")
        with pytest.raises(FinQASchemaError, match="'qa'"):
            parse_example(record)

    def test_missing_exe_ans_is_rejected(self):
        record = _record()
        record["qa"].pop("exe_ans")
        with pytest.raises(FinQASchemaError, match="qa.exe_ans"):
            parse_example(record)

    def test_empty_question_is_rejected(self):
        record = _record()
        record["qa"]["question"] = "   "
        with pytest.raises(FinQASchemaError, match="non-empty"):
            parse_example(record)

    def test_missing_id_is_rejected(self):
        record = _record()
        record.pop("id")
        with pytest.raises(FinQASchemaError, match="id"):
            parse_example(record)

    def test_ragged_table_row_is_rejected(self):
        with pytest.raises(FinQASchemaError, match="must be a list"):
            parse_example(_record(table=[["a", "b"], "not a row"]))

    def test_error_names_the_offending_record(self):
        record = _record(id="TEST/1")
        record["qa"]["exe_ans"] = None
        # exe_ans=None is a legitimate value; drop it instead to force the error.
        record["qa"].pop("exe_ans")
        with pytest.raises(FinQASchemaError) as excinfo:
            parse_example(record, index=7)
        assert "TEST/1" in str(excinfo.value)
        assert "index 7" in str(excinfo.value)

    def test_non_list_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with pytest.raises(FinQASchemaError, match="list of records"):
            load_split(bad)


class TestQualityReport:
    def test_reports_counts(self, examples):
        report = scan_quality(examples, source="fixture")
        assert report.total == 6
        assert report.unique_ids == 6
        assert report.duplicate_ids == ()
        assert report.empty_questions == 0
        assert report.empty_tables == 0
        assert report.ragged_tables == ()
        assert report.with_program_re == 6
        assert report.with_gold_inds == 6

    def test_counts_answer_kinds(self, examples):
        report = scan_quality(examples, source="fixture")
        assert report.answer_kinds["number"] == 5
        assert report.answer_kinds["string"] == 1

    def test_flags_duplicate_ids(self, examples):
        doubled = list(examples) + [examples[0]]
        report = scan_quality(doubled, source="duplicated")
        assert report.duplicate_ids == ("SAMPLE/2023/page_10.pdf-1",)

    def test_flags_ragged_tables(self):
        ex = parse_example(_record(table=[["a", "b"], ["c"]]))
        report = scan_quality([ex], source="ragged")
        assert "X/page_1.pdf-1" in report.ragged_tables

    def test_render_is_readable(self, examples):
        text = scan_quality(examples, source="fixture").render()
        assert "examples:          6" in text
        assert "answer kinds:" in text
