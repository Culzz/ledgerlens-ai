"""Tests for turning a model reply into an answer value."""

from __future__ import annotations

from decimal import Decimal

import pytest
from app.services.reasoning import (
    FORMAT_FALLBACK,
    FORMAT_JSON,
    FORMAT_MARKER,
    FORMAT_UNPARSED,
    extract_json_object,
    last_number_in,
    normalise_text,
    parse_answer,
    to_decimal,
)


class TestToDecimal:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("127.4", "127.4"),
            ("0.935", "0.935"),
            ("100", "100"),
            ("  42  ", "42"),
            ("1,234.5", "1234.5"),
            ("$1,234.50", "1234.5"),
            ("1,234,567", "1234567"),
            ("50%", "50"),
            ("-13", "-13"),
            ("+13", "13"),
            (".455", "0.455"),
            ("1e3", "1000"),
        ],
    )
    def test_reads_the_shapes_financial_text_uses(self, text, expected):
        assert to_decimal(text) == Decimal(expected)

    def test_brackets_mean_negative(self):
        assert to_decimal("(13)") == Decimal(-13)

    def test_unicode_minus_is_a_minus(self):
        assert to_decimal("\u2212138.1") == Decimal("-138.1")

    def test_percent_sign_is_stripped_not_applied(self):
        # Whether 93.5% equals 0.935 is a scoring question, not a parsing one.
        assert to_decimal("93.5%") == Decimal("93.5")

    def test_numbers_in_words_are_not_numbers(self):
        assert to_decimal("about one hundred") is None

    def test_yes_is_not_a_number(self):
        assert to_decimal("yes") is None

    def test_empty_and_none(self):
        assert to_decimal("") is None
        assert to_decimal(None) is None
        assert to_decimal("   ") is None

    def test_accepts_a_raw_number(self):
        assert to_decimal(127.4) == Decimal("127.4")
        assert to_decimal(5) == Decimal(5)


class TestHelpers:
    def test_normalise_text_collapses_and_lowercases(self):
        assert normalise_text("  YES \n") == "yes"

    def test_last_number_in(self):
        assert last_number_in("first 3 then 42.5") == "42.5"

    def test_last_number_in_strips_separators(self):
        assert last_number_in("total 1,234") == "1234"

    def test_last_number_absent(self):
        assert last_number_in("no figures here") is None


class TestExtractJsonObject:
    def test_plain_object(self):
        assert extract_json_object('{"answer": "5"}') == {"answer": "5"}

    def test_fenced_block(self):
        text = 'Sure:\n```json\n{"answer": "5"}\n```'
        assert extract_json_object(text) == {"answer": "5"}

    def test_object_embedded_in_prose(self):
        text = 'The result is {"answer": "5"} as computed.'
        assert extract_json_object(text) == {"answer": "5"}

    def test_gives_up_on_a_non_object(self):
        assert extract_json_object("no json at all") is None

    def test_ignores_a_json_array(self):
        assert extract_json_object("[1, 2, 3]") is None


class TestParseAnswer:
    def test_json_is_preferred(self):
        parsed = parse_answer('{"answer": "127.4", "reasoning": "subtracted"}')
        assert parsed.answer == "127.4"
        assert parsed.format == FORMAT_JSON
        assert parsed.reasoning == "subtracted"
        assert parsed.value == Decimal("127.4")

    def test_json_can_carry_evidence(self):
        parsed = parse_answer('{"answer": "5", "evidence": ["table_3", "text_1"]}')
        assert parsed.evidence == ("table_3", "text_1")

    def test_json_without_an_answer_field_falls_through(self):
        parsed = parse_answer('{"other": 1}\nANSWER: 8')
        assert parsed.format == FORMAT_MARKER
        assert parsed.answer == "8"

    def test_marker(self):
        parsed = parse_answer("I subtracted the two years.\nANSWER: 94")
        assert parsed.answer == "94"
        assert parsed.format == FORMAT_MARKER

    def test_marker_is_matched_case_insensitively(self):
        assert parse_answer("answer: 7").answer == "7"

    def test_marker_ignores_units(self):
        assert parse_answer("ANSWER: 127.4 million").answer == "127.4"

    def test_marker_handles_yes_no(self):
        parsed = parse_answer("ANSWER: yes")
        assert parsed.answer == "yes"
        assert parsed.value is None

    def test_last_marker_wins(self):
        parsed = parse_answer("ANSWER: 5\nOn reflection,\nANSWER: 9")
        assert parsed.answer == "9"

    def test_falls_back_to_the_last_number(self):
        parsed = parse_answer("The ratio comes out at 0.935 overall")
        assert parsed.answer == "0.935"
        assert parsed.format == FORMAT_FALLBACK

    def test_unparsed_when_there_is_no_number(self):
        parsed = parse_answer("I cannot answer that from this document.")
        assert parsed.answer is None
        assert parsed.format == FORMAT_UNPARSED
        assert parsed.parsed is False

    def test_empty_reply_is_unparsed(self):
        assert parse_answer("").format == FORMAT_UNPARSED
        assert parse_answer(None).format == FORMAT_UNPARSED

    def test_raw_is_kept_for_the_record(self):
        assert parse_answer("ANSWER: 1").raw == "ANSWER: 1"
