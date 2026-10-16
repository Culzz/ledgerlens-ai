"""Tests for scoring a prediction against the reference answer."""

from __future__ import annotations

import pytest
from app.services.reasoning import parse_answer

from evaluation.scoring import (
    OUTCOME_CORRECT,
    OUTCOME_CORRECT_PERCENT_EQUIVALENT,
    OUTCOME_ERROR,
    OUTCOME_PARSE_FAILED,
    OUTCOME_WRONG_NUMBER,
    OUTCOME_WRONG_TEXT,
    judge,
)


def scored(reply: str, reference, **kwargs):
    return judge(parse_answer(reply), reference, **kwargs)


class TestNumeric:
    def test_exact_match(self):
        verdict = scored("ANSWER: 127.4", 127.4)
        assert verdict.correct
        assert verdict.outcome == OUTCOME_CORRECT

    def test_rounding_at_the_finqa_precision(self):
        # FinQA's reference answers are rounded, so a small gap has to pass.
        assert scored("ANSWER: 24.69136", 24.691).correct

    def test_inside_the_tolerance(self):
        assert scored("ANSWER: 100.0005", 100).correct

    def test_outside_the_tolerance(self):
        assert not scored("ANSWER: 100.002", 100).correct

    def test_wrong_number_is_categorised(self):
        verdict = scored("ANSWER: 5", 7)
        assert verdict.outcome == OUTCOME_WRONG_NUMBER
        assert verdict.predicted == "5"

    def test_integer_reference_matches_decimal_reply(self):
        assert scored("ANSWER: 688.0", 688).correct


class TestPercentEquivalence:
    def test_a_percentage_reply_to_a_fraction_reference(self):
        # FinQA stores 93.5% as 0.935; models answer 93.5.
        verdict = scored("ANSWER: 93.5", 0.935)
        assert verdict.percent_equivalent
        assert not verdict.correct
        assert verdict.outcome == OUTCOME_CORRECT_PERCENT_EQUIVALENT

    def test_a_percent_sign_does_not_change_it(self):
        assert scored("ANSWER: 93.5%", 0.935).percent_equivalent

    def test_a_fraction_reply_to_a_fraction_reference_is_plain_correct(self):
        verdict = scored("ANSWER: 0.935", 0.935)
        assert verdict.correct
        assert verdict.outcome == OUTCOME_CORRECT

    def test_unrelated_numbers_are_not_percent_equivalent(self):
        assert not scored("ANSWER: 12", 0.935).percent_equivalent

    def test_zero_reference_is_not_percent_extended(self):
        assert not scored("ANSWER: 1", 0.0).correct

    def test_exact_match_sets_both_flags(self):
        verdict = scored("ANSWER: 0.935", 0.935)
        assert verdict.correct and verdict.percent_equivalent


class TestText:
    def test_yes_matches_yes(self):
        assert scored("ANSWER: yes", "yes").correct

    def test_no_matches_no(self):
        assert scored("ANSWER: no", "no").correct

    def test_case_and_spacing_are_ignored(self):
        assert scored("ANSWER:  YES ", "yes").correct

    def test_wrong_word_is_categorised_as_text(self):
        verdict = scored("ANSWER: maybe", "yes")
        assert verdict.outcome == OUTCOME_WRONG_TEXT

    def test_number_reference_with_a_word_reply_is_a_wrong_number(self):
        # One side is numeric, so the failure is numeric even though the reply is not.
        verdict = scored("ANSWER: lots", 5)
        assert verdict.outcome == OUTCOME_WRONG_NUMBER


class TestFailures:
    def test_unparseable_reply(self):
        verdict = scored("I cannot answer that.", 5)
        assert verdict.outcome == OUTCOME_PARSE_FAILED
        assert verdict.predicted is None
        assert not verdict.correct

    def test_empty_reply(self):
        assert scored("", 5).outcome == OUTCOME_PARSE_FAILED

    def test_a_call_error_is_not_scored_as_a_wrong_answer(self):
        verdict = scored("ANSWER: 5", 5, call_error="ProviderRequestFailed: nope")
        assert verdict.outcome == OUTCOME_ERROR
        assert not verdict.correct

    def test_the_reference_is_always_recorded(self):
        assert scored("ANSWER: 5", 7).reference == "7"


class TestToleranceIsConfigurable:
    def test_a_tighter_tolerance_rejects_what_the_default_accepts(self):
        assert scored("ANSWER: 100.0005", 100).correct
        assert not scored("ANSWER: 100.0005", 100, tolerance=1e-9).correct


@pytest.mark.parametrize("reference", [0.10935, 30.44323, -13.0])
def test_sign_and_precision_survive_the_round_trip(reference):
    reply = f"ANSWER: {reference}"
    assert scored(reply, reference).correct
