"""Deciding whether a predicted answer is right.

Two conventions, both reported, because they disagree on a real slice of FinQA
and choosing one silently would move the headline number.

**Strict.** The predicted number equals the reference number, after both are
parsed, within a tolerance. This is FinQA's own convention, so a number measured
this way can be placed next to published ones.

**Percent-equivalent.** The above, or one side is a hundred times the other.
FinQA stores a percentage answer as a fraction — ``0.935`` for 93.5% — while
models routinely answer ``93.5%``. Reading that as a wrong answer overstates the
error rate, and reading it as right without saying so overstates the accuracy.
So both counts are kept and the gap between them is visible in every report.

Answers that are not numbers, such as the ``yes``/``no`` questions, are compared
as normalised text under the strict rule and are unaffected by the percent rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.reasoning import ParsedAnswer, normalise_text, to_decimal

DEFAULT_TOLERANCE = 1e-3

# Why a prediction was marked wrong. Only mechanically determinable reasons live
# here; the interpretive ones from the protocol need a human or a second model.
OUTCOME_PARSE_FAILED = "parse_failed"
OUTCOME_WRONG_NUMBER = "wrong_number"
OUTCOME_WRONG_TEXT = "wrong_text"
OUTCOME_ANSWERED_WHEN_UNANSWERABLE = "answered_when_unanswerable"
OUTCOME_ABSTAINED = "abstained"
OUTCOME_CORRECT = "correct"
OUTCOME_CORRECT_PERCENT_EQUIVALENT = "correct_percent_equivalent"
OUTCOME_ERROR = "call_error"


@dataclass(frozen=True)
class Judgement:
    """How one prediction scored against the reference answer."""

    correct: bool
    percent_equivalent: bool
    outcome: str
    predicted: str | None
    reference: str

    @property
    def strict(self) -> bool:
        return self.correct


def _numeric_match(predicted: str, reference: str, tolerance: float) -> bool:
    predicted_value = to_decimal(predicted)
    reference_value = to_decimal(reference)
    if predicted_value is None or reference_value is None:
        return False
    return abs(predicted_value - reference_value) <= tolerance


def _percent_equivalent(predicted: str, reference: str, tolerance: float) -> bool:
    predicted_value = to_decimal(predicted)
    reference_value = to_decimal(reference)
    if predicted_value is None or reference_value is None:
        return False
    if reference_value == 0:
        # A hundred times zero is still zero, so there is nothing extra to try.
        return False
    return abs(predicted_value / 100 - reference_value) <= tolerance


def judge(
    parsed: ParsedAnswer,
    reference: object,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    call_error: str | None = None,
) -> Judgement:
    """Score one prediction against the reference answer."""
    reference_text = str(reference)

    if call_error is not None:
        return Judgement(False, False, OUTCOME_ERROR, None, reference_text)

    if not parsed.parsed:
        return Judgement(False, False, OUTCOME_PARSE_FAILED, None, reference_text)

    predicted = parsed.answer
    assert predicted is not None  # narrowed by the check above

    correct = _numeric_match(predicted, reference_text, tolerance)
    if not correct:
        # Fall back to text, which is what the yes/no questions need.
        correct = normalise_text(predicted) == normalise_text(reference_text)

    if correct:
        return Judgement(True, True, OUTCOME_CORRECT, predicted, reference_text)

    if _percent_equivalent(predicted, reference_text, tolerance):
        return (
            Judgement(False, True, OUTCOME_CORRECT_PERCENT_EQUIVALENT, predicted, reference_text)
        )

    if to_decimal(predicted) is not None or to_decimal(reference_text) is not None:
        outcome = OUTCOME_WRONG_NUMBER
    else:
        outcome = OUTCOME_WRONG_TEXT

    return Judgement(False, False, outcome, predicted, reference_text)
