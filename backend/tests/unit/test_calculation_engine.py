"""Tests for the deterministic calculation engine.

The engine is the one component that must be right before anything else is
built, because everything downstream trusts its answers. These tests cover the
operations, the reference encoding, number parsing and the failure modes.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from app.services.calculations import (
    GREATER_FALSE,
    GREATER_TRUE,
    CalculationError,
    Call,
    InvalidArgument,
    Literal,
    ProgramSyntaxError,
    UnresolvedReference,
    UnsupportedOperation,
    answers_match,
    default_table_resolver,
    execute_program,
    parse_program,
)
from app.services.calculations.engine import parse_number

TABLE = (
    ("", "2023", "2022"),
    ("Operating income", "150", "120"),
    ("Revenue", "900", "800"),
    ("Net income", "(1,234)", "2,500"),
)


class TestParsing:
    def test_single_step(self):
        steps = parse_program("subtract(150, 120)")
        assert len(steps) == 1
        assert steps[0].name == "subtract"
        assert [a.text for a in steps[0].args] == ["150", "120"]

    def test_nested_call(self):
        (step,) = parse_program("divide(subtract(150, 120), 120)")
        assert step.name == "divide"
        inner = step.args[0]
        assert isinstance(inner, type(step))
        assert inner.name == "subtract"

    def test_multiple_steps_and_references(self):
        steps = parse_program("subtract(150, 120), divide(#0, 120), multiply(#1, 100)")
        assert [s.name for s in steps] == ["subtract", "divide", "multiply"]
        assert steps[1].args[0] == Literal("#0")
        assert steps[1].args[0].is_reference
        assert steps[1].args[0].reference_index == 0

    def test_negative_numbers(self):
        (step,) = parse_program("add(-5, 3)")
        assert [a.text for a in step.args] == ["-5", "3"]

    def test_whitespace_is_ignored(self):
        (step,) = parse_program("  subtract( 150 ,  120 )  ")
        assert step.name == "subtract"

    @pytest.mark.parametrize(
        "bad",
        ["", "   ", "subtract(150, 120", "subtract 150, 120)", "#0", "revenue", "123, 456"],
    )
    def test_syntax_errors_are_rejected(self, bad):
        with pytest.raises((ProgramSyntaxError, CalculationError)):
            parse_program(bad)

    def test_a_single_figure_is_a_program(self):
        # The report already states the answer, so there is nothing to compute.
        (step,) = parse_program("189")
        assert step == Literal("189")

    def test_a_lone_reference_is_still_refused(self):
        # It was written for this case: a reference with no step behind it is not
        # an answer, it is a program that lost its beginning.
        with pytest.raises(ProgramSyntaxError, match="every step must be a function call"):
            parse_program("#0")

    def test_unexpected_character_is_reported(self):
        with pytest.raises(ProgramSyntaxError, match="unexpected character"):
            parse_program("subtract(150, &)")


class TestArithmetic:
    def test_subtract(self):
        assert execute_program("subtract(150, 120)").result == Decimal(30)

    def test_add(self):
        assert execute_program("add(150, 120)").result == Decimal(270)

    def test_multiply(self):
        assert execute_program("multiply(150, 2)").result == Decimal(300)

    def test_divide_keeps_decimal_type(self):
        trace = execute_program("divide(1, 3)")
        assert isinstance(trace.result, Decimal)

    def test_exp_with_integer_power_is_exact(self):
        assert execute_program("exp(2, 10)").result == Decimal(1024)

    def test_nested_arithmetic(self):
        trace = execute_program("divide(subtract(150, 120), 120)")
        assert trace.result == Decimal("0.25")

    def test_decimal_avoids_float_drift(self):
        # 0.1 + 0.2 is not 0.3 in binary floating point.
        assert execute_program("add(0.1, 0.2)").result == Decimal("0.3")

    def test_negative_operands(self):
        assert execute_program("add(-5, 3)").result == Decimal(-2)

    def test_division_by_zero_is_an_error_not_infinity(self):
        with pytest.raises(CalculationError, match="division by zero"):
            execute_program("divide(5, 0)")


class TestReferences:
    def test_multi_step_program_uses_previous_results(self):
        trace = execute_program("subtract(150, 120), divide(#0, 120), multiply(#1, 100)")
        assert trace.result == Decimal(25)

    def test_each_step_is_recorded(self):
        trace = execute_program("subtract(150, 120), multiply(#0, 2)")
        assert [s.index for s in trace.steps] == [0, 1]
        assert trace.steps[0].result == Decimal(30)
        assert trace.steps[1].result == Decimal(60)

    def test_reference_rendering_is_human_readable(self):
        trace = execute_program("subtract(150, 120), multiply(#0, 2)")
        rendered = trace.render()
        assert "#0 = subtract(150, 120) = 30" in rendered

    def test_forward_reference_is_rejected(self):
        with pytest.raises(UnresolvedReference, match="#1"):
            execute_program("multiply(#1, 2), add(1, 2)")


class TestComparison:
    def test_greater_true(self):
        assert execute_program("greater(150, 120)").result == GREATER_TRUE

    def test_greater_false(self):
        assert execute_program("greater(120, 150)").result == GREATER_FALSE

    def test_greater_result_is_not_numeric(self):
        assert execute_program("greater(150, 120)").is_numeric is False


class TestTableOperations:
    SETTLEMENTS = (
        ("december 31,", "2016", "2015", "2014"),
        ("settlements", "-13 ( 13 )", "-19 ( 19 )", "-2 ( 2 )"),
    )

    def test_sum_over_a_whole_row(self):
        assert execute_program("table_sum(operating income, none)", table=TABLE).result == Decimal(
            270
        )

    def test_max_over_a_whole_row(self):
        assert execute_program("table_max(revenue, none)", table=TABLE).result == Decimal(900)

    def test_min_over_a_whole_row(self):
        assert execute_program("table_min(revenue, none)", table=TABLE).result == Decimal(800)

    def test_average_over_a_whole_row(self):
        assert execute_program("table_average(revenue, none)", table=TABLE).result == Decimal(850)

    def test_parenthesised_negative_is_read_as_negative(self):
        assert execute_program("table_max(net income, 2023)", table=TABLE).result == Decimal(-1234)

    def test_column_restriction_by_label(self):
        assert execute_program(
            "table_sum(settlements, 2016)", table=self.SETTLEMENTS
        ).result == Decimal(-13)

    def test_real_average_case_with_footnote_markers(self):
        # From dev.json: AES/2016/page_191.pdf-3, exe_ans -11.33333.
        trace = execute_program("table_average(settlements, none)", table=self.SETTLEMENTS)
        assert answers_match(trace.result, Decimal("-11.33333"), tolerance=Decimal("1e-5"))

    def test_row_label_matching_is_case_insensitive(self):
        assert execute_program("table_sum(REVENUE, none)", table=TABLE).result == Decimal(1700)

    def test_unknown_row_is_rejected(self):
        with pytest.raises(InvalidArgument, match="no row labelled"):
            execute_program("table_sum(shareholders equity, none)", table=TABLE)

    def test_unknown_column_is_rejected(self):
        with pytest.raises(InvalidArgument, match="no column labelled"):
            execute_program("table_sum(revenue, 2019)", table=TABLE)

    def test_row_with_no_numbers_is_rejected(self):
        table = (("h", "2016"), ("notes", "( in millions )"))
        with pytest.raises(InvalidArgument, match="no numeric values"):
            execute_program("table_sum(notes, none)", table=table)

    def test_missing_table_is_reported(self):
        with pytest.raises(InvalidArgument, match="no table"):
            execute_program("table_sum(revenue, none)")

    def test_too_many_arguments_is_rejected(self):
        with pytest.raises(InvalidArgument, match="row label and an optional column"):
            execute_program("table_sum(revenue, 2023, 2022)", table=TABLE)

    def test_nested_call_in_table_arguments_is_rejected_at_parse_time(self):
        with pytest.raises(ProgramSyntaxError, match="unexpected"):
            execute_program("table_sum(add(1, 0), none)", table=TABLE)

    def test_resolver_rejects_non_literal_arguments(self):
        # Guards callers that build programs directly instead of going through
        # the parser, which rejects this case earlier.
        nested = Call("add", (Literal("1"), Literal("0")))
        with pytest.raises(InvalidArgument, match="literal labels"):
            default_table_resolver(TABLE, (nested, Literal("none")))


class TestConstants:
    """`const_<value>` is a literal, not a call. 159 dev records rely on this."""

    def test_const_is_the_literal_value(self):
        assert execute_program("divide(637, const_5)").result == Decimal("127.4")

    def test_const_100(self):
        assert execute_program("multiply(25, const_100)").result == Decimal(2500)

    def test_const_1000(self):
        assert execute_program("multiply(2, const_1000)").result == Decimal(2000)

    def test_const_1000000(self):
        assert execute_program("multiply(1, const_1000000)").result == Decimal(1000000)

    def test_const_with_m_prefix_is_negative(self):
        assert execute_program("multiply(351, const_m1)").result == Decimal(-351)

    def test_const_inside_a_multi_step_program(self):
        trace = execute_program("subtract(150, 120), divide(#0, 120), multiply(#1, const_100)")
        assert trace.result == Decimal(25)

    def test_malformed_const_is_rejected(self):
        with pytest.raises(ProgramSyntaxError, match="malformed constant"):
            parse_program("divide(1, const_)")


class TestNoneKeyword:
    def test_none_parses_as_a_literal_table_argument(self):
        (step,) = parse_program("table_max(revenue, none)")
        assert step.args[1] == Literal("none")

    def test_labels_with_spaces_are_one_argument(self):
        (step,) = parse_program("table_max(operating margin, none)")
        assert step.args[0] == Literal("operating margin")
        assert step.args[1] == Literal("none")


class TestUnknownOperations:
    def test_unknown_operation_raises(self):
        with pytest.raises(UnsupportedOperation, match="frobnicate"):
            execute_program("frobnicate(1, 2)")

    def test_wrong_argument_count_is_rejected(self):
        with pytest.raises(InvalidArgument, match="2 arguments"):
            execute_program("subtract(150)")

    def test_calling_a_comparison_result_as_a_number_is_rejected(self):
        with pytest.raises(InvalidArgument, match="needs numbers"):
            execute_program("greater(150, 120), add(#0, 1)")


class TestNumberParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("150", Decimal(150)),
            ("1,234", Decimal(1234)),
            ("(1,234)", Decimal(-1234)),
            ("( 1,234 )", Decimal(-1234)),
            ("$5,000", Decimal(5000)),
            ("12.5%", Decimal("0.125")),
            ("-42", Decimal(-42)),
            ("\u22125", Decimal(-5)),  # unicode minus
            ("\u2014", Decimal(0)),  # em dash means zero here
            ("-", Decimal(0)),
            ("0", Decimal(0)),
            (" 1 234 ", Decimal(1234)),
            (".455", Decimal("0.455")),
            ("-.5", Decimal("-0.5")),
            # Footnote markers, straight out of the real tables.
            ("-13 ( 13 )", Decimal(-13)),
            ("-1 ( 1 )", Decimal(-1)),
            ("100% ( 100 % )", Decimal(1)),
            ("$ 373", Decimal(373)),
        ],
    )
    def test_parses_financial_shapes(self, text, expected):
        assert parse_number(text) == expected

    def test_percent_means_a_fraction(self):
        # multiply(25%, 5283.3) answers 1320.825 in the real data, so 25% is 0.25.
        assert parse_number("25%") == Decimal("0.25")

    def test_percent_literals_in_programs(self):
        assert execute_program("multiply(526.5, 2.4%)").result == Decimal("12.636")

    def test_leading_decimal_literals_in_programs(self):
        assert execute_program("subtract(.3, .9)").result == Decimal("-0.6")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "n/a",
            "abc",
            "( in millions )",
            "( in thousands )",
            "total",
            "thereafter",
        ],
    )
    def test_unparseable_values_raise(self, bad):
        with pytest.raises(InvalidArgument):
            parse_number(bad)


class TestAnswerMatching:
    def test_exact_numbers_match(self):
        assert answers_match(Decimal(30), 30)

    def test_within_tolerance(self):
        assert answers_match(Decimal("25.00001"), 25)

    def test_outside_tolerance(self):
        assert not answers_match(Decimal("25.1"), 25)

    def test_tolerance_is_configurable(self):
        assert not answers_match(Decimal("25.01"), 25, tolerance=Decimal("1e-4"))

    def test_strings_compare_case_insensitively(self):
        assert answers_match("YES", "yes")

    def test_string_and_number_do_not_match(self):
        assert not answers_match("yes", 1)

    def test_none_never_matches(self):
        assert not answers_match(None, 30)
        assert not answers_match(30, None)

    def test_numeric_strings_compare_numerically(self):
        assert answers_match("30", 30)
