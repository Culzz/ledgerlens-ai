"""Tests for the named financial formulas.

Each expectation here was worked out separately - with rational arithmetic, in
``Fraction`` - rather than read back out of the implementation, so these check the
formulas rather than restate them.

Where a test asserts "close to" rather than "equal to", it says why. Two reasons
show up: a fractional power has no exact decimal form, and division in the engine's
Decimal context rounds to 28 significant digits, which is far beyond any financial
meaning but is not bit-exact.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from app.services.calculations import (
    FINANCE_PRIMITIVES,
    InvalidArgument,
    ProgramSyntaxError,
    UnsupportedOperation,
    execute_program,
)


def close(value: Decimal, expected: str, tolerance: str = "1e-9") -> None:
    """Assert two figures agree to a stated tolerance, and say what it was."""
    gap = abs(value - Decimal(expected))
    assert gap <= Decimal(tolerance), f"{value} differs from {expected} by {gap}"


class TestCagr:
    def test_the_rate_comes_back_as_a_fraction(self):
        # The convention that matters most, and the one worth stating twice:
        # a rate of 10% is 0.1. FinQA stores percentages as fractions, so a
        # formula answering 10 here would be wrong by a hundred and look fine.
        result = execute_program("cagr(100, 121, 2)").result
        assert result == Decimal("0.1")
        assert result != Decimal(10)

    def test_one_period_is_just_the_change(self):
        assert execute_program("cagr(1000, 1100, 1)").result == Decimal("0.1")

    def test_a_rate_that_is_not_a_round_number(self):
        # 1.5 ** (1/3) - 1, which has no exact decimal form. Checked to 1e-12
        # because the value came through float, as the docstring warns.
        result = execute_program("cagr(100, 150, 3)").result
        close(result, "0.1447142425533319", "1e-12")

    def test_a_shrinking_company_gives_a_negative_rate(self):
        close(execute_program("cagr(200, 100, 2)").result, "-0.2928932188134525", "1e-12")

    @pytest.mark.parametrize(
        "program, complaint",
        [
            ("cagr(0, 100, 2)", "positive starting figure"),
            ("cagr(-5, 100, 2)", "positive starting figure"),
            ("cagr(100, 0, 2)", "positive ending figure"),
            ("cagr(100, 121, 0)", "positive number of periods"),
            ("cagr(100, 121, -1)", "positive number of periods"),
        ],
    )
    def test_it_refuses_inputs_it_cannot_take_a_root_of(self, program, complaint):
        with pytest.raises(InvalidArgument, match=complaint):
            execute_program(program)


class TestPresentValue:
    def test_it_agrees_with_the_sum_it_is_a_shortcut_for(self):
        # A level cash flow is the sum of its discounted years, and that is how
        # the expectation is computed here - a different route to the same answer.
        cash_flow, rate, periods = Decimal(1000), Decimal("0.05"), 3
        total = sum(cash_flow / (1 + rate) ** year for year in range(1, periods + 1))
        got = execute_program("present_value(1000, 0.05, 3)").result
        assert abs(got - total) <= Decimal("1e-9")

    def test_the_value_for_the_documented_example(self):
        # 1000 a year for three years at 5%: 2723.248029370478...
        close(execute_program("present_value(1000, 0.05, 3)").result, "2723.248029370478")

    def test_a_zero_rate_leaves_the_cash_flows_alone(self):
        assert execute_program("present_value(1000, 0, 4)").result == Decimal(4000)

    def test_a_negative_rate_still_works(self):
        # Discounting is worth more than the cash flow when the rate is negative.
        result = execute_program("present_value(1000, -0.1, 2)").result
        assert result > Decimal(1000)

    def test_it_refuses_a_rate_it_cannot_discount_at(self):
        with pytest.raises(InvalidArgument, match="no present value"):
            execute_program("present_value(1000, -1, 3)")

    def test_it_refuses_a_period_count_that_is_not_a_count(self):
        with pytest.raises(InvalidArgument, match="positive number of periods"):
            execute_program("present_value(1000, 0.05, 0)")


class TestPerpetuity:
    def test_gordon_growth_value(self):
        # 100 growing at 3%, discounted at 8%: 100*1.03/0.05 = 2060
        assert execute_program("perpetuity(100, 0.08, 0.03)").result == Decimal(2060)

    def test_it_refuses_growth_at_or_above_the_discount_rate(self):
        with pytest.raises(InvalidArgument, match="above the growth rate"):
            execute_program("perpetuity(100, 0.05, 0.05)")
        with pytest.raises(InvalidArgument, match="above the growth rate"):
            execute_program("perpetuity(100, 0.05, 0.06)")


class TestDcf:
    def test_the_whole_thing(self):
        # 100 a year growing at 5%, discounted at 10%, with a terminal value
        # growing at 3%, over a three-year forecast. Worked out in fractions:
        # 1553.30578512396694214876...
        close(execute_program("dcf(100, 0.10, 0.05, 0.03, 3)").result, "1553.3057851239669421")

    def test_one_growth_rate_collapses_to_the_perpetuity(self):
        # The property that decided this function's shape. With the forecast and
        # the terminal growing at the same rate, the terminal value continues
        # exactly what the forecast stops, so the horizon drops out and the answer
        # is the perpetuity. True at every horizon, which is why two are checked.
        for years in (3, 7):
            built_in = execute_program(f"dcf(100, 0.10, 0.05, 0.05, {years})").result
            close(built_in, "2100")
        assert execute_program("perpetuity(100, 0.10, 0.05)").result == Decimal(2100)

    def test_a_cautious_terminal_assumption_is_worth_less(self):
        cautious = execute_program("dcf(100, 0.10, 0.05, 0.03, 3)").result
        same = execute_program("dcf(100, 0.10, 0.05, 0.05, 3)").result
        assert cautious < same

    def test_a_longer_forecast_is_worth_more_when_growth_beats_the_terminal(self):
        # Only because the forecast grows faster than the terminal value assumes.
        # With the two rates equal the horizon makes no difference at all, which
        # the test above pins down.
        three = execute_program("dcf(100, 0.10, 0.05, 0.03, 3)").result
        five = execute_program("dcf(100, 0.10, 0.05, 0.03, 5)").result
        assert five > three

    def test_a_forecast_with_no_growth_is_the_annuity_plus_a_terminal(self):
        # With both growth rates at zero the terminal is just the cash flow over
        # the rate. Composed here out of the engine's own steps, which is a check
        # the implementation cannot satisfy by repeating itself.
        trace = execute_program(
            "present_value(1000, 0.05, 3), perpetuity(1000, 0.05, 0), "
            "divide(#1, exp(1.05, 3)), add(#0, #2), dcf(1000, 0.05, 0, 0, 3)"
        )
        composed = trace.steps[3].result
        builtin = trace.steps[4].result
        assert abs(composed - builtin) <= Decimal("1e-9")

    def test_it_refuses_part_of_a_period(self):
        with pytest.raises(InvalidArgument, match="whole number of periods"):
            execute_program("dcf(100, 0.10, 0.05, 0.03, 2.5)")

    def test_it_refuses_a_terminal_value_that_will_not_converge(self):
        with pytest.raises(InvalidArgument, match="terminal growth"):
            execute_program("dcf(100, 0.05, 0.03, 0.05, 3)")


class TestEbitda:
    def test_it_adds_the_three_back(self):
        assert execute_program("ebitda(100, 20, 30, 50)").result == Decimal(200)

    def test_a_loss_year_still_works(self):
        assert execute_program("ebitda(-50, 20, 10, 30)").result == Decimal(10)

    def test_it_refuses_an_add_back_with_a_sign_on_it(self):
        # Not absorbed, and not flipped silently: a negative interest figure means
        # the caller read a statement that prints expenses negative, and being told
        # so is more useful than a number that quietly disagrees.
        with pytest.raises(InvalidArgument, match="adds back an expense"):
            execute_program("ebitda(100, -20, 30, 50)")

    def test_the_refusal_says_which_argument_and_what_to_pass(self):
        with pytest.raises(InvalidArgument) as caught:
            execute_program("ebitda(100, 20, -30, 50)")
        message = str(caught.value)
        assert "tax" in message
        assert "30" in message


class TestQuickRatio:
    def test_current_assets_less_inventory_over_current_liabilities(self):
        assert execute_program("quick_ratio(900, 300, 400)").result == Decimal("1.5")

    def test_it_refuses_to_divide_by_nothing(self):
        with pytest.raises(InvalidArgument, match="current liabilities"):
            execute_program("quick_ratio(900, 300, 0)")


class TestDebtToEquity:
    def test_the_direction_is_debt_over_equity(self):
        assert execute_program("debt_to_equity(450, 900)").result == Decimal("0.5")

    def test_a_highly_geared_company(self):
        assert execute_program("debt_to_equity(3000, 1000)").result == Decimal(3)

    def test_it_refuses_to_divide_by_nothing(self):
        with pytest.raises(InvalidArgument, match="equity"):
            execute_program("debt_to_equity(450, 0)")


class TestAsProgramSteps:
    """The formulas are engine operations, so everything else must keep working."""

    def test_a_formula_can_be_handed_an_earlier_step(self):
        trace = execute_program("subtract(900, 300), divide(#0, 400)")
        named = execute_program("quick_ratio(900, 300, 400)")
        assert trace.result == named.result

    def test_a_formula_can_be_handed_a_figure_read_from_a_table(self):
        table = (("", "2023"), ("Cash", "1200"), ("Debt", "600"))
        trace = execute_program(
            "table_sum(Cash, none), divide(#0, 600)",
            table=table,
        )
        assert trace.result == Decimal(2)
        assert trace.steps[0].result == Decimal(1200)

    def test_the_trace_names_the_formula(self):
        # This is what the audit trail shows, so the step has to read as the
        # quantity an analyst asked for rather than as opaque arithmetic.
        step = execute_program("cagr(100, 121, 2)").steps[0]
        assert step.operation == "cagr"
        assert step.arguments == ("100", "121", "2")
        assert step.render() == "#0 = cagr(100, 121, 2) = 0.1"

    def test_an_unknown_name_is_still_refused_as_unknown(self):
        with pytest.raises(UnsupportedOperation, match="dcf_x"):
            execute_program("dcf_x(1, 2, 3)")

    def test_the_wrong_number_of_arguments_names_them(self):
        # The message is what goes back to a model asked to try again, so it has
        # to say what the shape should have been.
        with pytest.raises(InvalidArgument) as caught:
            execute_program("dcf(100, 0.1, 0.05)")
        message = str(caught.value)
        assert "dcf takes 5 arguments" in message
        assert "cash_flow, rate, growth, terminal_growth, periods" in message
        assert "got 3" in message

    def test_a_formula_is_not_something_to_read_rows_with(self):
        # cagr is not a table operation, so its arguments are expressions. Naming a
        # row inside one is a mistake, and it fails rather than silently looking up
        # a label as if it were a figure.
        with pytest.raises(ProgramSyntaxError):
            execute_program("cagr(Revenue, Costs, 3)")

    def test_every_primitive_is_reachable_by_its_name(self):
        for name, primitive in FINANCE_PRIMITIVES.items():
            args = ", ".join(str(Decimal(i + 1)) for i in range(primitive.arity))
            # Either it computes or it refuses with a reason - never a KeyError,
            # which would mean the registry and the dispatch disagree.
            try:
                execute_program(f"{name}({args})")
            except InvalidArgument:
                pass
