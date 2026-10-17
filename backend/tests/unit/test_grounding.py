"""Tests for tracing a program's figures back to the report.

The check has one job and two ways to get it wrong. It can miss a figure the model
made up, and it can accuse a figure the report states. The second is worse - it
throws away correct answers - so most of what is here is about the boundary: unit
notes, parenthesised negatives, percents, and figures that are arithmetic rather
than data.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from app.services.calculations import ProgramSyntaxError, execute_program
from app.services.reasoning import (
    CONSTANT,
    GROUNDED,
    LABEL,
    PARTLY_GROUNDED,
    REFERENCE,
    SOURCED,
    UNGROUNDED,
    UNSOURCED,
    Document,
    Stated,
    audit,
    cell_numbers,
    declared_unit,
    numbers_in,
    read_report,
    readings,
    row_numbers,
)

TABLE = (
    ("", "2019", "2018"),
    ("revenue", "1200", "900"),
    ("cost of sales", "700", "600"),
    ("net income", "-13 ( 13 )", "40"),
)
CONTEXT = ("Revenue rose in 2019.", "The operating margin was 12.5% for the year.")
DOCUMENT = Document(id="doc", table=TABLE, context_text=CONTEXT)


def grounding(program: str, document: Document = DOCUMENT):
    trace = execute_program(program, table=document.table)
    return audit(trace, read_report(document))


def roles(program: str) -> list[tuple[str, str]]:
    result = grounding(program)
    return [
        (operand.text.split(" [")[0].split(" (")[0], operand.role)
        for step in result.steps
        for operand in step.operands
    ]


class TestReadingFigures:
    def test_prose_yields_every_figure_in_it(self):
        assert numbers_in("revenue of 1,200 rose from 900 in 2018") == {
            Decimal(1200),
            Decimal(900),
            Decimal(2018),
        }

    def test_a_percent_is_read_both_ways(self):
        # 12.5% is 0.125 to the engine and 12.5 to a model writing a program. Both
        # have to count, or the percent convention causes false accusations.
        assert numbers_in("the margin was 12.5%") == {Decimal("0.125"), Decimal("12.5")}

    def test_prose_with_no_figure_yields_nothing(self):
        assert numbers_in("revenue rose on stronger volumes") == set()

    def test_a_table_cell_is_read_the_engines_way(self):
        # -13, not -13 and 13. A program that loses a sign should not be able to
        # claim the cell as its source.
        assert cell_numbers("-13 ( 13 )") == {Decimal(-13)}

    def test_a_cell_with_no_digit_is_not_a_figure(self):
        # The parser reads a lone dash as zero, which would make every row a source
        # for zero. Unit notes and footnote markers are not figures either.
        assert row_numbers(("-", "( in millions )", "n/a", "total", "")) == set()

    def test_a_percent_cell_is_read_both_ways(self):
        assert cell_numbers("12.5%") == {Decimal("0.125"), Decimal("12.5")}

    def test_a_cell_stating_its_own_unit_is_read(self):
        # The table says "$ 739 million" and a program writes 739, because 739 is
        # the figure the cell states. Reading only the cell whole, which fails on
        # the unit word, made the check call a stated figure invented.
        found = cell_numbers("$ 739 million")

        assert Decimal(739) in found
        assert Decimal(739 * 10**6) in found

    def test_a_cell_reading_is_the_engines_reading_when_it_works(self):
        # No unit word, so no unit readings: one cell, one figure.
        assert cell_numbers("1,200") == {Decimal(1200)}


class TestUnits:
    def test_prose_scales_a_figure_that_carries_a_unit(self):
        # "for $ 240.0 million in 2010" states 240.0 under a unit, and the model
        # wrote 240000000.
        assert numbers_in("we repurchased shares for $ 240.0 million in 2010") >= {
            Decimal("240.0"),
            Decimal(240_000_000),
        }

    def test_the_look_ahead_is_anchored_at_the_figure(self):
        # "3 million ... 4 thousand". Each figure takes its own unit word, and the
        # years take none.
        assert readings("3 million in 2019 and 4 thousand in 2020") == [
            Stated(value=Decimal(3), unit=Decimal(10) ** 6),
            Stated(value=Decimal(2019)),
            Stated(value=Decimal(4), unit=Decimal(10) ** 3),
            Stated(value=Decimal(2020)),
        ]

    def test_an_amount_reads_in_the_other_units_a_report_uses(self):
        # The report says "$ 6.8 billion" and a program writes 6800, which is the
        # same amount in millions. The amount is the report's figure; which unit
        # either side happens to write it in is presentation.
        found = numbers_in("cash flow provided by operations was $ 6.8 billion in 2012")

        assert Decimal(6_800_000_000) in found
        assert Decimal(6_800_000) in found
        assert Decimal(6800) in found

    def test_a_report_says_what_its_figures_are_in(self):
        document = Document(
            id="doc",
            table=(("", "2019", "2018"), ("revenue", "240.0", "210.0")),
            context_text=("the table below is in millions .",),
        )
        report = read_report(document)

        assert report.unit == Decimal(10) ** 6
        assert report.unit_name == "millions"
        # The figure as written and the figure in full, because a program may use
        # either and both are the report saying the same thing.
        assert report.figures["table_1"] == {
            Decimal("240.0"),
            Decimal(240_000_000),
            Decimal("210.0"),
            Decimal(210_000_000),
        }

    def test_a_report_that_says_nothing_about_units(self):
        assert declared_unit(DOCUMENT) is None
        assert read_report(DOCUMENT).unit is None

    def test_a_stated_figure_written_out_in_full_is_sourced(self):
        document = Document(
            id="doc",
            table=(("", "2019", "2018"), ("revenue", "240.0", "210.0")),
            context_text=("the table below is in millions .",),
        )
        result = grounding("divide(240000000, 210000000)", document)

        assert result.verdict == GROUNDED
        assert result.unsourced == ()
        # And the trail says which unit the figures were compared in, rather than
        # showing a citation with no hint that the report writes them differently.
        assert result.report_unit == "millions"
        assert result.steps[0].render() == (
            "#0 = divide(240000000 [table_1], 210000000 [table_1]) = 1.142857142857142857142857143"
        )

    def test_a_figure_stated_in_another_unit_reads_in_the_reports_unit(self):
        # "4.6 billion" in a narrative and a table of millions: 4600. This is the
        # case that made the check throw away a right answer on the measured run.
        document = Document(
            id="doc",
            table=(("( in millions )", "2019"), ("cash", "5500")),
            context_text=("we held $ 4.6 billion of our cash abroad .",),
        )
        report = read_report(document)

        assert report.unit_name == "millions"
        assert Decimal(4600) in report.figures["text_0"]

        result = grounding("divide(4600, 5500)", document)
        assert result.verdict == GROUNDED
        assert result.unsourced == ()

    def test_the_load_bearing_unit_does_not_excuse_everything(self):
        document = Document(
            id="doc",
            table=(("", "2019", "2018"), ("revenue", "240.0", "210.0")),
            context_text=("the table below is in millions .",),
        )
        # 250 is not 240 and not 240 million either.
        result = grounding("divide(250000000, 210000000)", document)

        assert result.verdict == PARTLY_GROUNDED
        assert result.unsourced == ("250000000",)


class TestDocumentNumbers:
    def test_every_unit_gets_an_entry(self):
        numbers = read_report(DOCUMENT).figures

        assert set(numbers) == {"table_0", "table_1", "table_2", "table_3", "text_0", "text_1"}

    def test_a_row_is_read_cell_by_cell(self):
        # Row 0 is the header, so table_1 is the revenue row.
        assert read_report(DOCUMENT).figures["table_1"] == {Decimal(1200), Decimal(900)}

    def test_a_narrative_line_is_scanned(self):
        assert read_report(DOCUMENT).figures["text_1"] == {
            Decimal("0.125"),
            Decimal("12.5"),
        }


class TestVerdicts:
    def test_figures_from_a_row_are_sourced_and_cited(self):
        result = grounding("subtract(1200, 900)")

        assert result.verdict == GROUNDED
        assert not result.blocks
        assert result.labels == ("table_1",)
        assert result.sourced_count == 2

    def test_a_figure_the_report_does_not_state_is_unsourced(self):
        result = grounding("subtract(1200, 500)")

        assert result.verdict == PARTLY_GROUNDED
        assert result.unsourced == ("500",)
        # Partly grounded is reported, not refused. One unreadable figure is a
        # weaker answer, not an invented one.
        assert not result.blocks

    def test_a_program_of_invented_figures_is_refused(self):
        result = grounding("subtract(1111, 2222)")

        assert result.verdict == UNGROUNDED
        assert result.blocks
        assert result.unsourced == ("1111", "2222")

    def test_a_single_invented_figure_is_refused(self):
        result = grounding("999")

        assert result.verdict == UNGROUNDED
        assert result.blocks

    def test_a_single_stated_figure_is_grounded(self):
        result = grounding("1200")

        assert result.verdict == GROUNDED
        assert result.labels == ("table_1",)

    def test_a_scale_constant_is_allowed(self):
        # 100 to turn a ratio into a percent is arithmetic, not a reading of the
        # report, and refusing it would refuse most percentage answers.
        result = grounding("multiply(1200, 100)")

        assert result.verdict == GROUNDED
        assert result.unsourced == ()

    def test_a_percent_from_the_narrative_is_sourced(self):
        result = grounding("multiply(12.5, 2)")

        assert result.verdict == GROUNDED
        assert result.labels == ("text_1",)
        assert result.unsourced == ()

    def test_a_count_of_periods_is_allowed(self):
        # Averaging three years is divide(total, 3). The 3 is how many years there
        # are, not a figure the report states, and refusing it refused a correct
        # answer on the measured run.
        result = grounding("divide(1200, 3)")

        assert result.verdict == GROUNDED
        assert result.unsourced == ()

    def test_the_count_carve_out_stays_narrow(self):
        # 13 is past the last count and is not a scale, so it is still examined.
        result = grounding("divide(1200, 13)")

        assert result.verdict == PARTLY_GROUNDED
        assert result.unsourced == ("13",)

    def test_nested_figures_are_examined_too(self):
        # The engine records one step for the whole expression, so walking the steps
        # would see only the outer figure and miss both inner ones.
        result = grounding("divide(subtract(1111, 2222), 1200)")

        assert result.verdict == PARTLY_GROUNDED
        assert result.unsourced == ("1111", "2222")

    def test_a_table_argument_is_a_label_not_a_figure(self):
        # "revenue" must not be looked up as a number, and 47 must be.
        result = grounding("divide(table_sum(revenue, none), 47)")

        assert result.verdict == UNGROUNDED
        assert result.unsourced == ("47",)

    def test_a_pure_table_program_is_grounded(self):
        result = grounding("table_sum(revenue, none)")

        assert result.verdict == GROUNDED
        assert result.literal_count == 0


class TestAuditTrail:
    def test_each_figure_carries_the_unit_it_came_from(self):
        result = grounding("subtract(1200, 900)")

        assert result.steps[0].render() == "#0 = subtract(1200 [table_1], 900 [table_1]) = 300"

    def test_an_invented_figure_is_marked_in_the_trail(self):
        result = grounding("divide(1200, 45)")

        assert (
            result.steps[0].render()
            == "#0 = divide(1200 [table_1], 45 (unverified)) = 26.66666666666666666666666667"
        )

    def test_a_reference_is_shown_as_a_reference(self):
        result = grounding("subtract(1200, 900), divide(#0, 1200)")

        assert result.steps[1].render() == "#1 = divide(#0, 1200 [table_1]) = 0.25"

    def test_a_table_call_shows_its_labels(self):
        result = grounding("table_sum(revenue, none)")

        assert result.steps[0].render() == "#0 = table_sum(revenue, none) = 2100"

    def test_the_roles_are_the_ones_the_renderer_leans_on(self):
        assert roles("subtract(1200, 900), divide(#0, 999)") == [
            ("1200", SOURCED),
            ("900", SOURCED),
            ("#0", REFERENCE),
            ("999", UNSOURCED),
        ]

    def test_a_table_call_operand_is_neither_a_reference_nor_a_figure(self):
        result = grounding("table_sum(revenue, none)")
        operand = result.steps[0].operands[0]

        assert operand.role == LABEL

    def test_a_constant_is_reported_as_one(self):
        result = grounding("multiply(1200, 100)")
        operand = result.steps[0].operands[1]

        assert operand.role == CONSTANT
        assert operand.value == Decimal(100)

    def test_it_serialises_for_the_api(self):
        payload = grounding("subtract(1200, 900)").to_dict()

        assert payload["verdict"] == GROUNDED
        assert payload["labels"] == ["table_1"]
        assert payload["steps"][0]["operands"][0]["labels"] == ["table_1"]

    def test_a_program_that_will_not_parse_is_not_audited(self):
        # The engine refuses it before there is anything to audit. ``audit`` takes an
        # executed trace, so there is no path here that can quietly produce a
        # grounding verdict for a program that never ran.
        with pytest.raises(ProgramSyntaxError):
            grounding("subtract(1200 900)")


class TestItChecksTheWholeReportNotJustThePrompt:
    def test_a_figure_in_a_row_retrieval_never_surfaced_is_still_sourced(self):
        # The point of checking the document rather than the offered evidence: a
        # retrieval miss is not a fabrication, and treating it as one would throw
        # away correct answers for the retriever's mistake.
        document = Document(
            id="doc",
            table=TABLE,
            context_text=CONTEXT,
        )
        result = grounding("subtract(700, 600)", document)

        assert result.verdict == GROUNDED
        assert result.labels == ("table_2",)
