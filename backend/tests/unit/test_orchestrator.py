"""Tests for the retrieve-then-program path.

The important one is that the number never comes from the model. Everything else
here exists to keep that true: a reply can say whatever it likes and the answer is
still whatever the engine computed, or nothing at all.
"""

from __future__ import annotations

import json

import pytest
from app.services.llm import Completion, Usage
from app.services.reasoning import (
    GROUNDED,
    RERANKED,
    UNGROUNDED,
    Document,
    Orchestrator,
    label_for,
)
from app.services.retrieval import build_units

TABLE = (
    ("", "2019", "2018"),
    ("revenue", "1200", "900"),
    ("cost of sales", "700", "600"),
)
CONTEXT = ("Revenue rose in 2019.", "Cost of sales rose more slowly.")

DOCUMENT = Document(id="doc", table=TABLE, context_text=CONTEXT)


class ScriptedProvider:
    """A provider that replies with one canned string."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list] = []

    def complete(self, messages, *, temperature: float = 0.0, max_tokens=None) -> Completion:
        self.calls.append(list(messages))
        return Completion(text=self.reply, model=self.model, provider=self.name)


class SequencedProvider:
    """A provider that replies differently each time it is called.

    The last reply repeats, which is what a model that cannot fix its program
    actually does, and what the retry budget has to survive.
    """

    name = "sequenced"
    model = "sequenced-model"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def complete(self, messages, *, temperature: float = 0.0, max_tokens=None) -> Completion:
        self.calls.append(list(messages))
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            latency_seconds=0.5,
            cost_usd=0.001,
        )


def reply(program: str, evidence: list[str] | None = None, reasoning: str = "because") -> str:
    return json.dumps({"reasoning": reasoning, "program": program, "evidence": evidence or []})


def orchestrator() -> Orchestrator:
    return Orchestrator(ScriptedProvider(""))


class TestRetrieval:
    def test_the_header_row_is_always_first(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "what was revenue in 2019")

        assert retrieval.citations[0].label == "table_0"
        assert retrieval.citations[0].text.startswith("2019")

    def test_labels_match_the_evidence_encoding_used_everywhere_else(self):
        units = build_units(DOCUMENT)
        assert [label_for(unit) for unit in units] == [
            "table_0",
            "table_1",
            "table_2",
            "text_0",
            "text_1",
        ]

    def test_top_k_is_a_ceiling_on_the_ranked_units_but_not_on_the_header(self):
        # One ranked unit plus the header, which is never counted against top_k.
        assert (
            len(Orchestrator(ScriptedProvider(""), top_k=1).retrieve(DOCUMENT, "revenue").citations)
            == 2
        )

    def test_a_word_that_is_nowhere_in_the_document_falls_back_to_everything(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "profitability")

        assert retrieval.fallback
        # The whole document, header included.
        assert len(retrieval.citations) == len(build_units(DOCUMENT))

    def test_a_normal_question_does_not_use_the_fallback(self):
        assert not orchestrator().retrieve(DOCUMENT, "revenue").fallback

    def test_a_negative_top_k_is_refused(self):
        try:
            Orchestrator(ScriptedProvider(""), top_k=0)
        except ValueError as exc:
            assert "top_k" in str(exc)
        else:  # pragma: no cover - the assertion below is the test
            raise AssertionError("top_k=0 should have been refused")

    def test_a_retriever_that_is_not_one_of_the_two_is_refused(self):
        try:
            Orchestrator(ScriptedProvider(""), retriever="semantic")
        except ValueError as exc:
            assert "retriever" in str(exc)
        else:  # pragma: no cover - the assertion below is the test
            raise AssertionError("an unknown retriever should have been refused")

    def test_a_pool_too_shallow_to_hold_top_k_is_refused(self):
        try:
            Orchestrator(ScriptedProvider(""), top_k=10, pool=4)
        except ValueError as exc:
            assert "pool" in str(exc)
        else:  # pragma: no cover - the assertion below is the test
            raise AssertionError("pool=4 with top_k=10 should have been refused")

    def test_the_reranked_retriever_offers_the_header_row_once(self):
        # The header is context and is put in front by both retrievers. The reranked
        # one ranks only answerable units, so it cannot offer it a second time as a
        # candidate - which is what its own slot used to be spent on.
        retrieval = Orchestrator(ScriptedProvider(""), retriever=RERANKED).retrieve(
            DOCUMENT, "revenue in 2019"
        )

        labels = [citation.label for citation in retrieval.citations]
        assert labels[0] == "table_0"
        assert len(labels) == len(set(labels))

    def test_the_reranked_retriever_ranks_only_units_that_can_be_an_answer(self):
        retrieval = Orchestrator(ScriptedProvider(""), retriever=RERANKED, top_k=1).retrieve(
            DOCUMENT, "cost of sales in 2019"
        )

        # The header plus one ranked unit, the same ceiling as the lexical one.
        assert len(retrieval.citations) == 2
        assert retrieval.citations[0].label == "table_0"
        assert retrieval.citations[1].label != "table_0"


class TestResolve:
    def test_the_answer_is_the_engines(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1200, 900)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.answer == "300"
        assert grounded.failure is None
        assert grounded.steps == ("#0 = subtract(1200, 900) = 300",)

    def test_a_number_in_the_reply_is_not_the_answer(self):
        # The model asserts a figure in its reasoning and the program says
        # something else. The program wins, because the program is what ran.
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q",
            reply("subtract(1200, 900)", reasoning="the answer is 999"),
            DOCUMENT,
            retrieval.by_label,
        )

        assert grounded.answer == "300"

    def test_a_single_figure_is_an_answer(self):
        # The report states the figure outright and there is nothing to compute.
        # Refusing this turned a correct reading of the document into a failure.
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve("q", reply("1200"), DOCUMENT, retrieval.by_label)

        assert grounded.answer == "1200"
        assert grounded.steps == ("#0 = 1200",)

    def test_a_table_step_reads_the_row_it_names(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("table_sum(revenue, none)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.answer == "2100"

    def test_a_program_that_will_not_parse_produces_no_answer(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1200 900)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.answer is None
        assert "Syntax" in (grounded.failure or "")

    def test_a_row_the_table_does_not_have_produces_no_answer(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("table_sum(profit margin, none)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.answer is None
        assert "profit margin" in (grounded.failure or "")

    def test_division_by_zero_produces_no_answer(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve("q", reply("divide(1, 0)"), DOCUMENT, retrieval.by_label)

        assert grounded.answer is None
        assert "division by zero" in (grounded.failure or "")

    def test_a_reply_with_no_program_produces_failure(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", "I think the answer is 300.", DOCUMENT, retrieval.by_label
        )

        assert grounded.answer is None
        assert grounded.failure == "the reply carried no program"

    def test_a_citation_to_a_label_that_was_never_offered_is_dropped(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        offered = [citation.label for citation in retrieval.citations]

        grounded = orchestrator().resolve(
            "q",
            reply("subtract(1200, 900)", evidence=[offered[0], "table_99"]),
            DOCUMENT,
            retrieval.by_label,
        )

        assert [citation.label for citation in grounded.citations] == [offered[0]]

    def test_repeated_citations_are_kept_once(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        label = retrieval.citations[0].label

        grounded = orchestrator().resolve(
            "q", reply("subtract(1200, 900)", evidence=[label, label]), DOCUMENT, retrieval.by_label
        )

        assert [citation.label for citation in grounded.citations] == [label]


class TestAnswer:
    def test_end_to_end_with_a_scripted_reply(self):
        provider = ScriptedProvider(reply("subtract(1200, 900)", reasoning="revenue grew"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert grounded.ok
        assert grounded.answer == "300"
        assert grounded.completion is not None
        assert grounded.completion.model == "scripted-model"
        # One model call, and the prompt carried the retrieved rows rather than the
        # whole document.
        assert len(provider.calls) == 1
        prompt = provider.calls[0][1].content
        assert "table_1: revenue | 1200 | 900" in prompt
        assert "text_0: Revenue rose in 2019." in prompt

    def test_a_failed_program_still_reports_what_it_tried_to_read(self):
        provider = ScriptedProvider(reply("table_sum(profit, none)", evidence=["table_1"]))
        grounded = Orchestrator(provider).answer(DOCUMENT, "revenue")

        assert not grounded.ok
        assert [citation.label for citation in grounded.citations] == ["table_1"]

    def test_the_fallback_reaches_the_answer(self):
        # "profitability" appears nowhere in the document, so retrieval falls back
        # and the flag says so rather than the question looking like a model error.
        provider = ScriptedProvider(reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "profitability")

        assert grounded.retrieval_fallback
        assert grounded.answer == "300"


class TestGroundingGate:
    def test_a_program_of_invented_figures_is_refused(self):
        # Figures the report does not state, in a program that runs perfectly. This
        # is the case the engine alone cannot catch: the arithmetic is right and the
        # inputs are made up.
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1111, 2222)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.answer is None
        assert (grounded.failure or "").startswith("UngroundedAnswer")

    def test_the_refusal_names_the_figures_it_objected_to(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1111, 2222)"), DOCUMENT, retrieval.by_label
        )

        assert "1111" in (grounded.failure or "")
        assert "2222" in (grounded.failure or "")

    def test_one_unreadable_figure_does_not_refuse_the_answer(self):
        # 500 appears nowhere. Refusing on this would refuse a lot of correct
        # answers, so the verdict is recorded and the answer is kept.
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1200, 500)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.answer == "700"
        assert grounded.grounding is not None
        assert grounded.grounding.unsourced == ("500",)

    def test_the_gate_can_be_switched_off_and_the_verdict_still_recorded(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        lenient = Orchestrator(ScriptedProvider(""), enforce_grounding=False)
        grounded = lenient.resolve("q", reply("subtract(1111, 2222)"), DOCUMENT, retrieval.by_label)

        assert grounded.answer == "-1111"
        assert grounded.grounding is not None
        assert grounded.grounding.verdict == UNGROUNDED

    def test_a_grounded_answer_carries_the_audit_trail(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1200, 900)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.grounding is not None
        assert grounded.grounding.verdict == GROUNDED
        assert grounded.grounding.steps[0].render() == (
            "#0 = subtract(1200 [table_1], 900 [table_1]) = 300"
        )

    def test_the_plain_step_line_keeps_its_meaning(self):
        # Every recorded run already has `steps` in it. Annotating that field in
        # place would quietly change what an old number meant, so the citations go
        # in `grounding` instead.
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1200, 900)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.steps == ("#0 = subtract(1200, 900) = 300",)

    def test_a_figure_read_from_a_unit_the_prompt_never_had_is_visible(self):
        # No question matches the revenue row closely, so with top_k=1 the prompt
        # carries other units. The answer stands - the figure is in the report - but
        # reading a row the model was not shown is worth being able to count.
        limited = Orchestrator(ScriptedProvider(""), top_k=1)
        retrieval = limited.retrieve(DOCUMENT, "cost of sales")
        assert "table_1" not in {citation.label for citation in retrieval.citations}

        grounded = limited.resolve("q", reply("subtract(1200, 900)"), DOCUMENT, retrieval.by_label)

        assert grounded.answer == "300"
        assert grounded.unsourced_labels() == ("table_1",)

    def test_a_figure_that_was_shown_is_not_reported_as_missing(self):
        retrieval = orchestrator().retrieve(DOCUMENT, "revenue")
        grounded = orchestrator().resolve(
            "q", reply("subtract(1200, 900)"), DOCUMENT, retrieval.by_label
        )

        assert grounded.unsourced_labels() == ()

    def test_the_gate_feeds_the_repair_loop(self):
        # An invented program is now a failure the model gets told about, rather
        # than a wrong answer that nothing catches.
        provider = SequencedProvider(reply("subtract(1111, 2222)"), reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert grounded.ok
        assert grounded.answer == "300"
        assert len(grounded.repair_notes) == 1
        assert "UngroundedAnswer" in grounded.repair_notes[0]


class TestRepair:
    def test_a_program_that_will_not_run_is_sent_back(self):
        provider = SequencedProvider(reply("subtract(1200 900)"), reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert grounded.ok
        assert grounded.answer == "300"
        assert len(provider.calls) == 2

    def test_the_engine_error_goes_back_to_the_model(self):
        provider = SequencedProvider(
            reply("table_sum(profit margin, none)"), reply("subtract(1200, 900)")
        )
        Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        follow_up = provider.calls[1][-1].content
        assert "did not run" in follow_up
        # The engine's own words, because they say which mistake it was.
        assert "profit margin" in follow_up

    def test_the_prompt_is_still_there_on_the_second_turn(self):
        provider = SequencedProvider(reply("subtract(1200 900)"), reply("subtract(1200, 900)"))
        Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        second = provider.calls[1]
        assert len(second) == 4
        assert second[0].role == "system"
        # The model's own bad reply is between the question and the correction.
        assert second[2].role == "assistant"
        assert "subtract(1200 900)" in second[2].content

    def test_the_failure_that_was_got_past_is_recorded(self):
        provider = SequencedProvider(reply("subtract(1200 900)"), reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert len(grounded.repair_notes) == 1
        assert "Syntax" in grounded.repair_notes[0]

    def test_a_repair_is_paid_for_in_the_call_it_took(self):
        provider = SequencedProvider(reply("subtract(1200 900)"), reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert grounded.completion is not None
        assert grounded.completion.usage.prompt_tokens == 20
        assert grounded.completion.completion_tokens == 10
        assert grounded.completion.latency_seconds == pytest.approx(1.0)
        assert grounded.completion.cost_usd == pytest.approx(0.002)
        assert grounded.completion.attempts == 2

    def test_an_answer_that_worked_first_time_is_one_call(self):
        provider = SequencedProvider(reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert grounded.repair_notes == ()
        assert len(provider.calls) == 1
        assert grounded.completion is not None
        assert grounded.completion.latency_seconds == pytest.approx(0.5)

    def test_a_cost_that_is_unknown_stays_unknown_after_a_repair(self):
        # Summing one known price with one unknown one and calling it a total is
        # the silent zero this project refuses everywhere else.
        class Mixed(SequencedProvider):
            def complete(self, messages, *, temperature: float = 0.0, max_tokens=None):
                completion = super().complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                if len(self.calls) == 1:
                    return Completion(text=completion.text, model=self.model, provider=self.name)
                return completion

        provider = Mixed(reply("subtract(1200 900)"), reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider).answer(DOCUMENT, "how much did revenue grow")

        assert grounded.completion is not None
        assert grounded.completion.cost_usd is None

    def test_the_retry_budget_is_respected(self):
        provider = SequencedProvider(reply("subtract(1200 900)"))
        grounded = Orchestrator(provider, max_repairs=2).answer(
            DOCUMENT, "how much did revenue grow"
        )

        assert not grounded.ok
        assert len(provider.calls) == 3
        assert len(grounded.repair_notes) == 3

    def test_repairing_can_be_switched_off(self):
        provider = SequencedProvider(reply("subtract(1200 900)"), reply("subtract(1200, 900)"))
        grounded = Orchestrator(provider, max_repairs=0).answer(
            DOCUMENT, "how much did revenue grow"
        )

        assert not grounded.ok
        assert len(provider.calls) == 1

    def test_a_negative_retry_budget_is_refused(self):
        with pytest.raises(ValueError):
            Orchestrator(ScriptedProvider(""), max_repairs=-1)
