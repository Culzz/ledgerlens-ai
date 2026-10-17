"""Tests for the retrieve-then-program path.

The important one is that the number never comes from the model. Everything else
here exists to keep that true: a reply can say whatever it likes and the answer is
still whatever the engine computed, or nothing at all.
"""

from __future__ import annotations

import json

from app.services.llm import Completion
from app.services.reasoning import Document, Orchestrator, label_for
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
        # One model call, and the prompt carried the evidence rather than the
        # whole document.
        assert len(provider.calls) == 1
        assert "revenue 2019 1200 2018 900" in provider.calls[0][1].content

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
