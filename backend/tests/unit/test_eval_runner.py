"""Tests for the evaluation runner.

The whole harness is driven here through the fake provider and a temporary
directory, so resume, halting and scoring are all covered without a network, an
API key or any spend.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.core.config import get_settings
from app.services.dataset import FinQAExample
from app.services.llm import (
    BudgetLedger,
    FakeLLM,
    ProviderRateLimited,
    ProviderRequestFailed,
)
from app.services.reasoning import RERANKED

from evaluation.baselines import DirectLLM, ProgramLLM
from evaluation.run_eval import (
    build_config,
    execute,
    load_predictions,
    percentile,
    render,
    score_all,
    summarise,
)


def example(index: int, answer, *, question: str = "what was net sales?") -> FinQAExample:
    return FinQAExample(
        id=f"DOC/{index}",
        question=question,
        table=(("", "2016", "2015"), ("net sales", "100", "90")),
        pre_text=("the company reported net sales.",),
        post_text=("amounts are in millions.",),
        program="table_max(net sales, 2016)",
        exe_ans=answer,
    )


def system_answering(reply: str) -> DirectLLM:
    return DirectLLM(FakeLLM(responder=lambda messages: reply))


class FailingProvider:
    """A provider that always fails, for the error paths."""

    name = "failing"
    model = "none"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        raise self.error


def fresh_budget(tmp_path: Path, limit: float | None = 25.0) -> BudgetLedger:
    return BudgetLedger(tmp_path / "budget.json", limit_usd=limit)


class TestExecute:
    def test_answers_every_question(self, tmp_path):
        examples = [example(1, 100), example(2, 100), example(3, 200)]
        path = tmp_path / "predictions.jsonl"

        predictions, halted = execute(
            system_answering("ANSWER: 100"),
            examples,
            path,
            budget=fresh_budget(tmp_path),
        )

        assert halted is None
        assert len(predictions) == 3
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3

    def test_resuming_does_not_re_answer_or_spend_again(self, tmp_path):
        examples = [example(1, 100), example(2, 100)]
        path = tmp_path / "predictions.jsonl"
        budget = fresh_budget(tmp_path)

        provider = FakeLLM(responder=lambda messages: "ANSWER: 100")
        system = DirectLLM(provider)
        execute(system, examples, path, budget=budget)
        assert provider.call_count == 2

        _, halted = execute(system, examples, path, budget=budget)
        assert halted is None
        assert provider.call_count == 2, "a resumed run must not call the model again"

    def test_a_partial_run_resumes_only_what_is_left(self, tmp_path):
        examples = [example(1, 100), example(2, 100), example(3, 100)]
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "system": "direct@fake",
                    "question_id": "DOC/1",
                    "parsed": {"answer": "100", "format": "marked", "raw": "ANSWER: 100"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        provider = FakeLLM(responder=lambda messages: "ANSWER: 100")
        predictions, halted = execute(
            DirectLLM(provider), examples, path, budget=fresh_budget(tmp_path)
        )

        assert halted is None
        assert provider.call_count == 2, "the already-answered question is skipped"
        assert len(predictions) == 3

    def test_a_failed_question_is_retried_on_resume(self, tmp_path):
        """A provider hiccup is not an answer, so resuming must try it again."""
        examples = [example(1, 100)]
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "system": "direct@fake",
                    "question_id": "DOC/1",
                    "parsed": {"answer": None, "format": "unparsed", "raw": ""},
                    "call_error": "ProviderUnavailable: 503 on attempt 4",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        provider = FakeLLM(responder=lambda messages: "ANSWER: 100")
        predictions, halted = execute(
            DirectLLM(provider), examples, path, budget=fresh_budget(tmp_path)
        )

        assert halted is None
        assert provider.call_count == 1, "the failed question is worth another try"
        assert predictions["DOC/1"]["call_error"] is None
        assert predictions["DOC/1"]["parsed"]["answer"] == "100"

    def test_a_single_bad_question_does_not_stop_the_run(self, tmp_path):
        examples = [example(1, 100), example(2, 100)]
        path = tmp_path / "predictions.jsonl"
        provider = FailingProvider(ProviderRequestFailed("400: bad request"))

        predictions, halted = execute(
            DirectLLM(provider), examples, path, budget=fresh_budget(tmp_path)
        )

        assert halted is None
        assert provider.calls == 2
        assert all(record["call_error"] for record in predictions.values())

    def test_a_rate_limit_stops_the_run(self, tmp_path):
        examples = [example(1, 100), example(2, 100)]
        path = tmp_path / "predictions.jsonl"
        provider = FailingProvider(ProviderRateLimited("429: slow down"))

        predictions, halted = execute(
            DirectLLM(provider), examples, path, budget=fresh_budget(tmp_path)
        )

        assert halted is not None and "ProviderRateLimited" in halted
        assert provider.calls == 1, "it should stop rather than hammer the quota"
        assert predictions == {}

    def test_an_exhausted_budget_stops_before_calling(self, tmp_path):
        examples = [example(1, 100), example(2, 100)]
        provider = FakeLLM(responder=lambda messages: "ANSWER: 100")

        _, halted = execute(
            DirectLLM(provider),
            examples,
            tmp_path / "predictions.jsonl",
            budget=fresh_budget(tmp_path, limit=0.0),
        )

        assert halted is not None and "budget exhausted" in halted
        assert provider.call_count == 0

    def test_token_usage_lands_in_the_ledger(self, tmp_path):
        budget = fresh_budget(tmp_path)
        execute(
            system_answering("ANSWER: 100"),
            [example(1, 100)],
            tmp_path / "predictions.jsonl",
            budget=budget,
        )
        assert budget.calls == 1
        assert budget.total_tokens > 0


class TestLoadPredictions:
    def test_missing_file(self, tmp_path):
        assert load_predictions(tmp_path / "nope.jsonl") == {}

    def test_tolerates_a_half_written_last_line(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            '{"question_id": "DOC/1", "parsed": {"answer": "1"}}\n{"question_id": "DOC/2"',
            encoding="utf-8",
        )
        records = load_predictions(path)
        assert list(records) == ["DOC/1"]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text('\n{"question_id": "DOC/1"}\n\n', encoding="utf-8")
        assert list(load_predictions(path)) == ["DOC/1"]


class TestScoring:
    def test_accuracy_counts_failures_in_the_denominator(self, tmp_path):
        examples = [example(1, 100), example(2, 100), example(3, 200)]
        path = tmp_path / "predictions.jsonl"
        predictions, _ = execute(
            system_answering("ANSWER: 100"), examples, path, budget=fresh_budget(tmp_path)
        )

        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        assert summary["n"] == 3
        assert summary["correct"] == 2
        assert summary["accuracy"] == round(2 / 3, 4)

    def test_unanswered_questions_are_left_out_but_counted_against(self, tmp_path):
        examples = [example(1, 100), example(2, 100)]
        path = tmp_path / "predictions.jsonl"
        predictions, _ = execute(
            system_answering("ANSWER: 100"), examples, path, budget=fresh_budget(tmp_path)
        )
        summary = summarise(score_all(examples[:1], predictions, tolerance=1e-3))
        assert summary["n"] == 1

    def test_call_errors_are_counted_not_dropped(self, tmp_path):
        examples = [example(1, 100)]
        path = tmp_path / "predictions.jsonl"
        predictions, _ = execute(
            DirectLLM(FailingProvider(ProviderRequestFailed("nope"))),
            examples,
            path,
            budget=fresh_budget(tmp_path),
        )
        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        assert summary["call_errors"] == 1
        assert summary["accuracy"] == 0.0

    def test_percent_equivalent_is_reported_next_to_strict(self, tmp_path):
        examples = [example(1, 0.935)]
        path = tmp_path / "predictions.jsonl"
        predictions, _ = execute(
            system_answering("ANSWER: 93.5"), examples, path, budget=fresh_budget(tmp_path)
        )
        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        assert summary["accuracy"] == 0.0
        assert summary["accuracy_percent_equivalent"] == 1.0

    def test_format_counts_expose_how_often_the_fallbacks_fire(self, tmp_path):
        examples = [example(1, 100), example(2, 100)]
        path = tmp_path / "predictions.jsonl"
        responses = iter(["ANSWER: 100", "the answer is 100"])
        predictions, _ = execute(
            DirectLLM(FakeLLM(responses=responses)),
            examples,
            path,
            budget=fresh_budget(tmp_path),
        )
        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        assert summary["answer_formats"]["marked"] == 1
        assert summary["answer_formats"]["last_number"] == 1

    def test_parse_failure_rate(self, tmp_path):
        examples = [example(1, 100)]
        path = tmp_path / "predictions.jsonl"
        predictions, _ = execute(
            system_answering("I cannot tell."), examples, path, budget=fresh_budget(tmp_path)
        )
        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        assert summary["parse_failures"] == 1
        assert summary["parse_failure_rate"] == 1.0

    def test_answers_cut_off_by_the_provider_are_counted(self, tmp_path):
        """A truncated answer reads as a wrong one, so the count has to be visible."""
        examples = [example(1, 100)]
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "system": "direct@fake",
                    "question_id": "DOC/1",
                    "parsed": {"answer": None, "format": "unparsed", "raw": "the total is 1"},
                    "finish_reason": "length",
                    "total_tokens": 4096,
                    "reasoning_tokens": 3980,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        summary = summarise(score_all(examples, load_predictions(path), tolerance=1e-3))

        assert summary["truncated"] == 1
        assert summary["tokens"]["reasoning"] == 3980
        assert summary["tokens"]["total"] == 4096

    def test_reference_kind_breakdown(self, tmp_path):
        examples = [example(1, 100), example(2, "yes", question="did it rise?")]
        path = tmp_path / "predictions.jsonl"
        provider = FakeLLM(responses=["ANSWER: 100", "ANSWER: yes"])
        predictions, _ = execute(DirectLLM(provider), examples, path, budget=fresh_budget(tmp_path))
        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        assert summary["by_reference_kind"]["number"]["accuracy"] == 1.0
        assert summary["by_reference_kind"]["text"]["accuracy"] == 1.0

    def test_empty_run_summarises_to_zero(self):
        assert summarise([]) == {"n": 0}


class TestReport:
    def test_render_mentions_the_headline_numbers(self, tmp_path):
        examples = [example(1, 100)]
        path = tmp_path / "predictions.jsonl"
        predictions, _ = execute(
            system_answering("ANSWER: 100"), examples, path, budget=fresh_budget(tmp_path)
        )
        summary = summarise(score_all(examples, predictions, tolerance=1e-3))
        config = {
            "system": "direct@fake",
            "provider": "fake",
            "model": "fake-model",
            "split": "dev.json",
            "prompt_version": "direct-v1",
            "tolerance": 1e-3,
        }
        text = render(summary, config, complete=True)
        assert "accuracy:" in text
        assert "percent-equivalent:" in text
        assert "direct-v1" in text

    def test_render_flags_an_incomplete_run(self, tmp_path):
        summary = {
            "n": 1,
            "accuracy": 0.0,
            "accuracy_percent_equivalent": 0.0,
            "parse_failures": 0,
            "parse_failure_rate": 0.0,
            "unexecuted": 0,
            "unanswered": 0,
            "unanswered_rate": 0.0,
            "call_errors": 0,
            "truncated": 0,
            "answer_formats": {},
            "outcomes": {},
            "latency_seconds": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "total": 0.0},
            "tokens": {"total": 0, "reasoning": 0, "mean_per_question": 0.0},
            "cost": {"total_usd": 0.0, "unpriced_questions": 0},
            "by_reference_kind": {},
        }
        config = {
            "system": "s",
            "provider": "p",
            "model": "m",
            "split": "dev.json",
            "prompt_version": "v",
            "tolerance": 1e-3,
        }
        assert "INCOMPLETE" in render(summary, config, complete=False)

    def test_render_handles_an_empty_run(self):
        assert render({"n": 0}, {}, complete=False) == "no questions were scored"


class TestWhichRetriever:
    """Which retriever a run used has to be readable from the run, not inferred.

    Runs A, D and E were recorded before there was a choice, so the default has to
    keep producing the same name they were recorded under.
    """

    @staticmethod
    def provider():
        return FakeLLM(responder=lambda messages: "ANSWER: 1")

    def test_the_default_retriever_is_not_in_the_name(self):
        assert ProgramLLM(self.provider()).name == "program+repair1@fake"

    def test_a_reranked_run_says_so_in_its_name(self):
        system = ProgramLLM(self.provider(), retriever=RERANKED)
        assert system.name == "program+reranked+repair1@fake"

    def test_the_retriever_reaches_the_orchestrator(self):
        system = ProgramLLM(self.provider(), retriever=RERANKED)
        assert system.orchestrator.retriever == RERANKED

    def test_the_run_config_records_it(self):
        system = ProgramLLM(self.provider(), retriever=RERANKED)
        args = SimpleNamespace(split="dev.json", tolerance=1e-3, limit=None)
        config = build_config(
            args=args,
            settings=get_settings(),
            provider=system.provider,
            system=system,
            examples=[],
        )
        assert config["retriever"] == RERANKED


class TestPercentile:
    def test_empty(self):
        assert percentile([], 0.5) == 0.0

    def test_single_value(self):
        assert percentile([3.0], 0.9) == 3.0

    def test_median_of_two(self):
        assert percentile([1.0, 3.0], 0.5) == 2.0

    def test_p95_of_a_run_of_values(self):
        assert percentile([float(i) for i in range(1, 101)], 0.95) == 95.05
