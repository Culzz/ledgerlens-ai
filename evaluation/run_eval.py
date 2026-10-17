"""Run a system against a FinQA split and score it.

    python -m evaluation.run_eval --split data/finqa/dev.json --system direct

Results land in a timestamped directory under ``evaluation/reports/`` holding the
config, the aggregates and one line per question. Every number quoted anywhere
comes from a directory like that.

**Resuming.** The predictions file is append-only and the runner skips any
question already in it. A run stopped by a daily quota is continued by pointing
``--resume`` at the same directory; nothing is recomputed and nothing is spent
twice. That is the whole reason the file is written one question at a time rather
than in one go at the end.

**Failures.** A question whose call fails is recorded with the error and counted
against accuracy, because a system that cannot answer has not answered. Failures
are never dropped from the denominator. Rate limits, an exhausted budget and a
missing key stop the run instead, since they will repeat on every remaining
question.

The model is whatever ``LLM_PROVIDER`` names. With the default ``fake`` provider
this runs offline and spends nothing, which is how the harness is tested.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.services.dataset import FinQAExample, load_split
from app.services.llm import BudgetExceeded, BudgetLedger, LLMProvider, build_provider
from app.services.reasoning import FORMAT_UNEXECUTED, FORMAT_UNPARSED, ParsedAnswer, to_decimal

from evaluation.baselines import DirectLLM, ProgramLLM, System
from evaluation.scoring import DEFAULT_TOLERANCE, Judgement, judge

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "evaluation" / "reports"
DEFAULT_SYSTEM = "direct"
HALTED_EXIT_CODE = 3

SYSTEMS = {"direct": DirectLLM, "program": ProgramLLM}


@dataclass
class Outcome:
    """One question, its prediction, and how it scored."""

    question_id: str
    reference: str
    predicted: str | None
    correct: bool
    percent_equivalent: bool
    outcome: str
    answer_format: str
    reasoning: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    attempts: int = 1
    call_error: str | None = None
    finish_reason: str | None = None

    @property
    def billed_tokens(self) -> int:
        """The provider's own total when it gave one, otherwise the visible pair.

        Reasoning tokens are billed but do not appear in the message, so trusting
        prompt + completion alone understates a run on a model that thinks.
        """
        return self.total_tokens or (self.prompt_tokens + self.completion_tokens)


def percentile(values: Iterable[float], fraction: float) -> float:
    """Linear-interpolated percentile. Small inputs included, hence no numpy."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarise(outcomes: list[Outcome]) -> dict[str, Any]:
    """Aggregate a run. Accuracy denominators always include the failures."""
    n = len(outcomes)
    if not n:
        return {"n": 0}

    correct = sum(1 for o in outcomes if o.correct)
    percent_equivalent = sum(1 for o in outcomes if o.correct or o.percent_equivalent)
    # Producing no answer covers two different failures, and they are counted
    # apart: a reply that could not be read, and a reply that read fine but whose
    # program would not run. Blending them would hide the second, which is the
    # characteristic failure of a program-shaped system.
    parse_failures = sum(
        1 for o in outcomes if not o.call_error and o.answer_format == FORMAT_UNPARSED
    )
    unexecuted = sum(
        1 for o in outcomes if not o.call_error and o.answer_format == FORMAT_UNEXECUTED
    )
    unanswered = sum(
        1
        for o in outcomes
        if o.call_error or o.answer_format in (FORMAT_UNPARSED, FORMAT_UNEXECUTED)
    )
    latencies = [o.latency_seconds for o in outcomes if not o.call_error]
    priced = [o for o in outcomes if o.cost_usd is not None]

    by_outcome: dict[str, int] = {}
    for outcome in outcomes:
        by_outcome[outcome.outcome] = by_outcome.get(outcome.outcome, 0) + 1

    by_format: dict[str, int] = {}
    for outcome in outcomes:
        key = "call_error" if outcome.call_error else outcome.answer_format
        by_format[key] = by_format.get(key, 0) + 1

    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "accuracy_percent_equivalent": round(percent_equivalent / n, 4),
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / n, 4),
        "unexecuted": unexecuted,
        "unanswered": unanswered,
        "unanswered_rate": round(unanswered / n, 4),
        "call_errors": sum(1 for o in outcomes if o.call_error),
        "truncated": sum(1 for o in outcomes if o.finish_reason == "length"),
        "outcomes": dict(sorted(by_outcome.items())),
        "answer_formats": dict(sorted(by_format.items())),
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.5), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "total": round(sum(latencies), 4),
        },
        "tokens": {
            "prompt": sum(o.prompt_tokens for o in outcomes),
            "completion": sum(o.completion_tokens for o in outcomes),
            "reasoning": sum(o.reasoning_tokens for o in outcomes),
            "total": sum(o.billed_tokens for o in outcomes),
            "mean_per_question": round(sum(o.billed_tokens for o in outcomes) / n, 1),
        },
        "cost": {
            "total_usd": round(sum(o.cost_usd for o in priced), 6),
            "priced_questions": len(priced),
            "unpriced_questions": n - len(priced),
        },
        "by_reference_kind": _by_reference_kind(outcomes),
    }


def _by_reference_kind(outcomes: list[Outcome]) -> dict[str, dict[str, Any]]:
    """Break accuracy down by whether the reference was a number or a yes/no."""
    groups: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        kind = "number" if to_decimal(outcome.reference) is not None else "text"
        groups.setdefault(kind, []).append(outcome)
    return {
        kind: {
            "n": len(rows),
            "accuracy": round(sum(1 for r in rows if r.correct) / len(rows), 4),
        }
        for kind, rows in sorted(groups.items())
    }


def render(summary: dict[str, Any], config: dict[str, Any], *, complete: bool) -> str:
    if not summary.get("n"):
        return "no questions were scored"

    lines = [
        f"system:      {config['system']}",
        f"provider:    {config['provider']} ({config['model'] or 'no model set'})",
        f"split:       {config['split']}",
        f"prompt:      {config['prompt_version']}",
        f"question:    {summary['n']} scored"
        + ("" if complete else " (INCOMPLETE - resume to finish)"),
        "",
        f"accuracy:            {summary['accuracy']:.4f}   (strict, tolerance {config['tolerance']})",
        f"percent-equivalent:  {summary['accuracy_percent_equivalent']:.4f}   (also accepts x100)",
        f"no answer:           {summary['unanswered']} ({summary['unanswered_rate']:.4f})",
        f"  unreadable reply:  {summary['parse_failures']}",
        f"  unexecutable:      {summary['unexecuted']}",
        f"call errors:         {summary['call_errors']}",
        f"cut off by provider: {summary['truncated']}",
        "",
        "where the answer came from",
    ]
    for name, count in summary["answer_formats"].items():
        lines.append(f"  {name:<16}{count:>6}")

    lines.append("")
    lines.append("outcomes")
    for name, count in summary["outcomes"].items():
        lines.append(f"  {name:<32}{count:>6}")

    latency = summary["latency_seconds"]
    tokens = summary["tokens"]
    cost = summary["cost"]
    unpriced = cost["unpriced_questions"]
    lines.extend(
        [
            "",
            f"latency:  mean {latency['mean']:.3f}s  p50 {latency['p50']:.3f}s  "
            + f"p95 {latency['p95']:.3f}s  total {latency['total']:.1f}s",
            f"tokens:   {tokens['total']:,} total, {tokens['mean_per_question']} per question"
            + (
                f"  ({tokens['reasoning']:,} of them internal reasoning)"
                if tokens["reasoning"]
                else ""
            ),
            f"cost:     ${cost['total_usd']:.4f}"
            + (f"  ({unpriced} questions had no known price)" if unpriced else ""),
        ]
    )

    if summary["by_reference_kind"]:
        lines.append("")
        lines.append("by reference answer kind")
        for kind, row in summary["by_reference_kind"].items():
            lines.append(f"  {kind:<16}{row['n']:>6}  accuracy {row['accuracy']:.4f}")

    return "\n".join(lines)


def new_run_dir(out_root: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    return out_root / f"{stamp}_{name}"


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Read the append-only predictions file, tolerating a half-written last line."""
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # Only realistic cause is a kill mid-write. Everything before it
                # is intact, so keep what is there and redo that one question.
                continue
            question_id = record.get("question_id")
            if question_id:
                records[question_id] = record
    return records


def is_answered(record: dict[str, Any] | None) -> bool:
    """Whether a question has a usable answer already on disk.

    A record whose call failed is not an answer. Treating it as one would freeze
    a transient 429 or 503 into a permanent wrong answer, and on a metered free
    tier that is a day of quota spent on a provider hiccup.
    """
    return bool(record) and record.get("call_error") is None


def execute(
    system: System,
    examples: list[FinQAExample],
    predictions_path: Path,
    *,
    budget: BudgetLedger,
    progress_every: int = 0,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Answer every example not already answered, appending as it goes.

    Returns everything known afterwards plus, when the run stopped early, why.
    Written as a function rather than inlined into ``main`` so the resume and
    halt behaviour can be tested without a network or an API key.
    """
    predictions = load_predictions(predictions_path)
    remaining = [e for e in examples if not is_answered(predictions.get(e.id))]
    if not remaining:
        return predictions, None

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    halted: str | None = None
    done = 0

    with predictions_path.open("a", encoding="utf-8") as fh:
        for example in remaining:
            try:
                budget.check()
            except BudgetExceeded as exc:
                halted = str(exc)
                break

            try:
                prediction = system.answer(example)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see below
                # RUN_HALTING_ERRORS are re-raised by the system rather than
                # turned into a prediction. Anything arriving here is systemic
                # enough that continuing would just repeat it 800 times, so stop
                # and hand back the reason.
                halted = f"{type(exc).__name__}: {exc}"
                break

            record = prediction.to_record()
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            predictions[example.id] = record
            done += 1

            if prediction.call_error is None:
                budget.record(
                    cost_usd=prediction.cost_usd,
                    prompt_tokens=prediction.prompt_tokens,
                    completion_tokens=prediction.completion_tokens,
                )

            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(remaining)} this session")

    return predictions, halted


def score_all(
    examples: list[FinQAExample],
    predictions: dict[str, dict[str, Any]],
    *,
    tolerance: float,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for example in examples:
        record = predictions.get(example.id)
        if record is None:
            continue

        parsed = ParsedAnswer(
            answer=(record.get("parsed") or {}).get("answer"),
            format=(record.get("parsed") or {}).get("format", "unparsed"),
            raw=(record.get("parsed") or {}).get("raw", ""),
        )
        verdict: Judgement = judge(
            parsed, example.exe_ans, tolerance=tolerance, call_error=record.get("call_error")
        )
        outcomes.append(
            Outcome(
                question_id=example.id,
                reference=verdict.reference,
                predicted=verdict.predicted,
                correct=verdict.correct,
                percent_equivalent=verdict.percent_equivalent,
                outcome=verdict.outcome,
                answer_format=parsed.format,
                reasoning=(record.get("parsed") or {}).get("reasoning", ""),
                latency_seconds=record.get("latency_seconds", 0.0),
                prompt_tokens=record.get("prompt_tokens", 0),
                completion_tokens=record.get("completion_tokens", 0),
                total_tokens=record.get("total_tokens", 0),
                reasoning_tokens=record.get("reasoning_tokens", 0),
                cost_usd=record.get("cost_usd"),
                attempts=record.get("attempts", 1),
                call_error=record.get("call_error"),
                finish_reason=record.get("finish_reason"),
            )
        )
    return outcomes


def build_config(
    *,
    args: argparse.Namespace,
    settings: Settings,
    provider: LLMProvider,
    system: System,
    examples: list[FinQAExample],
) -> dict[str, Any]:
    return {
        "system": system.name,
        "provider": provider.name,
        "model": provider.model,
        "split": str(args.split),
        "split_records": len(examples),
        "prompt_version": system.prompt_version,
        "tolerance": args.tolerance,
        "limit": args.limit,
        "free_tier": getattr(provider, "free_tier", None),
        "max_retries": getattr(provider, "max_retries", None),
        "temperature": getattr(system, "temperature", None),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "budget_limit_usd": settings.llm_monthly_budget_usd,
    }


def save(run_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="path to a FinQA split json file")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, choices=sorted(SYSTEMS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--out", default=str(DEFAULT_REPORTS), help="where to put a new run")
    parser.add_argument("--resume", default=None, help="an existing run directory to continue")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    provider = build_provider(settings)
    system: System = SYSTEMS[args.system](provider)

    path = Path(args.split)
    examples = load_split(path)
    if args.limit:
        examples = examples[: args.limit]

    run_dir = (
        Path(args.resume)
        if args.resume
        else new_run_dir(Path(args.out), f"{system.name}_{path.stem}")
    )
    predictions_path = run_dir / "predictions.jsonl"

    print(f"provider: {provider.name} ({provider.model or 'no model set'})")
    print(f"system:   {system.name} [{system.prompt_version}]")
    print(f"questions: {len(examples)} in split")
    if predictions_path.exists():
        print(f"resuming: {run_dir}")

    budget = BudgetLedger.open(
        settings.llm_budget_ledger_path, limit_usd=settings.llm_monthly_budget_usd
    )
    predictions, halted = execute(
        system, examples, predictions_path, budget=budget, progress_every=args.progress_every
    )

    outcomes = score_all(examples, predictions, tolerance=args.tolerance)
    summary = summarise(outcomes)
    summary["budget"] = budget.summary()
    complete = len(predictions) >= len(examples)
    config = build_config(
        args=args, settings=settings, provider=provider, system=system, examples=examples
    )

    print()
    print(render(summary, config, complete=complete))

    if not args.no_save:
        save(run_dir, config, summary)
        print(f"\nrun directory: {run_dir}")

    if halted:
        print(f"\nstopped early: {halted}")
        print(
            f'resume with: python -m evaluation.run_eval --resume "{run_dir}" '
            f'--split "{args.split}" --system {args.system}'
        )
        return HALTED_EXIT_CODE

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
