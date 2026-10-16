"""Replay FinQA's gold reasoning programs through the calculation engine.

This is the milestone test that needs no language model at all. Each example
comes with the program a human annotator wrote and the answer that program
produces when executed. If the engine is correct, replaying the gold programs
should reproduce the gold answers.

That makes it a clean check on two things at once: whether the engine executes
the DSL correctly, and whether this project understands the dataset's encoding.
Any gap here is a bug in the engine, not a modelling failure, so it has to be
closed before baselines are worth running.

    python -m evaluation.replay_gold_programs --split data/finqa/dev.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.services.calculations import (
    CalculationError,
    InvalidArgument,
    ProgramSyntaxError,
    UnresolvedReference,
    UnsupportedOperation,
    answers_match,
    execute_program,
    parse_program,
)
from app.services.dataset import FinQAExample, load_split, scan_quality

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "evaluation" / "reports"


@dataclass
class ExampleOutcome:
    id: str
    steps: int
    ok: bool
    predicted: str | None = None
    expected: str | None = None
    error: str | None = None
    error_kind: str | None = None


@dataclass
class ReplayReport:
    split: str
    source: str
    tolerance: str
    generated_at: str
    total: int = 0
    matched: int = 0
    errors: int = 0
    mismatched: int = 0
    by_step_count: dict[str, dict[str, int]] = field(default_factory=dict)
    error_kinds: dict[str, int] = field(default_factory=dict)
    outcomes: list[ExampleOutcome] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def render(self) -> str:
        lines = [
            f"split:      {self.split}",
            f"examples:   {self.total}",
            f"matched:    {self.matched} ({self.accuracy:.4f})",
            f"errors:     {self.errors}",
            f"mismatched: {self.mismatched}",
        ]
        if self.error_kinds:
            lines.append("error kinds:")
            for kind, count in sorted(self.error_kinds.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {kind:<24} {count}")
        if self.by_step_count:
            lines.append("by step count:")
            lines.append(f"  {'steps':<8}{'n':>6}{'matched':>9}{'accuracy':>10}")
            for steps in sorted(self.by_step_count, key=lambda s: int(s)):
                stats = self.by_step_count[steps]
                acc = stats["matched"] / stats["total"] if stats["total"] else 0.0
                lines.append(
                    f"  {steps:<8}{stats['total']:>6}{stats['matched']:>9}{acc:>10.4f}"
                )
        return "\n".join(lines)


def _step_count(example: FinQAExample) -> int:
    try:
        return len(parse_program(example.program))
    except CalculationError:
        return example.program.count(",") + 1


def _classify(exc: CalculationError) -> str:
    if isinstance(exc, ProgramSyntaxError):
        return "syntax"
    if isinstance(exc, UnsupportedOperation):
        return "unsupported_op"
    if isinstance(exc, UnresolvedReference):
        return "bad_reference"
    if isinstance(exc, InvalidArgument):
        return "bad_argument"
    return "calculation"


def replay(examples, *, tolerance="1e-3") -> ReplayReport:
    from decimal import Decimal

    tol = Decimal(tolerance)
    report = ReplayReport(
        split="<memory>",
        source="<memory>",
        tolerance=tolerance,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    by_steps: dict[str, dict[str, int]] = {}
    error_kinds: Counter[str] = Counter()

    for example in examples:
        steps = _step_count(example)
        bucket = by_steps.setdefault(str(steps), {"total": 0, "matched": 0})
        bucket["total"] += 1
        report.total += 1

        outcome = ExampleOutcome(id=example.id, steps=steps, ok=False)
        try:
            trace = execute_program(example.program, table=example.table)
        except CalculationError as exc:
            report.errors += 1
            kind = _classify(exc)
            error_kinds[kind] += 1
            outcome.error = str(exc)
            outcome.error_kind = kind
            report.outcomes.append(outcome)
            continue

        outcome.predicted = str(trace.result)
        outcome.expected = str(example.exe_ans)

        if answers_match(trace.result, example.exe_ans, tolerance=tol):
            report.matched += 1
            bucket["matched"] += 1
            outcome.ok = True
        else:
            report.mismatched += 1
            error_kinds["answer_mismatch"] += 1
            outcome.error_kind = "answer_mismatch"

        report.outcomes.append(outcome)

    report.by_step_count = by_steps
    report.error_kinds = dict(error_kinds)
    return report


def save(report: ReplayReport, out_root: Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    target = out_root / f"{stamp}_replay_{name}"
    target.mkdir(parents=True, exist_ok=True)

    (target / "metrics.json").write_text(
        json.dumps(
            {k: v for k, v in asdict(report).items() if k != "outcomes"},
            indent=2,
        ),
        encoding="utf-8",
    )
    with (target / "outcomes.jsonl").open("w", encoding="utf-8") as fh:
        for outcome in report.outcomes:
            fh.write(json.dumps(asdict(outcome)) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="path to a FinQA split json file")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N examples")
    parser.add_argument("--tolerance", default="1e-3")
    parser.add_argument("--out", default=str(DEFAULT_REPORTS))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.split)
    examples = load_split(path)
    if args.limit:
        examples = examples[: args.limit]

    quality = scan_quality(examples, source=str(path))
    print(quality.render())
    print()

    report = replay(examples, tolerance=args.tolerance)
    report.split = str(path)
    report.source = str(path)
    print(report.render())

    if not args.no_save:
        target = save(report, Path(args.out), path.stem)
        print(f"\nsaved: {target}")

    failed = [
        o
        for o in report.outcomes
        if not o.ok
    ]
    if failed:
        print(f"\nfirst few failures (of {len(failed)}):")
        for outcome in failed[:5]:
            detail = outcome.error or (
                f"expected {outcome.expected}, got {outcome.predicted}"
            )
            print(f"  {outcome.id}: {detail}")

    return 0 if report.mismatched == 0 and report.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
