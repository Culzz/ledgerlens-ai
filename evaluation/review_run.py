"""Read a finished run and look at what the wrong answers actually are.

The strict and percent-equivalent counts are both real numbers, and the gap
between them is only worth trusting if the cases behind it are. This prints
them: how many wrong answers are a scale mismatch, how many are a correctly
computed value rounded, and how many are simply wrong - with the examples.

The point is not to move the headline, which stays FinQA's own strict rule so it
can sit next to published numbers. The point is that "wrong" on its own hides
three different failures, and a number nobody can take apart is a number nobody
should believe.

Usage: python -m evaluation.review_run <run-directory> [--examples N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.services.dataset import load_split
from app.services.reasoning import ParsedAnswer, to_decimal

from evaluation.scoring import judge

# Where a near miss stops being a rounding difference and starts being a wrong
# answer. One percent is generous for a figure someone would act on, and the
# buckets are printed either way so the boundary can be argued with.
NEAR_MISS = 0.01
MARGINAL = 0.05


@dataclass(frozen=True)
class Case:
    """One scored question, with everything needed to argue about it."""

    question_id: str
    question: str
    reference: str
    predicted: str | None
    outcome: str
    correct: bool
    percent_equivalent: bool

    @property
    def relative_error(self) -> float | None:
        """How far off, as a fraction of the reference. ``None`` if not comparable."""
        predicted = to_decimal(self.predicted or "")
        reference = to_decimal(self.reference)
        if predicted is None or reference is None or reference == 0:
            return None
        return abs(predicted - reference) / abs(reference)


def collect(run_dir: Path, split_path: Path) -> list[Case]:
    """Score every question in a run against its split."""
    predictions: dict[str, dict] = {}
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            predictions[record["question_id"]] = record

    cases: list[Case] = []
    for example in load_split(split_path):
        record = predictions.get(example.id)
        if record is None:
            continue
        parsed = record.get("parsed") or {}
        verdict = judge(
            ParsedAnswer(
                answer=parsed.get("answer"),
                format=parsed.get("format", "unparsed"),
                raw=parsed.get("raw", ""),
            ),
            example.exe_ans,
            call_error=record.get("call_error"),
        )
        cases.append(
            Case(
                question_id=example.id,
                question=example.question,
                reference=verdict.reference,
                predicted=verdict.predicted,
                outcome=verdict.outcome,
                correct=verdict.correct,
                percent_equivalent=verdict.percent_equivalent,
            )
        )
    return cases


def decompose(cases: list[Case]) -> dict[str, int]:
    """Count the wrong answers by how wrong they are.

    A wildly wrong number and a number that differs in the third decimal both
    report as ``wrong_number``, and they say completely different things about
    the system.
    """
    counts: Counter[str] = Counter()
    for case in cases:
        if case.correct:
            counts["correct"] += 1
        elif case.percent_equivalent:
            counts["scale_mismatch"] += 1
        elif case.outcome != "wrong_number":
            counts[case.outcome] += 1
        else:
            error = case.relative_error
            if error is None:
                counts["wrong_not_comparable"] += 1
            elif error <= NEAR_MISS:
                counts["near_miss"] += 1
            elif error <= MARGINAL:
                counts["marginal"] += 1
            else:
                counts["wrong"] += 1
    return dict(counts)


def render(cases: list[Case], examples: int) -> str:
    lines: list[str] = []
    counts = decompose(cases)
    total = len(cases)
    strict = sum(1 for case in cases if case.correct)
    percent = sum(1 for case in cases if case.percent_equivalent)

    lines.append(f"{total} questions scored")
    lines.append("")
    lines.append(f"strict                {strict:>4}  {strict / total:.4f}")
    lines.append(f"percent-equivalent    {percent:>4}  {percent / total:.4f}")
    lines.append("")
    lines.append("what the rest are")
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        lines.append(f"  {name:<22}{count:>4}")

    for title, key in (
        ("a percentage answered as a percentage", "scale_mismatch"),
        ("correctly computed, then rounded", "near_miss"),
    ):
        picked = [case for case in cases if _bucket(case) == key]
        if not picked:
            continue
        lines.append("")
        lines.append(f"{title} ({len(picked)})")
        for case in picked[:examples]:
            error = case.relative_error
            shown = f"{error:.3%}" if error is not None else "n/a"
            lines.append(f"  {case.question_id}  off by {shown}")
            lines.append(f"    {case.question[:110]}")
            lines.append(f"    reference {case.reference}   answered {case.predicted}")

    return "\n".join(lines)


def _bucket(case: Case) -> str:
    """Which bucket a single wrong case falls in, matching :func:`decompose`."""
    if case.correct:
        return "correct"
    if case.percent_equivalent:
        return "scale_mismatch"
    if case.outcome != "wrong_number":
        return case.outcome
    error = case.relative_error
    if error is None:
        return "wrong_not_comparable"
    if error <= NEAR_MISS:
        return "near_miss"
    return "marginal" if error <= MARGINAL else "wrong"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="a run directory containing predictions.jsonl")
    parser.add_argument("--split", default="data/finqa/dev.json")
    parser.add_argument("--examples", type=int, default=5, help="how many per bucket to print")
    args = parser.parse_args(argv)

    cases = collect(Path(args.run), Path(args.split))
    if not cases:
        print(f"no scored questions in {args.run} for {args.split}")
        return 1

    print(render(cases, examples=args.examples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
