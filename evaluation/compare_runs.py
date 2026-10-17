"""Compare two runs over the questions they have in common.

Two runs are only comparable where they answered the same questions. Putting two
accuracy figures side by side and hoping they came from the same place is how a
comparison goes wrong quietly, so this scores both over the intersection, says how
big the intersection is, and shows how the pairs divide.

    python -m evaluation.compare_runs <run_dir> <run_dir>

The last section is the useful one. "Only the first got it" and "only the second
got it" are counts, but they are also a list of question ids, and reading a handful
of them is usually the whole answer to "which one is better, and at what".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.dataset import FinQAExample, load_split

from evaluation.run_eval import Outcome, load_predictions, score_all
from evaluation.scoring import DEFAULT_TOLERANCE

EXAMPLES_SHOWN = 6


def read_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def describe(run_dir: Path, config: dict, scored: int) -> str:
    system = config.get("system", run_dir.name)
    prompt = config.get("prompt_version", "?")
    return f"{system} [{prompt}] {scored} scored"


def paired(
    left: list[Outcome], right: list[Outcome]
) -> dict[str, list[str]]:
    """Split the shared questions four ways."""
    right_by_id = {outcome.question_id: outcome for outcome in right}
    buckets: dict[str, list[str]] = {"both": [], "only_left": [], "only_right": [], "neither": []}

    for outcome in left:
        other = right_by_id[outcome.question_id]
        left_ok, right_ok = outcome.correct, other.correct
        if left_ok and right_ok:
            buckets["both"].append(outcome.question_id)
        elif left_ok:
            buckets["only_left"].append(outcome.question_id)
        elif right_ok:
            buckets["only_right"].append(outcome.question_id)
        else:
            buckets["neither"].append(outcome.question_id)
    return buckets


def render(
    left_dir: Path,
    right_dir: Path,
    left_config: dict,
    right_config: dict,
    left: list[Outcome],
    right: list[Outcome],
) -> str:
    shared = len(left)
    if shared == 0:
        return "these two runs have no questions in common, so there is nothing to compare"

    left_strict = sum(1 for o in left if o.correct) / shared
    right_strict = sum(1 for o in right if o.correct) / shared
    left_pct = sum(1 for o in left if o.correct or o.percent_equivalent) / shared
    right_pct = sum(1 for o in right if o.correct or o.percent_equivalent) / shared

    buckets = paired(left, right)

    lines = [
        f"first:  {describe(left_dir, left_config, shared)}",
        f"second: {describe(right_dir, right_config, shared)}",
        f"shared: {shared} questions, scored for both",
        "",
        f"{'':<18}{'first':>9}{'second':>9}",
        f"{'strict':<18}{left_strict:>9.4f}{right_strict:>9.4f}",
        f"{'percent-equivalent':<18}{left_pct:>9.4f}{right_pct:>9.4f}",
        "",
        "how the shared questions divide",
        f"  both got it      {len(buckets['both']):>5}",
        f"  only the first   {len(buckets['only_left']):>5}",
        f"  only the second  {len(buckets['only_right']):>5}",
        f"  neither          {len(buckets['neither']):>5}",
    ]

    for name, key in (("only the first", "only_left"), ("only the second", "only_right")):
        ids = buckets[key]
        if ids:
            lines.append("")
            lines.append(f"{name} answered, e.g.")
            lines.extend(f"  {question_id}" for question_id in ids[:EXAMPLES_SHOWN])
            if len(ids) > EXAMPLES_SHOWN:
                lines.append(f"  ... and {len(ids) - EXAMPLES_SHOWN} more")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Two positionals rather than one with nargs=2 and a tuple metavar. argparse
    # cannot format the tuple form for a positional, so that spelling works
    # everywhere except --help, which is where people meet it first.
    parser.add_argument("first", metavar="FIRST", help="a run directory")
    parser.add_argument("second", metavar="SECOND", help="the run directory to compare it against")
    parser.add_argument("--split", default="data/finqa/dev.json", help="the split both runs used")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples: list[FinQAExample] = load_split(args.split)

    prepared = []
    for raw in (args.first, args.second):
        run_dir = Path(raw)
        records = load_predictions(run_dir / "predictions.jsonl")
        prepared.append((run_dir, read_config(run_dir), records))

    # Only the questions both runs have a prediction for, in split order.
    shared = [
        example
        for example in examples
        if all(example.id in records for _, _, records in prepared)
    ]
    scored = [
        score_all(shared, records, tolerance=args.tolerance) for _, _, records in prepared
    ]

    (left_dir, left_config, _), (right_dir, right_config, _) = prepared
    print(render(left_dir, right_dir, left_config, right_config, scored[0], scored[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
