"""Replay a finished run through the grounding check, without any model calls.

The check is a new way to refuse an answer, and a gate that refuses correct answers
is worse than no gate. The one number that decides whether it is safe to leave on
is therefore: of the answers it objects to, how many were right?

That is answerable offline. Every run keeps the raw reply for each question, so the
check can be run over replies that already exist, for nothing, without spending a
second of anyone's quota or adding a single token to a measured run.

It also collects the figures the check objected to, most common first. The list of
things it calls invented is the only evidence for whether the constants it carves
out are the right ones - if 12 keeps turning up, the constant set is wrong, not the
model.

Usage: python -m evaluation.audit_grounding <run-directory> [--examples N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.services.dataset import load_split
from app.services.llm import FakeLLM
from app.services.reasoning import (
    FORMAT_JSON,
    FORMAT_UNEXECUTED,
    PARTLY_GROUNDED,
    Document,
    Orchestrator,
    ParsedAnswer,
)

from evaluation.scoring import judge


@dataclass(frozen=True)
class Replay:
    """One stored reply, run through the check again."""

    question_id: str
    question: str
    program: str | None
    before: str | None
    after: str | None
    reference: str
    was_correct: bool
    still_correct: bool
    verdict: str
    unsourced: tuple[str, ...]
    refused: bool

    @property
    def reproduced(self) -> bool:
        """Whether replaying the reply produced the answer the run recorded."""
        return self.before == self.after


def replay(run_dir: Path, split_path: Path) -> list[Replay]:
    """Run every stored reply in a run through the check again."""
    predictions: dict[str, dict] = {}
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            predictions[record["question_id"]] = record

    # No provider is used: resolve never calls one. The orchestrator is here for its
    # retrieval and its gate, both of which are deterministic.
    orchestrator = Orchestrator(FakeLLM())

    replayed: list[Replay] = []
    for example in load_split(split_path):
        record = predictions.get(example.id)
        if record is None:
            continue
        parsed = record.get("parsed") or {}
        raw = parsed.get("raw") or ""
        if not raw.strip():
            # The model said nothing, so there is nothing to check. Counting these as
            # refusals would credit the gate with a failure that was already counted.
            continue

        document = Document.from_example(example)
        retrieval = orchestrator.retrieve(document, example.question)
        grounded = orchestrator.resolve(example.question, raw, document, retrieval.by_label)

        before = judge(
            ParsedAnswer(
                answer=parsed.get("answer"), format=parsed.get("format", "unparsed"), raw=raw
            ),
            example.exe_ans,
            call_error=record.get("call_error"),
        )
        after = judge(
            ParsedAnswer(
                answer=grounded.answer,
                format=FORMAT_JSON if grounded.ok else FORMAT_UNEXECUTED,
                raw=raw,
            ),
            example.exe_ans,
            call_error=record.get("call_error"),
        )

        replayed.append(
            Replay(
                question_id=example.id,
                question=example.question,
                program=grounded.program,
                before=before.predicted,
                after=after.predicted,
                reference=before.reference,
                was_correct=before.correct,
                still_correct=after.correct,
                verdict=grounded.grounding.verdict if grounded.grounding else "no program",
                unsourced=grounded.grounding.unsourced if grounded.grounding else (),
                refused=(grounded.failure or "").startswith("UngroundedAnswer"),
            )
        )
    return replayed


def render(replays: list[Replay], examples: int) -> str:
    lines: list[str] = []
    lines.append(f"{len(replays)} stored replies replayed")
    lines.append("")

    drifted = [r for r in replays if not r.reproduced]
    lines.append(f"answers that changed on replay   {len(drifted)}")
    if drifted:
        lines.append("  (should be zero: the engine is deterministic, so a change here")
        lines.append("   means the recorded run and this code disagree about the answer)")
        for replay in drifted[:examples]:
            lines.append(f"    {replay.question_id}: {replay.before} -> {replay.after}")

    lines.append("")
    lines.append("verdicts")
    for name, count in sorted(Counter(r.verdict for r in replays).items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<20}{count:>5}")

    refused = [r for r in replays if r.refused]
    right = [r for r in refused if r.was_correct]
    wrong = [r for r in refused if not r.was_correct]
    correct_before = sum(1 for r in replays if r.was_correct)
    correct_after = sum(1 for r in replays if r.still_correct)

    lines.append("")
    lines.append("what the gate does")
    lines.append(f"  correct answers, gate off       {correct_before} of {len(replays)}")
    lines.append(f"  correct answers, gate on        {correct_after} of {len(replays)}")
    lines.append(f"  answers refused                 {len(refused)}")
    lines.append(f"    of those right                {len(right)}")
    lines.append(f"    of those already wrong        {len(wrong)}")

    if right:
        lines.append("")
        lines.append(f"correct answers the gate costs ({len(right)})")
        for replay in right[:examples]:
            lines.append(f"  {replay.question_id}")
            lines.append(f"    {replay.question[:110]}")
            lines.append(f"    program: {replay.program}")
            lines.append(f"    reference {replay.reference}   it answered {replay.before}")
            lines.append(f"    figures it objected to: {', '.join(replay.unsourced) or 'none'}")

    objectionable = Counter(figure for replay in refused for figure in replay.unsourced)
    if objectionable:
        lines.append("")
        lines.append("figures called invented, most common first")
        for figure, count in objectionable.most_common(20):
            lines.append(f"  {figure:<12}{count:>5}")

    partly = [r for r in replays if r.verdict == PARTLY_GROUNDED]
    lines.append("")
    lines.append(f"kept anyway, resting on a figure the report does not state  {len(partly)}")
    lines.append(f"  of those right               {sum(1 for r in partly if r.was_correct)}")
    for replay in [r for r in partly if r.was_correct][:examples]:
        lines.append(f"    {replay.question_id}  {', '.join(replay.unsourced)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="a run directory containing predictions.jsonl")
    parser.add_argument("--split", default="data/finqa/dev.json")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args(argv)

    replays = replay(Path(args.run), Path(args.split))
    if not replays:
        print(f"no replies to replay in {args.run} for {args.split}")
        return 1

    print(render(replays, args.examples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
