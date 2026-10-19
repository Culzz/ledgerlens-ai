"""Measure the ceiling that retrieval places on everything downstream.

The answer to a question comes from one document. Retrieval decides which rows
and narrative lines get handed to the reader. If the annotated evidence is not in
what was retrieved, no model can answer correctly, however good it is.

So this asks one narrow question: for each question, does the retriever put the
annotated evidence inside the top k? That number is an upper bound on end-to-end
accuracy, and it costs nothing to compute, because no model is involved.

    python -m evaluation.retrieval_ceiling --split data/finqa/dev.json

Three things worth knowing when reading the output.

* Ground truth is ``ann_table_rows`` and ``ann_text_rows``, the annotation
  FinQA's own retriever was trained against. ``gold_inds`` is a separate
  annotation and disagrees on a minority of records. That count is reported
  rather than hidden.
* ``complete@k`` is the ceiling. It asks whether *all* the evidence a question
  needs is in the top k, which is what a reader given only those k units has.
* ``complete@k`` is capped by arithmetic before retrieval even starts: a question
  needing a table row and a narrative line can never be satisfied by k=1. The
  count of questions in that position is printed with the table, so a low
  low-k column is not misread as a retrieval failure.
* Retrieval is measured inside the question's own document, not across the whole
  corpus, because that is how the documents are actually used.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.services.dataset import FinQAExample, load_split
from app.services.retrieval import (
    DEFAULT_POOL,
    DEFAULT_WEIGHTS,
    FEATURES,
    BM25Index,
    Candidate,
    EvidenceUnit,
    Weights,
    answerable_units,
    build_units,
    evidence_kinds,
    gold_unit_ids,
    rerank,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "evaluation" / "reports"
DEFAULT_KS = (1, 3, 5, 10)

KIND_LABELS = {
    ("table",): "table only",
    ("text",): "text only",
    ("table", "text"): "table + text",
}


@dataclass
class QuestionResult:
    """What retrieval did for one question."""

    id: str
    gold_units: int
    document_units: int
    needs: str
    first_hit_rank: int | None
    hit_at: dict[str, bool] = field(default_factory=dict)
    complete_at: dict[str, bool] = field(default_factory=dict)
    recall_at: dict[str, float] = field(default_factory=dict)


@dataclass
class CeilingReport:
    split: str
    retriever: str
    generated_at: str
    ks: tuple[int, ...]
    records: int = 0
    corpus_units: int = 0
    evaluated: int = 0
    skipped_no_annotation: int = 0
    skipped_no_units: int = 0
    gold_inds_disagreements: int = 0
    # What the retriever was allowed to do. Recorded with the numbers because a
    # ceiling without its settings is not comparable to anything.
    pool: int = 0
    answerable_only: bool = False
    reranked: bool = False
    weights: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)
    by_evidence: dict = field(default_factory=dict)


def rate(flags, total: int) -> float:
    return round(sum(flags) / total, 4) if total else 0.0


def summarise(results: list[QuestionResult], ks: tuple[int, ...]) -> dict:
    """Aggregate one group of questions."""
    n = len(results)
    summary: dict = {
        "n": n,
        "hit@k": {},
        "complete@k": {},
        "recall@k": {},
        "gold_exceeds_k": {},
    }

    for k in ks:
        key = str(k)
        summary["hit@k"][key] = rate([r.hit_at[key] for r in results], n)
        summary["complete@k"][key] = rate([r.complete_at[key] for r in results], n)
        summary["recall@k"][key] = (
            round(sum(r.recall_at[key] for r in results) / n, 4) if n else 0.0
        )
        # Questions whose evidence simply cannot fit in k slots. These cap
        # complete@k from below, and have nothing to do with ranking quality.
        summary["gold_exceeds_k"][key] = sum(1 for r in results if r.gold_units > k)

    reciprocal = [1 / r.first_hit_rank for r in results if r.first_hit_rank]
    summary["mrr"] = round(sum(reciprocal) / n, 4) if n else 0.0
    summary["mean_gold_units"] = round(sum(r.gold_units for r in results) / n, 3) if n else 0.0
    return summary


def gold_inds_disagreements(examples: list[FinQAExample]) -> int:
    """Records where the two evidence annotations describe different table rows."""
    count = 0
    for example in examples:
        from_gold_inds = {
            int(key.split("_")[1]) for key in example.gold_inds if key.startswith("table_")
        }
        if from_gold_inds != set(example.ann_table_rows):
            count += 1
    return count


def load_weights(path: str | None) -> Weights:
    """The reranker's weights: the module's tuned defaults, or a JSON override.

    An override has to be a whole weight vector, not a patch: a run's settings are
    recorded with its numbers, and a file that set two of six weights would describe
    a retriever nobody could reproduce from the record.
    """
    if not path:
        return DEFAULT_WEIGHTS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [name for name in FEATURES if name not in raw]
    if missing:
        raise SystemExit(f"{path} is missing weights for: {', '.join(missing)}")
    return Weights(**{name: float(raw[name]) for name in FEATURES})


def retriever_name(args, weights: Weights) -> str:
    """What was run, as one line, for the artifact to carry."""
    name = f"BM25(k1={args.k1}, b={args.b})"
    if args.answerable:
        name += " + answerable only"
    if args.rerank:
        shown = " ".join(
            f"{feature}={value:g}" for feature, value in zip(FEATURES, weights.vector()) if value
        )
        name += f" + rerank[{shown}]"
    return name


def render(report: CeilingReport) -> str:
    ks = report.ks
    skipped = f"{report.skipped_no_annotation} with no annotation"
    if report.skipped_no_units:
        skipped += f", {report.skipped_no_units} with no units"

    lines = [
        f"retriever:      {report.retriever}",
        f"split:          {report.split}",
        f"records:        {report.records}",
        f"corpus:         {report.corpus_units:,} units",
        f"evaluated:      {report.evaluated} questions",
        f"skipped:        {skipped}",
        f"gold_inds:      {report.gold_inds_disagreements} records disagree with ann_*_rows",
        f"gold units:     {report.overall.get('mean_gold_units', 0)} per question",
        f"pool:           {report.pool} units ranked per question",
    ]
    if report.reranked:
        lines.append(f"weights:        {report.weights}")
    if report.answerable_only:
        lines.append("candidates:     answerable units only (table header rows excluded)")
    lines += [
        "",
        "                " + "".join(f"{'k=' + str(k):>9}" for k in ks),
    ]

    for label, key in (("recall", "recall@k"), ("hit", "hit@k"), ("complete", "complete@k")):
        cells = "".join(f"{report.overall[key][str(k)]:>9.4f}" for k in ks)
        lines.append(f"{label:<16}{cells}")
    lines.append(f"{'mrr':<16}{report.overall.get('mrr', 0.0):>9.4f}")

    exceeds = report.overall.get("gold_exceeds_k", {})
    lines.append("")
    lines.append(
        "questions with more gold units than k (cannot fit, so cannot be found): "
        + ", ".join(f"k={k} {exceeds.get(str(k), 0)}" for k in ks)
    )

    if report.by_evidence:
        lines.append("")
        lines.append("by evidence needed (complete@k is the ceiling on end-to-end accuracy)")
        lines.append("  " + f"{'group':<16}{'n':>6}" + "".join(f"{'k=' + str(k):>9}" for k in ks))
        for group, summary in report.by_evidence.items():
            cells = "".join(f"{summary['complete@k'][str(k)]:>9.4f}" for k in ks)
            lines.append(f"  {group:<16}{summary['n']:>6}{cells}")

    return "\n".join(lines)


def save(
    report: CeilingReport, per_question: list[QuestionResult], out_root: Path, name: str
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    target = out_root / f"{stamp}_retrieval_ceiling_{name}"
    target.mkdir(parents=True, exist_ok=True)

    config = {
        "split": report.split,
        "retriever": report.retriever,
        "ks": list(report.ks),
        "pool": report.pool,
        "answerable_only": report.answerable_only,
        "reranked": report.reranked,
        "weights": report.weights,
        "ground_truth": "qa.ann_table_rows + qa.ann_text_rows",
        "scope": "within the question's own document",
    }
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    with (target / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for result in per_question:
            fh.write(json.dumps(asdict(result)) + "\n")

    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="path to a FinQA split json file")
    parser.add_argument("--limit", type=int, default=None, help="only the first N records")
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--out", default=str(DEFAULT_REPORTS))
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--answerable",
        action="store_true",
        help="rank only units that can be an answer, dropping table header rows",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="reorder the top of the lexical ranking with the tuned feature weights",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=None,
        help=f"how deep to rank before taking k (default {DEFAULT_POOL} when reranking)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="a JSON file of weights, overriding the module's tuned defaults",
    )
    args = parser.parse_args(argv)

    ks = tuple(sorted({int(part) for part in args.ks.split(",") if part.strip()}))
    if not ks:
        parser.error("at least one k value is required")

    max_k = max(ks)
    pool = args.pool if args.pool is not None else (DEFAULT_POOL if args.rerank else max_k)
    if pool < max_k:
        parser.error(f"the pool is {pool} but k goes up to {max_k}")

    weights = load_weights(args.weights)
    if args.rerank and not any(
        value for feature, value in zip(FEATURES, weights.vector()) if feature != "bm25"
    ):
        parser.error(
            "--rerank needs a weight above zero besides bm25, or it only repeats the lexical order"
        )

    path = Path(args.split)
    examples = load_split(path)
    if args.limit:
        examples = examples[: args.limit]

    index = BM25Index(k1=args.k1, b=args.b)
    candidates_by_record: dict[str, set[str]] = {}
    units_by_record: dict[str, dict[str, EvidenceUnit]] = {}
    corpus_units = 0
    for example in examples:
        units = build_units(example)
        corpus_units += len(units)
        if args.answerable:
            # The one definition of which units can be answers, rather than the
            # filter being restated wherever retrieval happens.
            units = answerable_units(example)
        candidates_by_record[example.id] = {unit.id for unit in units}
        units_by_record[example.id] = {unit.id: unit for unit in units}
        for unit in units:
            index.add(unit.id, unit.text)
    index.finalize()
    results: list[QuestionResult] = []
    skipped_no_annotation = 0
    skipped_no_units = 0

    for example in examples:
        gold = gold_unit_ids(example)
        if not gold:
            skipped_no_annotation += 1
            continue

        candidates = candidates_by_record[example.id]
        if not candidates:
            skipped_no_units += 1
            continue

        ranked = index.search(example.question, candidates=candidates, top_k=pool)
        if args.rerank:
            units = units_by_record[example.id]
            ranked = rerank(
                example.question,
                [Candidate(unit=units[doc_id], score=score) for doc_id, score in ranked],
                weights=weights,
                top_k=max_k,
            )
        ranked_ids = [doc_id for doc_id, _ in ranked][:max_k]
        first_hit = next(
            (rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in gold),
            None,
        )

        needs = evidence_kinds(example)
        result = QuestionResult(
            id=example.id,
            gold_units=len(gold),
            document_units=len(candidates),
            needs=KIND_LABELS.get(needs, " + ".join(needs) or "unknown"),
            first_hit_rank=first_hit,
        )
        for k in ks:
            top = set(ranked_ids[:k])
            key = str(k)
            result.hit_at[key] = bool(top & gold)
            result.complete_at[key] = gold <= top
            result.recall_at[key] = round(len(top & gold) / len(gold), 4)

        results.append(result)

    report = CeilingReport(
        split=str(path),
        retriever=retriever_name(args, weights),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ks=ks,
        records=len(examples),
        corpus_units=corpus_units,
        evaluated=len(results),
        skipped_no_annotation=skipped_no_annotation,
        skipped_no_units=skipped_no_units,
        gold_inds_disagreements=gold_inds_disagreements(examples),
        pool=pool,
        answerable_only=bool(args.answerable),
        reranked=bool(args.rerank),
        weights={name: value for name, value in zip(FEATURES, weights.vector())},
    )
    report.overall = summarise(results, ks)

    grouped: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        grouped[result.needs].append(result)
    report.by_evidence = {group: summarise(rows, ks) for group, rows in sorted(grouped.items())}

    print(render(report))

    if not args.no_save:
        target = save(report, results, Path(args.out), path.stem)
        print(f"\nsaved: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
