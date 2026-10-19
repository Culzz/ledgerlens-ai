"""Tune the reranker's weights on train, then report them on dev.

    python -m evaluation.tune_reranker --limit 2500

The weights in ``app/services/retrieval/rerank.py`` are numbers, and a number
nobody can explain is a number nobody should trust. This is where they come from: a
grid search over complete@10 on the train split, and then one look at dev.

Five things about the protocol, because they are the whole point of the file.

* **Selection happens on train only.** Dev is touched once, by the winning weights,
  to report what they do on questions that had no part in choosing them.
* **The gold set is the unfiltered annotation.** The candidate pool holds only
  answerable units, so a question whose annotation pointed at a header row could
  never be complete and the loss would show up in the metric rather than being
  defined away. On this corpus there are no such questions, which is the point.
* **Features are computed once and cached.** They do not depend on the weights, so
  the search is a set of dot products over a cached matrix rather than a re-run of
  retrieval. That is what makes a grid of this size affordable, and it means every
  point in the grid was scored against exactly the same pool.
* **The search prefers the simplest winner.** Several weightings land within noise
  of the best score, and the one with the smallest total magnitude is taken among
  them. A grid this size can fit a little of the train split by accident, and
  preferring the flattest solution is the cheap defence.
* **The split, the limit and the grid all go in the artifact.** A weight vector
  without them is not reproducible, and this file's output is the only thing behind
  the numbers in ``rerank.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.services.dataset import FinQAExample, load_split
from app.services.retrieval import (
    DEFAULT_POOL,
    FEATURES,
    BM25Index,
    Candidate,
    Weights,
    answerable_units,
    build_units,
    gold_unit_ids,
    tokenize,
)
from app.services.retrieval.rerank import features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "finqa" / "train.json"
DEFAULT_DEV = ROOT / "data" / "finqa" / "dev.json"
DEFAULT_REPORTS = ROOT / "evaluation" / "reports"

# The grid. BM25 is held at 1 as the unit of comparison: every other weight reads as
# "how much of a BM25 point is this feature worth". Letting BM25 move too would only
# rescale every row of the search identically.
GRID = {
    "bm25": (1.0,),
    "numbers": (0.0, 0.5, 1.0, 2.0),
    "coverage": (0.0, 0.5, 1.0),
    "phrase": (0.0, 0.5, 1.0),
    "kind": (0.0, 0.5, 1.0, 2.0),
}

# A weighting within this much of the best on train counts as tied, and the tie is
# broken toward the flattest one.
TIE = 0.002


@dataclass
class Pool:
    """One question's candidates, with everything needed to rescore them.

    The feature vectors do not depend on the weights, so a search over weights is a
    dot product over this and nothing else.
    """

    id: str
    ids: list[str]
    scores: list[float]
    vectors: list[list[float]]
    gold: list[str]
    counted: bool

    def order(self, weights: Sequence[float]) -> list[int]:
        totals = [
            sum(weight * value for weight, value in zip(weights, vector)) for vector in self.vectors
        ]
        # The same tie-break the reranker uses: lexical score, then id.
        return sorted(
            range(len(self.ids)),
            key=lambda index: (-totals[index], -self.scores[index], self.ids[index]),
        )

    def complete_at(self, weights: Sequence[float], k: int) -> bool:
        if not self.counted:
            return False
        order = self.order(weights)
        return set(self.gold) <= {self.ids[index] for index in order[:k]}

    def first_hit(self, weights: Sequence[float]) -> int | None:
        if not self.counted:
            return None
        gold = set(self.gold)
        for rank, index in enumerate(self.order(weights), start=1):
            if self.ids[index] in gold:
                return rank
        return None


@dataclass
class Extraction:
    split: str
    pool: int
    features: list[str] = field(default_factory=list)
    records: int = 0
    units: int = 0
    answerable: int = 0
    pools: list[Pool] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "split": self.split,
                    "pool": self.pool,
                    "features": self.features,
                    "records": self.records,
                    "units": self.units,
                    "answerable": self.answerable,
                    "pools": [asdict(pool) for pool in self.pools],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Extraction:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            split=raw["split"],
            pool=raw["pool"],
            features=raw["features"],
            records=raw["records"],
            units=raw["units"],
            answerable=raw["answerable"],
            pools=[Pool(**pool) for pool in raw["pools"]],
        )


def extract(examples: Sequence[FinQAExample], *, split: str, pool: int) -> Extraction:
    """Rank every question's document once, and keep the candidates.

    The index holds the parent blocks as well as the units, so a row can be scored
    partly by the table it is in. Parents are never candidates: the ranking is over
    answerable units, which is the header-row exclusion measured on its own.
    """
    index = BM25Index()
    units_by_record: dict[str, list] = {}
    total_units = 0

    for example in examples:
        every = build_units(example)
        total_units += len(every)
        kept = answerable_units(example)
        units_by_record[example.id] = kept
        for unit in kept:
            index.add(unit.id, unit.text)
    index.finalize()

    extraction = Extraction(
        split=split,
        pool=pool,
        features=list(FEATURES),
        records=len(examples),
        units=total_units,
        answerable=sum(len(kept) for kept in units_by_record.values()),
    )

    for example in examples:
        kept = units_by_record[example.id]
        if not kept:
            continue

        ranked = index.search(example.question, candidates={unit.id for unit in kept}, top_k=pool)
        score_of = dict(ranked)
        rows = [
            Candidate(unit=unit, score=score_of[unit.id]) for unit in kept if unit.id in score_of
        ]
        if not rows:
            continue

        question_tokens = tokenize(example.question)
        top_score = max(row.score for row in rows)
        vectors = [
            list(
                features(
                    question_tokens,
                    tokenize(row.unit.text),
                    is_table=row.unit.is_table,
                    score=row.score,
                    top_score=top_score,
                )
            )
            for row in rows
        ]

        gold = gold_unit_ids(example)
        extraction.pools.append(
            Pool(
                id=example.id,
                ids=[row.unit.id for row in rows],
                scores=[row.score for row in rows],
                vectors=vectors,
                # Deliberately not intersected with the pool: see the module docstring.
                gold=sorted(gold),
                counted=bool(gold),
            )
        )

    return extraction


def summary(pools: Sequence[Pool], weights: Sequence[float], k: int) -> tuple[float, float]:
    """complete@k and mrr over the questions that carry an annotation."""
    counted = [pool for pool in pools if pool.counted]
    complete = sum(pool.complete_at(weights, k) for pool in counted) / len(counted)
    reciprocal = [1 / rank for pool in counted if (rank := pool.first_hit(weights)) is not None]
    return complete, sum(reciprocal) / len(counted)


def search(pools: Sequence[Pool], k: int) -> list[tuple[float, float, tuple[float, ...]]]:
    """Every weighting in the grid, best first."""
    counted = [pool for pool in pools if pool.counted]
    rows: list[tuple[float, float, tuple[float, ...]]] = []
    for values in itertools.product(*(GRID[key] for key in FEATURES)):
        complete = sum(pool.complete_at(values, k) for pool in counted) / len(counted)
        reciprocal = [1 / rank for pool in counted if (rank := pool.first_hit(values)) is not None]
        rows.append((complete, sum(reciprocal) / len(counted), values))
    rows.sort(key=lambda row: (-row[0], -row[1], sum(abs(value) for value in row[2])))
    return rows


def alone(name: str) -> tuple[float, ...]:
    """BM25 plus one feature, so a feature can be judged on its own."""
    return tuple(1.0 if key in ("bm25", name) else 0.0 for key in FEATURES)


def describe(values: Sequence[float]) -> str:
    return (
        "  ".join(f"{name}={value:g}" for name, value in zip(FEATURES, values) if value) or "none"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument(
        "--pool", type=int, default=DEFAULT_POOL, help="how deep the reranker may reach"
    )
    parser.add_argument("--k", type=int, default=10, help="the k the search maximises")
    parser.add_argument("--limit", type=int, default=None, help="only the first N records")
    parser.add_argument("--cache", default=None, help="where to cache the extracted features")
    parser.add_argument("--no-dev", action="store_true", help="skip the dev check")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache) if args.cache else Path(tempfile.gettempdir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(split_path: str) -> Extraction:
        # The limit is part of the cache key, so a smoke run over 200 records cannot
        # be mistaken later for a run over the whole split.
        suffix = f"-{args.limit}" if args.limit else ""
        path = cache_dir / f"ledgerlens-rerank-{Path(split_path).stem}-{args.pool}{suffix}.json"
        if path.exists():
            stored = Extraction.load(path)
            # The vectors in there are one number per feature. A cache written before
            # the feature set changed would load without complaining and every weight
            # after it would mean the wrong thing, so it is checked instead of trusted.
            if stored.features == list(FEATURES):
                return stored
            print(f"cache:    {path} was built for {stored.features}, ignoring it")
        examples = load_split(split_path)
        if args.limit:
            examples = examples[: args.limit]
        print(f"extracting {len(examples)} records from {split_path} ...", flush=True)
        result = extract(examples, split=str(split_path), pool=args.pool)
        result.save(path)
        return result

    train = cached(args.train)
    counted = len([pool for pool in train.pools if pool.counted])
    print(
        f"train: {counted} annotated questions, {train.answerable:,} answerable of"
        f" {train.units:,} units, pool {train.pool}\n"
    )

    ranked = search(train.pools, args.k)
    best = ranked[0][0]
    tied = [row for row in ranked if row[0] >= best - TIE]
    chosen = min(tied, key=lambda row: sum(abs(value) for value in row[2]))

    print(f"{len(ranked)} weightings scored on train; {len(tied)} within {TIE} of the best\n")
    print(f"{'train complete@' + str(args.k):<20}{'mrr':>9}   weights")
    for complete, mrr, values in ranked[:6]:
        print(f"{complete:<20.4f}{mrr:>9.4f}   {describe(values)}")
    print(f"\nchosen (flattest of the tied): {describe(chosen[2])}")
    print(f"weights tuple: {list(chosen[2])}\n")

    print(f"{'train, one feature beside bm25':<32}{'complete@' + str(args.k):>12}{'mrr':>9}")
    for name in FEATURES:
        complete, mrr = summary(train.pools, alone(name), args.k)
        print(f"  {name:<30}{complete:>12.4f}{mrr:>9.4f}")

    dev_result = None
    if not args.no_dev:
        dev = cached(args.dev)
        dev_counted = len([pool for pool in dev.pools if pool.counted])
        print(
            f"\ndev: {dev_counted} annotated questions, {dev.answerable:,} answerable of"
            f" {dev.units:,} units, pool {dev.pool}"
        )
        print("dev had no part in choosing these weights\n")
        print(f"{'dev':<32}{'complete@' + str(args.k):>12}{'mrr':>9}")
        dev_result = {"split": args.dev, "questions": dev_counted}
        for label, key, values in (
            ("  bm25 only", "baseline", Weights().vector()),
            ("  reranked", "reranked", chosen[2]),
        ):
            complete, mrr = summary(dev.pools, values, args.k)
            print(f"{label:<32}{complete:>12.4f}{mrr:>9.4f}")
            dev_result[key] = {"complete_at_k": round(complete, 4), "mrr": round(mrr, 4)}

        # The five best on train, scored on dev afterwards, once the choice had already
        # been made. If the whole band lands in the same place here, then which one the
        # tie rule picked is not carrying the result, which is worth knowing before
        # quoting any of them. The chosen weighting is the "reranked" line above.
        print("\nthe five best on train, on dev afterwards")
        band = []
        for complete, _, values in ranked[:5]:
            dev_complete, dev_mrr = summary(dev.pools, values, args.k)
            band.append(
                {
                    "weights": describe(values),
                    "train_complete_at_k": round(complete, 4),
                    "dev_complete_at_k": round(dev_complete, 4),
                    "dev_mrr": round(dev_mrr, 4),
                }
            )
            print(f"  train {complete:.4f}  dev {dev_complete:.4f}  {describe(values)}")
        dev_result["band"] = band
        spreads = [row["dev_complete_at_k"] for row in band]
        dev_result["band_dev_spread"] = round(max(spreads) - min(spreads), 4)

    target = DEFAULT_REPORTS / "rerank_weights.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "k": args.k,
                "pool": args.pool,
                "tie": TIE,
                "features": list(FEATURES),
                # The whole grid, so the search can be repeated and so it is visible
                # which weightings were considered and which were never on the table.
                "grid": {key: list(value) for key, value in GRID.items()},
                "cross_validation": "weights chosen on train, reported on dev",
                "train": {
                    "split": args.train,
                    "limit": args.limit,
                    "questions": counted,
                    "complete_at_k": round(chosen[0], 4),
                    "mrr": round(chosen[1], 4),
                    "best_weighting": describe(ranked[0][2]),
                    "tied_with_best": len(tied),
                },
                "chosen": {name: value for name, value in zip(FEATURES, chosen[2])},
                "dev": dev_result,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
