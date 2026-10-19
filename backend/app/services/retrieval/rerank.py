"""A deterministic reranker over the head of a lexical ranking.

BM25 finds the right table and gets the right row to the top of the list more often
than not. What it cannot do is order what it found, because it only knows about
terms. Two things it cannot see, and that this adds:

* A table row is a far more likely answer than a narrative line. Measured on the
  dev split's annotations, 17.4% of rows are evidence against 1.8% of lines. That
  is a prior about what the unit is, not about what it says, so no amount of term
  weighting will discover it.
* Order. BM25 is a bag of words, so a unit containing "flight attendants" scores
  the same as one containing both words far apart.

The score is a weighted sum of features, and the weights are tuned on the train
split and reported on dev by ``evaluation/tune_reranker.py``. Nothing here learns at
runtime, reads a model, or depends on the order candidates arrive in.

Two things were tried and are not here.

* **Scoring a unit partly by the block it sits in** - the table a row belongs to,
  the narrative a line belongs to - as a fourth feature. It needs the enclosing
  block indexed alongside the units, which changes the index's average document
  length and therefore every BM25 score in the corpus, so the reranked comparison
  would no longer have been against the baseline the README records. It also
  scored 0.7532 beside BM25 against BM25's 0.7857 on train, and took weight zero in
  every weighting that came near the top of the grid. It is designed for a report
  with many tables, where it would say *which* table; on this corpus a document has
  one, so it only says what the table prior already says.
* **Fuzzy number matching** - treating 1,234 and 1234 as the same figure. The
  tokeniser already does that, so the feature would have been a second copy of the
  first one.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .bm25 import tokenize
from .units import EvidenceUnit

# The features, in a fixed order. A tuple rather than a mapping, so that a weight
# and the number it multiplies cannot drift apart silently - ``Weights.vector`` is
# built from this and a test asserts the two agree.
FEATURES = ("bm25", "numbers", "coverage", "phrase", "kind")

# How long a run of shared words is allowed to count for. Three is enough to tell a
# phrase from a coincidence and short enough that a long question cannot farm it.
MAX_RUN = 3

# How deep the lexical pass ranks before the reranker takes over. 25 is where the
# lexical ceiling stops improving much - complete@25 is 0.944 against complete@10's
# 0.812, and past 25 it only creeps up - so the units the reranker can rescue are
# almost all within 25, and ranking further would cost time for nothing.
DEFAULT_POOL = 25


@dataclass(frozen=True)
class Weights:
    """How much each feature counts. Tuned, not guessed."""

    bm25: float = 1.0
    numbers: float = 0.0
    coverage: float = 0.0
    phrase: float = 0.0
    kind: float = 0.0

    def vector(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in FEATURES)


def lexical_weights() -> Weights:
    """The unweighted reranker: BM25 order, untouched.

    Kept as a named thing rather than a magic value, because "reranking changed
    nothing" and "reranking was not run" have to be distinguishable.
    """
    return Weights()


# Tuned on the train split by evaluation/tune_reranker.py, which reports them once on
# dev. The grid, the tie rule and both sets of numbers are in
# evaluation/reports/rerank_weights.json.
#
# One feature, at half a BM25 point. That is what the search chose, and the ablation
# behind it reads as a result in itself: beside BM25, the table prior alone takes
# train complete@10 from 0.8002 to 0.8674, while the best weighting in the whole grid
# reaches 0.8713 - four tenths of a point for four more weights. On dev the chosen
# one-feature weighting scores 0.9034 and the one using all four extra features scores
# 0.8961, so the rest was not buying anything that survived the split. Keeping it
# would have made the model harder to defend, not better.
DEFAULT_WEIGHTS = Weights(bm25=1.0, kind=0.5)


@dataclass(frozen=True)
class Candidate:
    """One unit, with the score the lexical pass gave it."""

    unit: EvidenceUnit
    score: float


def numbers_in(tokens: Sequence[str]) -> set[str]:
    """The numeric tokens of a piece of text."""
    return {token for token in tokens if token[:1].isdigit()}


def _longest_run(question: Sequence[str], unit: Sequence[str]) -> int:
    """The longest run of consecutive question tokens that appears in the unit."""
    if not question or not unit:
        return 0
    triples = set(zip(unit, unit[1:], unit[2:]))
    pairs = set(itertools.pairwise(unit))

    for run in range(min(MAX_RUN, len(question)), 0, -1):
        windows = set(zip(*(question[offset:] for offset in range(run))))
        if run >= 3 and windows & triples:
            return run
        if run == 2 and windows & pairs:
            return 2
        if run == 1 and set(question) & set(unit):
            return 1
    return 0


def features(
    question_tokens: Sequence[str],
    unit_tokens: Sequence[str],
    *,
    is_table: bool,
    score: float,
    top_score: float,
) -> tuple[float, ...]:
    """The feature vector for one candidate, in ``FEATURES`` order.

    Every feature is bounded to [0, 1] before it is weighted, so a weight means the
    same thing whatever the units of the underlying quantity were. That matters for
    BM25, whose scores are unbounded: it is divided by the largest score in the same
    candidate pool, so the feature describes a unit's standing inside the question's
    own document rather than an absolute number that varies from document to
    document.
    """
    question_numbers = numbers_in(question_tokens)
    unit_numbers = numbers_in(unit_tokens)
    question_words = {token for token in question_tokens if not token[:1].isdigit()}
    unit_words = {token for token in unit_tokens if not token[:1].isdigit()}

    named = (
        len(question_numbers & unit_numbers) / len(question_numbers) if question_numbers else 0.0
    )
    covered = len(question_words & unit_words) / len(question_words) if question_words else 0.0
    run = _longest_run(question_tokens, unit_tokens) / MAX_RUN

    return (
        score / top_score if top_score > 0 else 0.0,
        named,
        covered,
        run,
        1.0 if is_table else 0.0,
    )


def rerank(
    question: str,
    candidates: Sequence[Candidate],
    *,
    weights: Weights | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Order candidates by the weighted sum of their features, best first.

    The caller decides what a candidate is: this orders what it is given, and
    ``answerable_units`` is where a label is kept out of the candidate set. Passing
    ``lexical_weights()`` gets the BM25 order back, which is what a measurement has
    to use as its baseline.

    Deterministic to the last decimal: ties fall back to the lexical score and then
    to the unit id, so two runs over the same document cannot disagree, and the
    order does not depend on the order the candidates arrive in.
    """
    chosen = weights if weights is not None else DEFAULT_WEIGHTS
    if not candidates:
        return []

    question_tokens = tokenize(question)
    weight_vector = chosen.vector()
    top_score = max(candidate.score for candidate in candidates)

    scored: list[tuple[float, float, str]] = []
    for candidate in candidates:
        vector = features(
            question_tokens,
            tokenize(candidate.unit.text),
            is_table=candidate.unit.is_table,
            score=candidate.score,
            top_score=top_score,
        )
        total = sum(weight * value for weight, value in zip(weight_vector, vector))
        scored.append((total, candidate.score, candidate.unit.id))

    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [(unit_id, total) for total, _, unit_id in scored[:top_k]]
