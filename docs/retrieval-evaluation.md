# Retrieval evaluation

BM25 over 26,834 evidence units, dev split, 818 questions carrying annotations.
`complete@k` is the share of questions where *all* the evidence a question needs
is inside the top k, so it caps end-to-end accuracy for anything that reads only
those k units.

| configuration | recall@10 | complete@5 | complete@10 | MRR |
| ---- | ---- | ---- | ---- | ---- |
| BM25, every unit | 0.868 | 0.665 | 0.812 | 0.686 |
| the header row is no longer a candidate | 0.879 | 0.675 | 0.828 | 0.697 |
| then the reranker | 0.939 | 0.785 | 0.903 | 0.821 |

Two changes, measured separately, because they are different claims.

## The header row is a label, not evidence

Row 0 of every table holds the column headings, and it is never the answer: across
all 883 dev records the annotation points at it zero times. BM25 could not know
that, and put it in the top ten for 48% of questions — spending a slot to re-tell
the reader something it already has, since the header goes at the front of the
prompt either way. Keeping it out of the candidate set is worth +1.6 points.
`EvidenceUnit.answerable` in `units.py` is the whole definition of the rule, and
everything that retrieves goes through it rather than restating it.

## Then rank what BM25 found

BM25 gets the right table near the top and cannot order what is inside it, because
it only knows terms. The reranker scores the lexical top 25 by a weighted sum of
features — the lexical score, the share of the question's numbers the unit
contains, word coverage, a phrase match, and a prior that a table row is evidence
more often than a narrative line, 17.4% of rows against 1.8% of lines — and keeps
the best 10. Worth +7.6 points.

| k=10, by what the question needs | n | BM25 | + not-the-header | + reranker |
| ---- | ---- | ---- | ---- | ---- |
| table only | 504 | 0.827 | 0.843 | 0.960 |
| table + narrative | 124 | 0.629 | 0.661 | 0.710 |
| narrative only | 190 | 0.890 | 0.895 | 0.879 |

That last column is a real trade rather than a clean win: narrative-only questions
give up 1.6 points and the other 628 gain between 4.8 and 11.7. Worth saying out
loud instead of quoting the total.

## The one-weight model, and the ablation that killed the rest

The model is one number, chosen on the train split (2,500 records, 2,277
questions) and reported once on dev: `bm25=1, kind=0.5`. The ablation is the
interesting part — beside BM25, the table prior alone takes train complete@10 from
0.800 to 0.867, and adding the four lexical features to it reaches 0.871. Four
tenths of a point for four more weights, and on dev the one-feature weighting
scores 0.903 against the all-features one's 0.896. The rest was not buying
anything that survived the split, so it is not in the model.

`evaluation/tune_reranker.py` runs the grid, prefers the flattest weighting among
those tied within 0.002 on train, re-scores the top five on dev afterwards to show
the choice is not carrying the result, and writes
`evaluation/reports/rerank_weights.json`.

## Runs

`2026-10-16T11-57-33_retrieval_ceiling_dev` for BM25, then
`2026-10-19T00-08-49_retrieval_ceiling_dev` and
`2026-10-19T00-08-22_retrieval_ceiling_dev` for the two rows below it. Each run's
`config.json` and `metrics.json` are committed, so every number above can be
re-derived from the split rather than taken on trust.

The reranker is not in the end-to-end runs A, D and E in `docs/evaluation.md`,
which were recorded before it existed. The retriever is a named setting on the
orchestrator, so a run that used it is a different system and carries `rerank` in
its name; the results there describe the system they were measured on.

## What the ceiling number says about FinQA

A useful thing fell out of the ceiling number. FinQA documents are small — a few
table rows and a couple of dozen narrative lines — so a prompt containing the
whole document already contains the evidence. On this benchmark retrieval is not
the bottleneck; the reasoning is. Retrieval and citation are what the product
needs, and FinQA cannot tell you whether they work.

The dense half of the plan is still open — eight dev questions have no gold unit
that lexical matching can reach at all, which is a floor no amount of reranking
lifts. The honest place to spend on it is after the run-to-run measurement in
`docs/evaluation.md` is wide enough to tell it worked.
