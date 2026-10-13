# LedgerLens AI

Ask questions about financial reports and get answers you can actually check.

You point it at a report and ask something like "how much did operating income grow between 2022
and 2023". It finds both figures, does the arithmetic, and shows you where each number came from.
If it can't find what it needs, it says so instead of inventing something.

The chat part isn't the interesting bit. The point is that the language model never does the maths
and never gets to supply a number.

## Status

Early. Right now this repo is scaffolding and docs. No pipeline code yet, and no benchmark results.
If you see accuracy numbers for LedgerLens anywhere, they didn't come from here. Numbers only appear
in this repo once a script under `evaluation/` produces them.

## Why it's built this way

Answers to financial questions get assembled from pieces scattered across a document: a table on
page 12, a line of prose on page 40, a footnote, last year's comparative column. Most of the work is
finding the right pieces and getting the arithmetic right.

Plain search finds the words but can't calculate. A plain LLM will happily write a confident
paragraph with a subtly wrong number in it. So the job is split:

- retrieval finds the evidence
- a small deterministic engine does the arithmetic
- the model reads the question and writes the answer

The model never supplies a financial value. The extracted document and the calculation engine are
the source of truth.

## Two tracks, and why that matters

The benchmark (FinQA) hands you the table already parsed. So it tests whether the system picks the
right rows and does the right sums. It does not test reading a real PDF. Those are separate pieces
of work, and only one of them is measured by the benchmark.

- **Benchmark track** - FinQA JSON in, answer out, scored against the reference answer. Measured.
- **Product track** - PDF in, text and tables extracted, then the same reasoning on top. Not covered
  by FinQA at all, and it needs its own evaluation.

It's tempting to blur these together. Don't. A good benchmark score says nothing about whether the
PDF reader works.

More detail in `docs/architecture.md`.

## What I expect from it

Putting the table inline and asking GPT-4 already scores around 65% on FinQA, and chain-of-thought
style prompting gets into the 70s. So beating that on raw accuracy isn't the goal and probably won't
happen.

What's worth measuring is whether the answers can be verified: citations that point at real evidence,
arithmetic that checks out, and a clean refusal when the evidence isn't there. That's where this can
be useful even if it isn't the most accurate thing in the room.

## Layout

```
backend/     FastAPI service and the actual logic (retrieval, reasoning, calculations, verification)
frontend/    React workspace
evaluation/  benchmark runner, baselines, metrics, reports
data/        dataset and uploads, git-ignored
docs/        requirements, architecture, evaluation plan, security
scripts/     data download and local run helpers
```

## Getting started

```
git clone <url>
cd ledgerlens-ai
python -m venv .venv
.venv\Scripts\activate          # or: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

No API key needed yet. The LLM bits live behind an optional extra (`.[llm]`) and don't get installed
until the baselines exist.

## Plan

```
1. repo, FinQA loader, data checks        <- here
2. calculation engine, tested against FinQA's own reasoning programs
3. evaluation harness + two baselines
4. hybrid retrieval + orchestrator
5. verifier and abstention
6. PDF ingestion
7. API and frontend
8. experiments, docker, writeup
```

## What this isn't

Not a stock predictor, not investment advice, not something to run real money past. Scoring well on
a benchmark doesn't make it suitable for an actual investment decision.

## Benchmark

FinQA (Chen et al., EMNLP 2021). Splits used as published, never reshuffled. Notes on that and the
metric definitions are in `docs/evaluation_protocol.md`. No dataset files are committed here.
