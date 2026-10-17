# LedgerLens AI

Ask questions about financial reports and get answers you can actually check.

You point it at a report and ask something like "how much did operating income grow between 2022
and 2023". It finds both figures, does the arithmetic, and shows you where each number came from.
If it can't find what it needs, it says so instead of inventing something.

The chat part isn't the interesting bit. The point is that the language model never does the maths
and never gets to supply a number.

## Status

Early. The dataset loader, the calculation engine, a retrieval measurement and the evaluation
harness are in.

The engine replays every gold program in FinQA's dev split exactly, which is a check on the engine
and not on a model.

Retrieval has been measured as well. BM25 over 26,834 evidence units, dev split, 818 questions
carrying annotations:

| k | recall@k | complete@k |
| ---- | ---- | ---- |
| 1 | 0.432 | 0.329 |
| 3 | 0.662 | 0.532 |
| 5 | 0.764 | 0.665 |
| 10 | 0.868 | 0.812 |

`complete@k` is the share of questions where *all* the evidence a question needs is inside the top
k, so it caps end-to-end accuracy for anything that reads only those k units.

A useful thing fell out of that number. FinQA documents are small — a few table rows and a couple of
dozen narrative lines — so a prompt containing the whole document already contains the evidence. On
this benchmark retrieval is not the bottleneck; the reasoning is. Retrieval and citation are what
the product needs, and FinQA cannot tell you whether they work.

The evaluation harness and the first baseline exist, and the model layer is provider-neutral. A real
provider is wired up, and it is metered: Gemini's free tier allows 20 requests a day per model, so a
150-question pilot is around eight days of resuming rather than one afternoon. The runner is
resumable for exactly this reason, and a run that hits the ceiling prints when the quota returns and
the command that continues it.

There is no accuracy figure in this repo yet. If you see accuracy numbers for LedgerLens anywhere,
they didn't come from here. Numbers only appear once a script under `evaluation/` produces them.

`LLM_PROVIDER` defaults to `fake`, which answers offline with no key and no spend. That is how the
harness, its resume behaviour and its error handling are tested.

```
python -m evaluation.replay_gold_programs --split data/finqa/dev.json
python -m evaluation.retrieval_ceiling   --split data/finqa/dev.json
python -m evaluation.check_provider
python -m evaluation.run_eval            --split data/finqa/dev.json --system direct --limit 150
```

`check_provider` spends one request to confirm the key and the model work, which is cheaper than
finding out forty minutes into a run.

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

No API key needed. The model layer is provider-neutral and defaults to an offline provider, so
everything runs without one. When you want real answers, set `LLM_PROVIDER` and `LLM_MODEL` in
`.env` and add a key — the adapter speaks the OpenAI chat-completions shape, so Gemini, Groq,
Mistral, NVIDIA Build, OpenRouter and OpenAI are a change to configuration rather than to code.

Worth checking a provider's free tier before pointing a split at it, and checking the specific model
rather than the provider, because those terms change without notice. Gemini's free tier allows 20
requests a day per model, Groq's 1,000, and Mistral's free mode answers 429 on the first request for
its Medium and Small chat models while its Ministral ones still answer.

## Plan

```
1. repo, FinQA loader, data checks                                    done
2. calculation engine, tested against FinQA's own reasoning programs   done
3. lexical retrieval, and the ceiling it puts on accuracy              done
4. evaluation harness and baselines                                    <- here
5. hybrid retrieval and orchestrator
6. verifier and abstention
7. PDF ingestion
8. API and frontend
9. experiments, docker, writeup
```

The engine replays the 883 gold programs in the dev split with no model involved, so a wrong answer
from here on is a retrieval or reasoning problem, not an arithmetic one.

## What this isn't

Not a stock predictor, not investment advice, not something to run real money past. Scoring well on
a benchmark doesn't make it suitable for an actual investment decision.

## Benchmark

FinQA (Chen et al., EMNLP 2021). Splits used as published, never reshuffled. Notes on that and the
metric definitions are in `docs/evaluation_protocol.md`. No dataset files are committed here.
