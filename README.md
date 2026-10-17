# LedgerLens AI

Ask questions about financial reports and get answers you can actually check.

You point it at a report and ask something like "how much did operating income grow between 2022
and 2023". It finds both figures, does the arithmetic, and shows you where each number came from.
If it can't find what it needs, it says so instead of inventing something.

The chat part isn't the interesting bit. The point is that the language model never does the maths
and never gets to supply a number.

## Status

The dataset loader, the calculation engine, the retrieval measurement, the evaluation harness, two
baselines, the API and the frontend are in. Verification and PDF ingestion are not — see the Plan.

The engine replays all 883 gold programs in FinQA's dev split exactly, with no model involved. That is
a check on the engine and not on a model, and it is what makes every number further down attributable:
a wrong answer from here on is a retrieval or reasoning problem, not an arithmetic one.

The model layer is provider-neutral and `LLM_PROVIDER` defaults to `fake`, which answers offline with
no key and no spend. That is how the harness, its resume behaviour and its error handling are tested.
Switching to a real provider is configuration, not code.

```
python -m evaluation.replay_gold_programs --split data/finqa/dev.json
python -m evaluation.retrieval_ceiling   --split data/finqa/dev.json
python -m evaluation.check_provider
python -m evaluation.run_eval            --split data/finqa/dev.json --system program --limit 150
python -m evaluation.review_run          evaluation/reports/<run-directory>
python -m evaluation.compare_runs        evaluation/reports/<run-a> evaluation/reports/<run-b>
```

`check_provider` spends one request to confirm the key and the model work, which is cheaper than
finding out forty minutes into a run.

## Results

One model throughout: `ministral-14b-2512` on Mistral's free tier. FinQA's dev split, all 883
questions, no failed calls. It is a small model on purpose — the interesting question is what a
system around a small model can do, not what a large one can do on its own.

Both baselines answer from the document alone. Neither sees FinQA's reasoning program, its annotated
evidence, or the reference answer.

| | strict | percent-equivalent | no answer |
| ---- | ---- | ---- | ---- |
| A — the whole document in the prompt | 0.2231 | 0.5108 | 0 |
| D — retrieve, then write a program | 0.3296 | 0.5051 | 76 |

`strict` is FinQA's own rule and the headline: the answer matches the reference to within 1e-3.
`percent-equivalent` additionally accepts an answer a hundred times the reference, because FinQA
stores some percentages as fractions and a model that writes "93.5" for 0.935 is not always wrong.
It is a diagnostic, not a second headline, and it is only loosened in that one direction.

**Strict is the column that moved.** Making the engine do the arithmetic and the model write a program
is worth 10.7 points — same questions, same model. Paired over all 883, D answers 163 questions A
missed and misses 69 that A got.

**The looser column did not move**, and that is the number worth staring at. 155 of D's non-strict
answers are the same percentage disagreement as ever: where FinQA stores 0.35 the model writes
35.099. D commits to a figure the engine computed rather than one the model recalled, so it makes that
mistake less often — 155 questions against A's 254 — but A already gets those questions right in the
other convention, so the totals land in the same place. A gain that shows up in one column and not the
other is the kind to be suspicious of, which is why both are printed.

**What it costs.** 76 questions (8.6%) came back as a reply that could not be turned into a runnable
program, mostly the model naming an evidence label where a value belongs — `divide(table_1, table_2)`
— or putting `$` and `/` inside a literal. Those count as no answer, not as wrong answers, which is
why they show up as a lower headline instead of hiding inside one. Four replies were truncated by the
provider's output limit.

Of the rest: 338 wrong numbers, 155 the percentage convention, 10 correct but rounded off
(`16750000` where the reference is `16750002.0`), 8 marginal, 5 wrong text.

**What it means.** The engine replays every gold program exactly and a 14B model writing the programs
still lands at a third. What is left is evidence selection and notation, not arithmetic. That is the
argument for the retrieval and verification milestones, and against swapping in a bigger model and
calling the problem solved.

### Retrieval

BM25 over 26,834 evidence units, dev split, 818 questions carrying annotations:

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

### Reading these numbers

Each figure carries the split, the question count, the provider, the model and the prompt version it
came from, and every one is in that run's `config.json`. Nothing is averaged across providers or
question counts and nothing is quoted without its count. If you see accuracy numbers for LedgerLens
that don't correspond to a run directory, they didn't come from here.

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
backend/     FastAPI service and the actual logic (retrieval, reasoning, calculations)
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

## Running it somewhere

The image builds the frontend and the API together, so there is one container, one port and one URL.
It gets built on the host's machines, which is why nothing here needs Docker working locally.

`render.yaml` is the whole deploy. Push the repository, then on Render: **New → Blueprint**, point it
at the repository. It reads the file, builds the image, and asks for `MISTRAL_API_KEY` in the
dashboard — nothing else needs a value. The service comes up at `https://<name>.onrender.com`.

With no key set the app still runs and answers nothing, saying why on the card, so a deploy without a
key is still a working demo of the retrieval and the interface.

Two things about free tiers. The instance sleeps after fifteen minutes with no traffic and takes
about a minute to wake, so the first question after a quiet spell is slow. And the key is a shared
secret sitting behind a public URL, which is why the app paces itself between calls and waits out a
rate limit rather than spending through one.

## Plan

```
1. repo, FinQA loader, data checks                                    done
2. calculation engine, tested against FinQA's own reasoning programs   done
3. lexical retrieval, and the ceiling it puts on accuracy              done
4. evaluation harness and baselines                                    done
5. orchestrator: retrieve the evidence, then write a program           done
6. verifier and abstention                                             <- here
7. hybrid retrieval and reranking
8. PDF ingestion
9. API and frontend                                                    done
10. experiments and writeup
```

Two things the results above make obvious, and they are the next two things. The 76 unexecutable
replies: a reply that cannot be turned into a program should come back as a refusal with a reason
rather than as a failure, which is what milestone 6 is for. And retrieval is measured but not
improved — milestone 7 is the hybrid pass, and since the ceiling says retrieval is not what is
holding the benchmark back, that will be reported as a retrieval metric and not as an accuracy claim.

## What this isn't

Not a stock predictor, not investment advice, not something to run real money past. Scoring well on
a benchmark doesn't make it suitable for an actual investment decision.

## Benchmark

FinQA (Chen et al., EMNLP 2021). Splits used as published, never reshuffled. Notes on that and the
metric definitions are in `docs/evaluation_protocol.md`. No dataset files are committed here.
