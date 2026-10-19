# LedgerLens AI

Ask questions about financial reports and get answers you can actually check.

You point it at a report and ask something like "how much did operating income grow between 2022
and 2023". It finds both figures, does the arithmetic, and shows you where each number came from.
If it can't find what it needs, it says so instead of inventing something.

The chat part isn't the interesting bit. The point is that the language model never does the maths
and never gets to supply a number.

## Status

The dataset loader, the calculation engine, the retrieval measurement, the evaluation harness, two
baselines, the API, the frontend and the verification layer are in. PDF ingestion is not — see the
Plan.

Every figure in a written program is looked up in the report before the answer is shown, against the
whole document rather than the retrieved rows. A program built entirely out of figures the report does
not state is refused instead of answered, and a program that will not run goes back to the model once
with the engine's reason. That check was measured before it was trusted: on the live run it refused 39
answers, and all 39 were already wrong.

The engine replays all 883 gold programs in FinQA's dev split exactly, with no model involved. That is
a check on the engine and not on a model, and it is what makes every number further down attributable:
a wrong answer from here on is a retrieval or reasoning problem, not an arithmetic one.

Run: `2026-10-16T16-15-30_replay_dev` — 883 of 883 matched, 0 mismatched, 0 errors. Its `metrics.json`
is committed.

The model layer is provider-neutral and `LLM_PROVIDER` defaults to `fake`, which answers offline with
no key and no spend. That is how the harness, its resume behaviour and its error handling are tested.
Switching to a real provider is configuration, not code.

```
python -m evaluation.replay_gold_programs --split data/finqa/dev.json
python -m evaluation.retrieval_ceiling   --split data/finqa/dev.json
python -m evaluation.retrieval_ceiling   --split data/finqa/dev.json --answerable --rerank
python -m evaluation.tune_reranker       --limit 2500
python -m evaluation.check_provider
python -m evaluation.run_eval            --split data/finqa/dev.json --system program --limit 150
python -m evaluation.review_run          evaluation/reports/<run-directory>
python -m evaluation.compare_runs        evaluation/reports/<run-a> evaluation/reports/<run-b>
python -m evaluation.audit_grounding     evaluation/reports/<run-directory>
```

The first two cost nothing and spend no quota: they replay reasoning programs the dataset already
contains, and they measure retrieval without a model in the loop. `tune_reranker` needs
`data/finqa/train.json`, which the download script fetches but the repository does not carry.

`check_provider` spends one request to confirm the key and the model work, which is cheaper than
finding out forty minutes into a run.

`audit_grounding` replays a finished run's stored replies through the verification check. It spends
nothing, because it calls no model, and it is how the check was measured before it was trusted.

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
| E — D, plus the figure check and one repair pass | 0.3488 | 0.5345 | 78 |

`strict` is FinQA's own rule and the headline: the answer matches the reference to within 1e-3.
`percent-equivalent` additionally accepts an answer a hundred times the reference, because FinQA
stores some percentages as fractions and a model that writes "93.5" for 0.935 is not always wrong.
It is a diagnostic, not a second headline, and it is only loosened in that one direction.

Each row is a directory under `evaluation/reports/`: `2026-10-16T15-09-40_direct@mistral_dev` for A,
`2026-10-16T16-15-45_program@mistral_dev` for D, `2026-10-17T11-25-38_program+repair1@mistral_dev`
for E. `config.json` and `metrics.json` are committed for each of them, so a number quoted here can be
checked against the run that produced it. The per-question predictions are not committed, because they
are large and working data.

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

**The trust layer, and how much of it is real.** E is D plus the figure check and one repair pass:
17 questions better on strict, 26 on percent-equivalent. Most of that is not the trust layer, and the
data says so rather than leaving it to a reader to wonder.

The repair loop only touches a question whose *first* attempt fails, so the 775 questions where it
never fired took the identical code path in both runs. On those the answer moved from wrong to right
34 times and from right to wrong 26 times. **Unchanged code moves by about ±30 questions out of 883**,
so a 17-question difference is inside the noise — and 30 of those 60 flips are the strict-versus-percent
convention, where the same question comes back as `93.5` in one run and `0.935` in the next.

What can be attributed is the population the feature touched. The loop fired on 108 questions (12.2%):
10 ended correct where D had been wrong or silent, 1 ended wrong where D had been right. That is a net
of +9, about a point, and it is the honest size of the feature. It costs a second model call on those
108 questions, and that is where E's 14.9% token increase comes from.

**The check cost nothing.** E refused 39 answers, and none of the 39 was right in D — all 39 were
already wrong. That was the measurement the check had to pass before it shipped, and it passed it live
rather than only in a replay.

**What the failures look like.** The 108 questions that reached the repair loop are mostly malformed
replies rather than subtly wrong reasoning: a reply carrying no program, `unexpected end of program in
'null'`, or a program built entirely from `0`. That is the honest shape of the remaining error, and it
says the next useful work is making the model produce a well-formed program at all, not making the
arithmetic better.

### Retrieval

BM25 over 26,834 evidence units, dev split, 818 questions carrying annotations. `complete@k` is the
share of questions where *all* the evidence a question needs is inside the top k, so it caps
end-to-end accuracy for anything that reads only those k units.

| configuration | recall@10 | complete@5 | complete@10 | MRR |
| ---- | ---- | ---- | ---- | ---- |
| BM25, every unit | 0.868 | 0.665 | 0.812 | 0.686 |
| the header row is no longer a candidate | 0.879 | 0.675 | 0.828 | 0.697 |
| then the reranker | 0.939 | 0.785 | 0.903 | 0.821 |

Two changes, measured separately, because they are different claims.

**The header row is a label, not evidence.** Row 0 of every table holds the column headings, and it
is never the answer: across all 883 dev records the annotation points at it zero times. BM25 could
not know that, and put it in the top ten for 48% of questions — spending a slot to re-tell the reader
something it already has, since the header goes at the front of the prompt either way. Keeping it out
of the candidate set is worth +1.6 points. `EvidenceUnit.answerable` in `units.py` is the whole
definition of the rule, and everything that retrieves goes through it rather than restating it.

**Then rank what BM25 found.** BM25 gets the right table near the top and cannot order what is inside
it, because it only knows terms. The reranker scores the lexical top 25 by a weighted sum of features
— the lexical score, the share of the question's numbers the unit contains, word coverage, a phrase
match, and a prior that a table row is evidence more often than a narrative line, 17.4% of rows
against 1.8% of lines — and keeps the best 10. Worth +7.6 points.

| k=10, by what the question needs | n | BM25 | + not-the-header | + reranker |
| ---- | ---- | ---- | ---- | ---- |
| table only | 504 | 0.827 | 0.843 | 0.960 |
| table + narrative | 124 | 0.629 | 0.661 | 0.710 |
| narrative only | 190 | 0.890 | 0.895 | 0.879 |

That last column is a real trade rather than a clean win: narrative-only questions give up 1.6 points
and the other 628 gain between 4.8 and 11.7. Worth saying out loud instead of quoting the total.

The model is one number, chosen on the train split (2,500 records, 2,277 questions) and reported once
on dev: `bm25=1, kind=0.5`. The ablation is the interesting part — beside BM25, the table prior alone
takes train complete@10 from 0.800 to 0.867, and adding the four lexical features to it reaches
0.871. Four tenths of a point for four more weights, and on dev the one-feature weighting scores
0.903 against the all-features one's 0.896. The rest was not buying anything that survived the split,
so it is not in the model. `evaluation/tune_reranker.py` runs the grid, prefers the flattest weighting
among those tied within 0.002 on train, re-scores the top five on dev afterwards to show the choice is
not carrying the result, and writes `evaluation/reports/rerank_weights.json`.

Runs: `2026-10-16T11-57-33_retrieval_ceiling_dev` for BM25, then
`2026-10-19T00-08-49_retrieval_ceiling_dev` and `2026-10-19T00-08-22_retrieval_ceiling_dev` for the two
rows below it. Each run's `config.json` and `metrics.json` are committed, so every number above can be
re-derived from the split rather than taken on trust.

The reranker is not in the end-to-end runs A, D and E, which were recorded before it existed. The
retriever is a named setting on the orchestrator, so a run that used it is a different system and
carries `rerank` in its name; the results below describe the system they were measured on.

A useful thing fell out of the ceiling number. FinQA documents are small — a few table rows and a
couple of dozen narrative lines — so a prompt containing the whole document already contains the
evidence. On this benchmark retrieval is not the bottleneck; the reasoning is. Retrieval and citation
are what the product needs, and FinQA cannot tell you whether they work.

### Verification

Every figure in a program is classified against the whole report: one the report states, arithmetic
the model wrote itself, or neither. That last group is what the check is for, and it can be measured
with no model at all by replaying a finished run's replies.

| run D, 883 replies | |
| ---- | ---- |
| `grounded` | 692 |
| `partly_grounded` | 73 |
| `ungrounded`, refused | 42 |
| no program to check | 76 |
| correct answers, check off | 291 of 883 |
| correct answers, check on | 291 of 883 |
| refused answers that were right | 0 |
| refused answers that were already wrong | 42 |

The check costs nothing in accuracy on this split, and it removes 42 wrong answers. 26 of the 42 are
one behaviour: a program written entirely out of `0`, which runs cleanly and means nothing.

The 73 `partly_grounded` answers are the more interesting group and they are not refused. An answer
resting on one figure the report does not state is a weaker answer, not an invented one — 10 of the 73
were right. The audit trail names the figure either way, so a reader can see which one it was.

This bounds the check, not the repair loop standing next to it. A refused program goes back to the
model once with the engine's reason, and a replay of stored replies never sees the second attempt. Run
E is that live measurement, and it puts the same number on the check: 39 refusals, none of them right.

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

## What the engine can compute

The engine is the part of this that doesn't call a model. It runs a small program to the letter, and it
knows three kinds of thing.

**Arithmetic** - `add`, `subtract`, `multiply`, `divide`, and `exp` for powers, plus `greater` for the
yes/no questions. **Table selection** - `table_sum`, `table_max`, `table_min`, `table_average`, which
take a row label and a column header rather than a figure. **Financial formulas** - `cagr`,
`present_value`, `perpetuity`, `dcf`, `ebitda`, `quick_ratio` and `debt_to_equity`.

The formulas are named because the name is the useful part. `divide(637, 5)` is arithmetic, and nothing
downstream can say what it was for. A compound growth rate is a quantity someone asked for, and the
trail can only say `cagr` if there is a `cagr` to call. Each one takes its operands from the caller, so
nothing here goes hunting for "revenue" in a report and guesses; where a formula refuses, it names the
argument and the amount to pass instead, because on the model path that text is what gets retried.

Two things about `dcf` worth knowing before anyone uses it. It takes five arguments because the years
you can see and the years you cannot are two different assumptions - the forecast growth and the
terminal growth - and with a single rate for both the answer collapses to `perpetuity` at *every*
horizon, which is a property worth seeing in a signature rather than discovering in a spreadsheet. And
a forecast whose years differ from one another still can't be written as a single call, because a step
holds one number; that one is a sum of `present_value` steps, and the trail then shows every year
instead of one figure.

None of these are in the benchmark prompt. `program-v3` produced the runs in the table above, and
adding operations to it would have changed what those numbers mean, so the primitives are engine
capability for the product track and the prompt is left alone. Every formula is checked against a hand
calculation in `backend/tests/unit/test_finance.py` and against the guardrail in `test_grounding.py`,
since a formula the audit didn't recognise would be a way to get a figure past it.

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
api/         the one function Vercel serves, which imports the app above
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

One Vercel project. The frontend is built at deploy time and served from the CDN; the API is a single
Python function at `api/index.py`, which imports the same `create_app()` uvicorn runs locally — there
is no second code path and no CORS between the halves.

`vercel.json` is the whole deploy. It builds `frontend/dist`, fetches the FinQA dev split during the
build (the dataset is not committed), and gives the function sixty seconds, which is enough for a
model call plus a repair round on a free tier that answers in ten to thirty. The function bundle
carries `backend/` and `data/`; the build output never enters it.

Add a project on Vercel, import the repository, and set `MISTRAL_API_KEY` in Environment Variables.
The framework preset is set to none in `vercel.json`, so leave the dashboard's detection alone;
nothing else needs a value. The first deploy takes a few minutes and every push rebuilds it.

With no key set the app still runs and answers nothing, saying why on the card, so a deploy without a
key is still a working demo of the retrieval and the interface.

Two things about free tiers. A function is created on demand, so the first request after a quiet
spell pays for loading the corpus — a second or two, not a minute. And the key is a shared secret
sitting behind a public URL, which is why the app paces itself between calls and waits out a rate
limit rather than spending through one.

The `Dockerfile` is left over from an earlier deploy target. It builds the same app as one image, and
nothing in the hosted path uses it.


## Plan

```
1. repo, FinQA loader, data checks                                    done
2. calculation engine, tested against FinQA's own reasoning programs   done
3. lexical retrieval, and the ceiling it puts on accuracy              done
4. evaluation harness and baselines                                    done
5. orchestrator: retrieve the evidence, then write a program           done
6. verification: check the figures, refuse what has no home, repair a  done
   program that will not run
7. retrieval: keep labels out of the candidate set, then rerank         done
8. PDF ingestion
9. API and frontend                                                    done
10. experiments and writeup
```

The 76 unexecutable replies are what milestone 6 was for, and the check above is half of it. A program
that will not run, and a program whose figures have no home in the report, both go back to the model
once with the engine's reason attached — one round, no arguing. It is measured now, in run E, and the
measurement is less flattering than the feature: it fires on 108 questions and fixes 9 of them.

The obvious next step is not another feature but a better measurement. ±30 questions is too wide to
see a nine-question effect, and that number comes from comparing two runs of the same configuration.
Repeating D on a second day, and running a subset of the split several times, would put a real band
around every later claim instead of a plausible one. Quota on a free tier is what makes that a plan
rather than a paragraph.

Retrieval is where the measurement is not under-powered, because it needs no model: the ceiling above
moves deterministically when the retriever changes, so an improvement there is provable on a laptop in
seconds rather than inferred from a run-to-run difference. That is why it came before anything else on
the list, and why the two changes are reported as a retrieval metric with their own runs rather than as
an accuracy claim. The dense half of the plan is still open — eight dev questions have no gold unit
that lexical matching can reach at all, which is a floor no amount of reranking lifts — and the honest
place to spend on it is after the measurement above is wide enough to tell it worked.

## What this isn't

Not a stock predictor, not investment advice, not something to run real money past. Scoring well on
a benchmark doesn't make it suitable for an actual investment decision.

## Benchmark

FinQA (Chen et al., EMNLP 2021). Splits used as published, never reshuffled. Notes on that and the
metric definitions are in `docs/evaluation_protocol.md`. No dataset files are committed here.
