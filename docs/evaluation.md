# End-to-end evaluation

The protocol — dataset, splits, leakage rules, metric definitions — is in
`docs/evaluation_protocol.md`. This page records what was measured on the FinQA dev
split and what the numbers say. Every figure below can be checked against the run
directory named beside it under `evaluation/reports/`; each directory carries a
`config.json` (split, provider, model, prompt version) and a `metrics.json`.

## Running the evaluation

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

The first two cost nothing and spend no quota: they replay reasoning programs the
dataset already contains, and they measure retrieval without a model in the loop.
`tune_reranker` needs `data/finqa/train.json`, which the download script fetches
but the repository does not carry.

`check_provider` spends one request to confirm the key and the model work, which
is cheaper than finding out forty minutes into a run. `audit_grounding` replays a
finished run's stored replies through the verification check; it spends nothing,
because it calls no model — that is how the check in `docs/verification.md` was
measured before it was trusted.

One model throughout: `ministral-14b-2512` on Mistral's free tier. FinQA's dev
split, all 883 questions, no failed calls. It is a small model on purpose — the
interesting question is what a system around a small model can do, not what a
large one can do on its own.

Both baselines answer from the document alone. Neither sees FinQA's reasoning
program, its annotated evidence, or the reference answer.

## Results

| | strict | percent-equivalent | no answer |
| ---- | ---- | ---- | ---- |
| A — the whole document in the prompt | 0.2231 | 0.5108 | 0 |
| D — retrieve, then write a program | 0.3296 | 0.5051 | 76 |
| E — D, plus the figure check and one repair pass | 0.3488 | 0.5345 | 78 |

`strict` is FinQA's own rule and the headline: the answer matches the reference to
within 1e-3. `percent-equivalent` additionally accepts an answer a hundred times
the reference, because FinQA stores some percentages as fractions and a model that
writes "93.5" for 0.935 is not always wrong. It is a diagnostic, not a second
headline, and it is only loosened in that one direction.

Each row is a directory under `evaluation/reports/`:
`2026-10-16T15-09-40_direct@mistral_dev` for A,
`2026-10-16T16-15-45_program@mistral_dev` for D,
`2026-10-17T11-25-38_program+repair1@mistral_dev` for E. `config.json` and
`metrics.json` are committed for each of them, so a number quoted anywhere can be
checked against the run that produced it. The per-question predictions are not
committed, because they are large and working data.

**Strict is the column that moved.** Making the engine do the arithmetic and the
model write a program is worth 10.7 points — same questions, same model. Paired
over all 883, D answers 163 questions A missed and misses 69 that A got.

**The looser column did not move**, and that is the number worth staring at. 155
of D's non-strict answers are the same percentage disagreement as ever: where
FinQA stores 0.35 the model writes 35.099. D commits to a figure the engine
computed rather than one the model recalled, so it makes that mistake less often —
155 questions against A's 254 — but A already gets those questions right in the
other convention, so the totals land in the same place. A gain that shows up in
one column and not the other is the kind to be suspicious of, which is why both
are printed.

## What the pipeline costs per question

76 questions (8.6%) came back as a reply that could not be turned into a runnable
program, mostly the model naming an evidence label where a value belongs —
`divide(table_1, table_2)` — or putting `$` and `/` inside a literal. Those count
as no answer, not as wrong answers, which is why they show up as a lower headline
instead of hiding inside one. Four replies were truncated by the provider's output
limit.

Of the rest: 338 wrong numbers, 155 the percentage convention, 10 correct but
rounded off (`16750000` where the reference is `16750002.0`), 8 marginal, 5 wrong
text.

## What the repair pass is worth

The trust layer (E) is D plus the figure check and one repair pass: 17 questions
better on strict, 26 on percent-equivalent. Most of that is not the trust layer,
and the data says so rather than leaving it to a reader to wonder.

The repair loop only touches a question whose *first* attempt fails, so the 775
questions where it never fired took the identical code path in both runs. On those
the answer moved from wrong to right 34 times and from right to wrong 26 times.
**Unchanged code moves by about ±30 questions out of 883**, so a 17-question
difference is inside the noise — and 30 of those 60 flips are the
strict-versus-percent convention, where the same question comes back as `93.5` in
one run and `0.935` in the next.

What can be attributed is the population the feature touched. The loop fired on
108 questions (12.2%): 10 ended correct where D had been wrong or silent, 1 ended
wrong where D had been right. That is a net of +9, about a point, and it is the
honest size of the feature. It costs a second model call on those 108 questions,
and that is where E's 14.9% token increase comes from.

## The verification check in the end-to-end runs

Run D refused 42 answers on the ungrounded-figure check; run E refused 39. Those
are two different runs, not one number borrowed between them — D without the
repair pass, E with it. In E none of the 39 refused answers had been right in D —
all 39 were already wrong. That was the measurement the check had to pass before
it shipped, and it passed it live rather than only in a replay.

**What the failures look like.** The 108 questions that reached the repair loop
are mostly malformed replies rather than subtly wrong reasoning: a reply carrying
no program, `unexpected end of program in 'null'`, or a program built entirely
from `0`. That is the honest shape of the remaining error, and it says the next
useful work is making the model produce a well-formed program at all, not making
the arithmetic better.

## Three different properties, measured separately

A credible answer needs three things, and passing one of them does not imply the
others:

1. **Grounding** — every figure the program uses is stated in the report. What the
   verification check measures.
2. **Calculation correctness** — the arithmetic runs as written. What the engine's
   883/883 gold-program replay establishes (see `docs/calculation-engine.md`).
3. **Answer correctness** — the figures were the *right* figures and the program
   was the *right* formula. What the strict score measures — and where the
   remaining error lives.

A grounded, correctly-executed program can still be the wrong program: right
number, wrong year. The runs above score all three at once; the retrieval and
verification pages break the pieces apart.

## Reading these numbers

Each figure carries the split, the question count, the provider, the model and the
prompt version it came from, and every one is in that run's `config.json`. Nothing
is averaged across providers or question counts and nothing is quoted without its
count. If you see accuracy numbers for LedgerLens that don't correspond to a run
directory, they didn't come from here.

## Expectations, stated up front

Putting the table inline and asking GPT-4 already scores around 65% on FinQA, and
chain-of-thought style prompting gets into the 70s. So beating that on raw accuracy
is not the goal and probably will not happen with a 14B model writing programs.

What is worth measuring is whether the answers can be verified: citations that
point at real evidence, arithmetic that checks out, and a clean refusal when the
evidence is not there. That is where this can be useful even if it is not the most
accurate thing in the room.

## The next experiment

The obvious next step is not another feature but a better measurement. ±30
questions is too wide to see a nine-question effect, and that number comes from
comparing two runs of the same configuration. Repeating D on a second day, and
running a subset of the split several times, would put a real band around every
later claim instead of a plausible one. Quota on a free tier is what makes that a
plan rather than a paragraph.

Retrieval is where the measurement is not under-powered, because it needs no
model: the ceiling in `docs/retrieval-evaluation.md` moves deterministically when
the retriever changes, so an improvement there is provable on a laptop in seconds
rather than inferred from a run-to-run difference.
