# The calculation engine

The engine is the part of LedgerLens that never calls a model. It runs a small
program to the letter, and it knows three kinds of thing.

## Operations

**Arithmetic** — `add`, `subtract`, `multiply`, `divide`, and `exp` for powers,
plus `greater` for the yes/no questions. **Table selection** — `table_sum`,
`table_max`, `table_min`, `table_average`, which take a row label and a column
header rather than a figure. **Financial formulas** — `cagr`, `present_value`,
`perpetuity`, `dcf`, `ebitda`, `quick_ratio` and `debt_to_equity`.

The formulas are named because the name is the useful part. `divide(637, 5)` is
arithmetic, and nothing downstream can say what it was for. A compound growth rate
is a quantity someone asked for, and the trail can only say `cagr` if there is a
`cagr` to call. Each one takes its operands from the caller, so nothing here goes
hunting for "revenue" in a report and guesses; where a formula refuses, it names
the argument and the amount to pass instead, because on the model path that text
is what gets retried.

## Why `dcf` takes five arguments

It takes five because the years you can see and the years you cannot are two
different assumptions — the forecast growth and the terminal growth — and with a
single rate for both the answer collapses to `perpetuity` at *every* horizon,
which is a property worth seeing in a signature rather than discovering in a
spreadsheet. And a forecast whose years differ from one another still can't be
written as a single call, because a step holds one number; that one is a sum of
`present_value` steps, and the trail then shows every year instead of one figure.

## Frozen prompt, growing engine

None of these operations are in the benchmark prompt. `program-v3` produced the
end-to-end runs in `docs/evaluation.md`, and adding operations to it would have
changed what those numbers mean, so the primitives are engine capability for the
product track and the prompt is left alone.

## What the engine's replay proves — and what it does not

The engine replays all 883 gold programs in FinQA's dev split exactly, with no
model involved:

```
python -m evaluation.replay_gold_programs --split data/finqa/dev.json
```

Run: `2026-10-16T16-15-30_replay_dev` — 883 of 883 matched, 0 mismatched, 0
errors. Its `metrics.json` is committed.

That is a check on the **engine**, not on a model: it says the arithmetic executes
FinQA's own reasoning programs correctly. It says nothing about whether the model
picks the right figures or writes the right program — that is what the end-to-end
strict score measures, and it is why a wrong answer downstream is attributed to
retrieval or reasoning rather than arithmetic.

## The guardrails

Every formula is checked against a hand calculation in
`backend/tests/unit/test_finance.py` and against the guardrail in
`backend/tests/unit/test_grounding.py`, since a formula the audit didn't recognise
would be a way to get a figure past it.

The free, no-model measurements that need no key:

```
python -m evaluation.replay_gold_programs --split data/finqa/dev.json
python -m evaluation.retrieval_ceiling   --split data/finqa/dev.json
```

`tune_reranker` (see `docs/retrieval-evaluation.md`) needs `data/finqa/train.json`,
which the download script fetches but the repository does not carry.
