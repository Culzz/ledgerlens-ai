# Model card

What's in here, what it's for, and where it breaks.

Version 0.3. The numbers below come from runs under `evaluation/reports/`, and none were written
before those runs existed.

## What it actually is

Not a single trained model. A small system made of parts, and not all the parts are there yet:

| Part | Choice | State |
|---|---|---|
| LLM | provider-neutral adapter over the OpenAI chat-completions shape | built. Reads the question, writes the program or the answer. Never a source of numbers |
| keyword search | in-process BM25 over table rows and narrative lines | built |
| reranking | the lexical top 25 rescored by a weighted feature sum tuned on train; deterministic, no model | built. In the served system; runs A, D and E predate it and are recorded as lexical |
| calculation | Python `Decimal`, a fixed allowlist of operations, plus named financial formulas (`cagr`, `dcf`, `perpetuity`, `present_value`, `ebitda`, `quick_ratio`, `debt_to_equity`) for the product path | built. What the benchmark measured is the arithmetic and the table selection; the formulas are never used by the benchmark prompt |
| verification | every figure in a written program looked up in the whole report; a program resting on no figure the report states is refused | built |
| repair | a program that will not run, or that fails the check, goes back to the model once with the engine's error | built |
| evaluation | a plain-Python harness over stored JSONL, and pytest | built |
| embeddings | local sentence-transformers, MiniLM class | planned |
| table parsing | PyMuPDF plus pdfplumber | planned |

The last two are only needed for the product track — reading a real PDF. The benchmark track runs
without them, and it is the track with numbers below.

## What it's for

For analysts, investors, researchers and finance teams poking at financial reports. Also, honestly,
as a portfolio and research project about evidence-grounded numerical reasoning.

Not for: investment advice, autonomous transactions, credit or lending decisions, audit reliance, or
anything where an unverified number has real consequences.

## How it works, and what follows from that

The model reads the question and writes a short program. A deterministic engine runs that program
against values taken from the document. If the question turns out not to match anything in the
document, retrieval stands aside, the whole document goes into the prompt, and the answer says so.

The consequence is that the model is never the source of truth for a number. That buys faithfulness
and auditability. It does not by itself buy accuracy, and the numbers below are the size of the gap.

Before any number is shown, every figure in the program is looked up in the report — the whole
document, not only the evidence rows that were retrieved. That is deliberate: a retrieval miss must
not turn into an accusation that the model invented a figure. The reading is arithmetic rather than
fuzzy matching. A figure that names its own unit ("$6.8 billion") is an amount and is read in any unit
the report might use. A bare one is read as itself and as the report's declared unit scaled. A percent
is read as a fraction and as a percentage, and is never unit-converted.

A figure that cannot be found is named in the audit trail rather than quietly dropped. If every figure
in a program is missing, the arithmetic is sound and the inputs are invented, and the answer is
refused instead of reported. If some are found, the answer stands and the trail says which figure was
the weak point.

## Limitations

- **Accuracy.** A third of FinQA's dev split under its own strict rule, with a small model. GPT-4 with
  the table inline is around 65% in the literature. This is not claiming to blow past that, and it
  does not.
- **A reply has to become a program.** 8.6% of questions came back as a reply the engine could not
  run, mostly the model naming an evidence label where a value belongs, replying with nothing, or
  writing a program made of `0`. Those are counted as no answer, not as wrong answers. One repair pass
  recovers 9% of them, which is why the rate is 8.8% with the pass and not 8.6% without: the loop
  fires on 12.2% of questions and mostly does not help.
- **The smallest effect is not measurable at this size.** Unchanged code moves about ±30 questions out
  of 883 between runs. The repair loop's contribution is a net +9, so it is real in direction and
  unproven in size, and a 17-question headline over it says more about the run than the feature.
- **Notation, not arithmetic, is what fails.** The engine replays all 883 gold programs exactly
  (`2026-10-16T16-15-30_replay_dev`, 883 matched, 0 mismatched). The
  remaining errors are picking the wrong evidence and writing a percentage where FinQA stores a
  fraction — 155 questions in this run. Minting a number is rare rather than absent: the check found
  42 replies in D, 4.8% of the split, built entirely out of figures the report does not state, and
  refused them; on the live run it refused 39 and every one of them was wrong already.
- **Retrieval is measured, not solved.** Ranked evidence caps end-to-end accuracy at complete@10 of
  0.90 on FinQA, up from 0.81 before the reranker, and the eight questions no lexical pass can reach
  set a floor under that. FinQA's documents are small enough that the cap is not what is binding here.
  It would bind on a real report, which is the one thing this benchmark cannot test.
- **The reranker's gain is a retrieval result, not an accuracy result.** It was measured on the
  retrieval ceiling, where the instrument is deterministic and needs no model. Its effect on the
  end-to-end numbers has not been measured, because doing so costs a full run and the churn between
  runs of the same configuration is wider than the effect being looked for.
- **The benchmark isn't the product.** FinQA gives you the table already parsed. The PDF path isn't
  covered by it and needs separate evaluation. A good score here says nothing about whether the PDF
  reader works.
- **Extraction is fragile.** Merged cells, footnotes, multi-page tables and scanned documents are
  known failure modes, and none of the extraction path is built yet.
- **The check reads figures, not questions.** It can tell that a number came from nowhere. It cannot
  tell that a question was unanswerable, so a question with no evidence in it can still get an
  arithmetically sound answer assembled from figures that do happen to appear in the report. Real
  abstention — "the report does not say" — is still not built, and a refusal is currently a side
  effect of a figure check rather than a judgement about answerability.
- **No unit or currency conversion.** Figures are never silently converted or inflation-adjusted.
- **English, USD/EUR-heavy financial-report style.** Out-of-domain documents are untested.

## Data

FinQA (Chen et al., EMNLP 2021), splits as published, never reshuffled. Leakage rules and metric
definitions are in `docs/evaluation_protocol.md`. No dataset files are committed.

## Results

Model throughout: `ministral-14b-2512` on Mistral's free tier, temperature 0, 0 failed calls.

| Run | Split | n | Strict | Percent-equivalent | No answer | p95 latency | Cost/question |
|---|---|---|---|---|---|---|---|
| A — whole document in the prompt (direct-v1) | FinQA dev | 883 | 0.2231 | 0.5108 | 0 | 3.37s | $0.0000 |
| D — retrieve, then write a program (program-v3) | FinQA dev | 883 | 0.3296 | 0.5051 | 76 | 3.92s | $0.0000 |
| E — D, plus the figure check and one repair pass | FinQA dev | 883 | 0.3488 | 0.5345 | 78 | 4.75s | $0.0000 |

`Strict` is FinQA's own rule and the headline. `Percent-equivalent` also accepts an answer a hundred
times the reference, because FinQA stores some percentages as fractions; it is a diagnostic, not a
second headline.

Cost is zero because the provider's free tier is free, not because the tokens are free. The runs
record the tokens either way: 1,128,042 for A, 954,589 for D and 1,096,487 for E. E costs 14.9% more
tokens than D and its p95 is 21% higher, because a repair is a second model call.

Nothing appears in this table unless a committed script produced it, and each run's `config.json`
carries its split, question count, provider, model and prompt version: `2026-10-16T15-09-40_direct@mistral_dev`
for A, `2026-10-16T16-15-45_program@mistral_dev` for D and
`2026-10-17T11-25-38_program+repair1@mistral_dev` for E. Those `config.json` and `metrics.json` files
are committed, so every figure above can be checked against the run that produced it.

### E against D, and how much of it is real

E is 17 questions better than D on strict, and 26 better on percent-equivalent. A single run cannot
resolve a difference of that size, and the data says so.

The repair loop only touches a question whose **first** attempt fails, so the 775 questions where it
never fired took the identical code path in both runs: same prompt, same model, same one attempt.
On those questions the answer moved from wrong to right 34 times and from right to wrong 26 times.
Run-to-run variation on unchanged code is therefore around ±30 questions, and the +17 is inside it.
30 of those 60 flips are the strict-versus-percent convention — the same question where the model
writes `93.5` in one run and `0.935` in the other — which is a further reason the strict column should
not be read to the question.

The part that can be attributed to the repair loop is the 108 questions it fired on (12.2%). Ten of
them ended correct where D was wrong or answered nothing; one ended wrong where D had been right. That
is a net of +9 questions, about one point, and it is the honest size of the feature. Nine of the 108
failures it saw were recovered.

What it was shown is worth as much as the number. Most of the 108 are malformed replies rather than
subtly wrong ones: `ProgramSyntaxError: unexpected end of program in 'null'`, a reply carrying no
program at all, or a program made entirely out of `0`. The arithmetic was never the thing to repair.

### What the check catches, and what it costs

The check can be measured without a model, by replaying a finished run's stored replies through it.
That is `python -m evaluation.audit_grounding evaluation/reports/<run>`, and it costs nothing because
no model is called. Replaying run D:

| | |
|---|---|
| verdict `grounded` | 692 |
| `partly_grounded` | 73 |
| `ungrounded` — refused | 42 |
| no program to check | 76 |
| correct answers, check off | 291 of 883 |
| correct answers, check on | 291 of 883 |
| refusals that were right | 0 |
| refusals that were already wrong | 42 |
| answers kept but flagged as resting on an unstated figure | 73, of which 10 right |

On this split the check costs nothing in accuracy and removes 42 answers that were wrong anyway. The
42 are dominated by a single behaviour: a program written entirely out of `0`, which runs cleanly and
means nothing — 26 of the 42. The other 73 flagged answers are the interesting case rather than the
failure case, since 10 of them were right and every one of them is now labelled for the reader.

This bounds the check, not the repair loop beside it: a replay of stored replies never sees a second
attempt. The live run above does, and it puts the same number on it — E refused 39 answers and none of
the 39 was right in D.

## Notes on honest reporting

- A gain smaller than the run-to-run spread is reported as a gain smaller than the spread. E's 17
  questions are inside the ±30 that unchanged code moves by, and the section above says so.
- The one thing that *can* be attributed inside a noisy comparison is the population the feature
  touched, so it is reported that way instead: 108 questions reached the repair loop, 10 improved and 1
  regressed.
- A good benchmark score doesn't mean it's fit for a real investment or lending decision.
- Both scoring conventions are printed next to each other, because the one that flatters the system
  is the one that would otherwise get quoted alone.
- Every failure is counted as a failure. Nothing is dropped between the run and the table.
- A gain that appears in one metric and not another gets written up as exactly that, not as a win.
- No confidence score is presented as correctness unless its calibration is measured.
- Users get told when their content is sent to an external provider.

## Changelog

- 0.3 — verification and the repair loop built and measured: run E added to the results, with its gain
  attributed between the feature and run-to-run noise rather than claimed whole; the offline replay
  instrument documented; the check's three verdicts written into the limitations
- 0.2 — parts table split into built and planned, results filled in from the two dev-split runs,
  limitations rewritten around what the measurements actually showed
- 0.1 — initial card, structure and intended use only, no results
