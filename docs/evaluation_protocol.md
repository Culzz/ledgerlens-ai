# Evaluation protocol

Written before implementation, on purpose. The point is that results can't get reverse-engineered
from whatever the system happens to produce, and that comparisons against baselines are fair.

Version 0.1. No results recorded yet.

## The question

Does combining evidence retrieval with deterministic calculation tools improve numerical accuracy
and evidence faithfulness in financial question answering, compared with direct prompting and
conventional RAG?

Sub-questions:

- how much does retrieval quality affect the final answer
- does citing sources make answers easier to verify
- how well does it tell when a question can't be answered from the evidence
- what does each extra component cost in latency and money

## Dataset

FinQA (Chen et al., EMNLP 2021). Original train/dev/test splits, used as published.

Each record looks like this. Note that the answer fields are nested under `qa`, which is easy to
miss:

```jsonc
{
  "pre_text":  ["..."],                    // prose before the table
  "post_text": ["..."],                    // prose after the table
  "table":     [["..."]],                  // already parsed
  "id":        "ETR/2016/page_23.pdf-2",
  "qa": {
    "question":   "...",
    "program":    "subtract(5829, 5735)",  // the reasoning DSL
    "gold_inds":  { },                     // supporting facts
    "exe_ans":    94,                      // reference answer
    "program_re": "..."                    // nested form
  }
}
```

| Split | Use |
|---|---|
| train | prompt examples / few-shot selection only |
| dev | development and all baseline comparisons |
| test | held out. Run once, at a milestone boundary, for the headline number |

### Leakage rules

Not negotiable.

1. Never randomly recombine splits.
2. Never train or tune on test answers.
3. `exe_ans`, `program` and `gold_inds` are not visible to the inference pipeline. The scorer joins
   them back on afterwards.
4. Few-shot examples come from train only.
5. Prompt iteration happens on dev. Touching test while iterating invalidates the result.
6. Every run records dataset version, split, config hash, model name and version, and prompt version.

## Systems compared

Same questions, same model, same source context where it applies.

| ID | System | Retrieval | Arithmetic |
|---|---|---|---|
| A | direct LLM | none, whole table in the prompt | model does it |
| B | retrieval-augmented LLM | hybrid | model does it |
| C | hybrid RAG | hybrid + rerank | model does it |
| D | LedgerLens | hybrid + rerank | deterministic engine + verifier |

Built so far: A is `--system direct`, and D is `--system program`, which runs retrieval and the
engine but still has lexical retrieval where the table says hybrid plus rerank. B and C are the
middle rungs and are not built, because once the engine is in place a run that hands retrieved rows
back to the model for it to do the arithmetic itself measures less than the run next to it costs.

Baseline A has to get the table in context. If it's set up to fail, the whole comparison is
worthless. For scale, the numbers that already exist in the literature: GPT-4 with the table is
around 65%, GPT-4 with chain-of-thought style prompting gets into the 70s, and FinQAPT reported
80.6% at the module level but 62% end to end. Those are recorded here before we run anything so the
results can be placed honestly rather than spun.

## Metrics

| Metric | Meaning |
|---|---|
| execution accuracy | final number matches the reference within tolerance |
| exact match | normalised answers are identical |
| evidence recall@k | the annotated supporting evidence is in the top k |
| evidence precision@k | how much of what was retrieved is relevant |
| citation correctness | cited sources actually support the claim |
| unsupported claim rate | how often it asserts things the evidence doesn't back |
| abstention quality | correct refusals, and correct non-refusals |
| latency | end to end, including retrieval and model calls |
| cost per question | model and compute cost under the measured config |

Tolerance is fixed in advance, default `1e-3` following FinQA's convention, and recorded with the
run. Comparisons are paired across systems on identical questions, with confidence intervals where
they're meaningful.

## Error categories

Each failure gets one primary category, so effort goes where it's actually needed:

`wrong_evidence_retrieved`, `correct_evidence_wrong_figures`, `period_confused`,
`unit_or_scale_mismatch`, `arithmetic_error`, `percentage_vs_percentage_point`,
`unsupported_inference`, `fabricated_citation`, `failed_to_abstain`, `extraction_error`

One of these is decided mechanically rather than by reading: a program that will not run is an
`unexecuted` answer, counted apart from an unreadable reply. It is the characteristic failure of the
engine-based system, and it means the model answered in the right shape and misread the document,
which is a different thing to fix than a model that would not answer at all.

## Reporting rules

1. No result without a command. Every number comes from a script under `evaluation/` and is stored
   under `evaluation/reports/`.
2. Hand-picked examples never stand in for aggregate performance.
3. Failures get reported alongside successes. Accuracy, abstention and unsupported-claim rate travel
   together.
4. No invented targets. Baselines establish how hard the dataset actually is, first.
5. No "confidence score" presented as correctness unless its calibration is defined and measured.
   Retrieval similarity is not answer correctness.
6. The split is stated on every figure. Dev and test results are never mixed silently.

## Planned experiments

1. Replay every gold `program` and report exact match against `exe_ans`, broken down by step count.
   This validates the engine with no model in the loop.
2. Systems A, B and C on dev: accuracy, latency, cost.
3. System D on dev, paired against A, B and C.
4. Error analysis: category distribution per system.
5. Ablations: retrieval strategy, reranking, the verifier, abstention threshold.
6. Best configuration on test, once.

## Extensions

These are where the project can say something a basic RAG demo can't.

**Numerical faithfulness.** Do the answer, the figures it selected, the arithmetic and the citations
all agree with each other? Errors get reported by kind instead of collapsed into "wrong".

**Evidence sufficiency and abstention.** Missing periods, conflicting figures, incomplete tables,
questions that can't be answered from what was supplied. Measures both the false answers and the
unnecessary refusals.

**Adaptive retrieval and computation.** Route by question type, so a simple lookup doesn't cost the
same as a multi-step calculation. Reports the accuracy, latency and cost trade-off.
