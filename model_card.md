# Model card

What's in here, what it's for, and where it breaks.

Version 0.1. No evaluation results yet. This gets filled in from runs under `evaluation/`, not before.

## What it actually is

Not a single trained model. A small system made of parts:

| Part | Choice | Why |
|---|---|---|
| LLM | provider-neutral adapter, OpenAI to start | reads the question, writes the answer. Not a source of numbers |
| embeddings | local sentence-transformers, MiniLM class | semantic search with no per-call cost |
| keyword search | PostgreSQL full-text search | exact terms, periods, abbreviations |
| reranking | optional cross-encoder | precision on the final evidence set |
| table parsing | PyMuPDF plus pdfplumber | text and tables from born-digital PDFs |
| calculation | Python `Decimal` with a fixed allowlist | exact arithmetic, no float drift |
| evaluation | pandas and pytest | accuracy, retrieval metrics, error analysis |

## What it's for

For analysts, investors, researchers and finance teams poking at financial reports. Also, honestly,
as a portfolio and research project about evidence-grounded numerical reasoning.

Not for: investment advice, autonomous transactions, credit or lending decisions, audit reliance, or
anything where an unverified number has real consequences.

## How it works, and what follows from that

The model reads the question and proposes a plan. A deterministic engine does the arithmetic on
values pulled from stored evidence. A verifier checks that the citations exist and the figures agree.
If the evidence isn't there, it refuses.

The consequence is that the model is never the source of truth for a number. That buys
faithfulness and auditability. It doesn't by itself buy accuracy.

## Limitations

- **Accuracy ceiling.** GPT-4 with the table is around 65% on FinQA in the literature. This project
  isn't claiming to blow past that.
- **Retrieval is the bottleneck.** Published end-to-end pipelines show big drops from module to
  pipeline (FinQAPT: 80.6% to 62%). Retrieval failures are expected to dominate the error
  distribution, and that's the thing to work on.
- **The benchmark isn't the product.** FinQA gives you the table already parsed. The PDF path isn't
  covered by it and needs separate evaluation.
- **Extraction is fragile.** Merged cells, footnotes, multi-page tables and scanned documents are
  known failure modes.
- **Ambiguity.** A genuinely ambiguous row is treated as unresolved rather than guessed, which will
  sometimes mean refusing a question it could have answered.
- **No unit or currency conversion.** Figures are never silently converted or inflation-adjusted.
- **English, USD/EUR-heavy financial-report style.** Out-of-domain documents are untested.

## Data

FinQA (Chen et al., EMNLP 2021), splits as published. Leakage rules and metric definitions are in
`docs/evaluation_protocol.md`.

## Results

To be filled in from `evaluation/reports/`.

| Run | Split | System | Exec acc | Exact match | p95 latency | Cost/question |
|---|---|---|---|---|---|---|
| | | | | | | |

Nothing appears in this table unless a committed script produced it.

## Notes on honest reporting

- A good benchmark score doesn't mean it's fit for a real investment or lending decision.
- Failures get reported next to successes. Unsupported-claim and abstention rates sit beside accuracy.
- No confidence score is presented as correctness unless its calibration is measured.
- Users get told when their content is sent to an external provider.

## Changelog

- 0.1 - initial card, structure and intended use only, no results
