# Model card

What's in here, what it's for, and where it breaks.

Version 0.2. The numbers below come from runs under `evaluation/reports/`, and none were written
before those runs existed.

## What it actually is

Not a single trained model. A small system made of parts, and not all the parts are there yet:

| Part | Choice | State |
|---|---|---|
| LLM | provider-neutral adapter over the OpenAI chat-completions shape | built. Reads the question, writes the program or the answer. Never a source of numbers |
| keyword search | in-process BM25 over table rows and narrative lines | built |
| calculation | Python `Decimal` with a fixed allowlist of operations | built |
| evaluation | pandas and pytest | built |
| verification | checks that the citations exist and the figures agree | not built |
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

## Limitations

- **Accuracy.** A third of FinQA's dev split under its own strict rule, with a small model. GPT-4 with
  the table inline is around 65% in the literature. This is not claiming to blow past that, and it
  does not.
- **A reply has to become a program.** 8.6% of questions came back as a reply the engine could not
  run, mostly the model naming an evidence label where a value belongs. Those are counted as no
  answer, not as wrong answers.
- **Notation, not arithmetic, is what fails.** The engine replays all 883 gold programs exactly. The
  remaining errors are picking the wrong evidence and writing a percentage where FinQA stores a
  fraction — 155 questions in this run. Minting a number is not the failure mode.
- **Retrieval is measured, not solved.** Ranked evidence caps end-to-end accuracy at complete@10 of
  0.81 on FinQA, and FinQA's documents are small enough that the cap is not what is binding here. It
  would bind on a real report, which is the one thing this benchmark cannot test.
- **The benchmark isn't the product.** FinQA gives you the table already parsed. The PDF path isn't
  covered by it and needs separate evaluation. A good score here says nothing about whether the PDF
  reader works.
- **Extraction is fragile.** Merged cells, footnotes, multi-page tables and scanned documents are
  known failure modes, and none of the extraction path is built yet.
- **Abstention is not built.** A question with no evidence in it currently gets an answer attempt
  rather than a refusal. The 76 unexecutable replies are where that shows.
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

`Strict` is FinQA's own rule and the headline. `Percent-equivalent` also accepts an answer a hundred
times the reference, because FinQA stores some percentages as fractions; it is a diagnostic, not a
second headline.

Cost is zero because the provider's free tier is free, not because the tokens are free. The run
records the tokens either way: 1,128,042 for A and 954,589 for D.

Nothing appears in this table unless a committed script produced it, and each run's `config.json`
carries its split, question count, provider, model and prompt version.

## Notes on honest reporting

- A good benchmark score doesn't mean it's fit for a real investment or lending decision.
- Both scoring conventions are printed next to each other, because the one that flatters the system
  is the one that would otherwise get quoted alone.
- Every failure is counted as a failure. Nothing is dropped between the run and the table.
- A gain that appears in one metric and not another gets written up as exactly that, not as a win.
- No confidence score is presented as correctness unless its calibration is measured.
- Users get told when their content is sent to an external provider.

## Changelog

- 0.2 — parts table split into built and planned, results filled in from the two dev-split runs,
  limitations rewritten around what the measurements actually showed
- 0.1 — initial card, structure and intended use only, no results
