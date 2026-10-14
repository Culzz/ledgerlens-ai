# Requirements

Baseline spec, written before implementation so "done" means something specific.

Version 0.1.

## Scope

In scope for the first release:

- PDF upload and processing
- text and table extraction from financial documents
- document and evidence indexing
- financial question answering
- deterministic arithmetic, ratios and percentage calculations
- evidence-linked answers with citations
- benchmark evaluation on FinQA
- document history and query history
- REST API and a web UI
- tests, containerisation, deployment, docs

Out of scope, at least for now:

- live stock prices or trading
- investment recommendations
- autonomous financial transactions
- credit or lending decisions
- letting the LLM execute arbitrary code
- enterprise accounting integrations
- guaranteeing extraction from every scanned or badly formed PDF
- fine-tuning a model from scratch

## Two tracks

Worth stating in the requirements, because it changes what can be verified.

| Track | Input | Evaluated by | Status |
|---|---|---|---|
| Benchmark | FinQA pre-parsed table and text | execution accuracy against the reference answer | measured |
| Product | uploaded PDF, extracted here | separate documented corpus | not measured yet |

FinQA supplies tables that are already parsed, so it does not exercise PDF extraction at all. A
benchmark score says nothing about ingestion quality, and the two must not be reported as though
they're the same thing.

## Users

- **Analyst** - uploads reports, asks questions, inspects sources and calculations, saves analyses.
- **Admin** - manages users, processing jobs, document lifecycle, config, logs.
- **Developer / evaluator** - runs benchmark experiments, compares model versions, reviews failures,
  generates reports. For the MVP this is a CLI rather than a dashboard.

## Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Users can upload supported financial PDFs | must |
| FR-02 | File type, size and upload status are validated | must |
| FR-03 | Text, tables and source metadata are extracted | must |
| FR-04 | Users can see processing status and failures | must |
| FR-05 | Evidence is indexed for retrieval | must |
| FR-06 | Users can ask questions about a chosen document | must |
| FR-07 | Evidence is retrieved before an answer is generated | must |
| FR-08 | Calculations run through deterministic tools | must |
| FR-09 | Answers include supporting evidence and calculation detail | must |
| FR-10 | The system can decline when evidence is insufficient | must |
| FR-11 | Users can review past questions and answers | should |
| FR-12 | Users can export an analysis with its citations | should |
| FR-13 | Developers can run the FinQA evaluation pipeline | must |
| FR-14 | Developers can compare baseline and proposed results | must |
| FR-15 | Admins can inspect processing and application errors | should |
| FR-16 | Users can delete documents and derived data | must |

## Non-functional requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Numerical correctness | deterministic tests pass for supported operations and edge cases |
| NFR-02 | Traceability | every factual answer has verifiable sources, or is marked unsupported |
| NFR-03 | Security | uploads validated, access control enforced, secrets out of source |
| NFR-04 | Reproducibility | fixed dataset versions, recorded config, repeatable eval scripts |
| NFR-05 | Reliability | a failed job reports an actionable error without corrupting the record |
| NFR-06 | Maintainability | ingestion, retrieval, reasoning, calculation and API stay separable |
| NFR-07 | Performance | p95 under 10s on a defined local workload, excluding big document ingestion |
| NFR-08 | Observability | request duration, retrieval stats, model version, token usage, error categories |
| NFR-09 | Privacy | no access to another user's documents or history |
| NFR-10 | Portability | runs locally from documented setup, and via Docker Compose |

NFR-07 is a goal, not a measurement. It gets measured under a documented workload and revised if the
chosen provider or machine can't hit it.

## MVP acceptance

The MVP is done when all of these are true:

- [ ] a PDF uploads and ends with a completed or clearly failed status
- [ ] a question can be answered from an indexed document without hardcoding the answer
- [ ] arithmetic runs in a tested calculation engine
- [ ] the answer shows the source passage or table row it used
- [ ] an unanswerable question does not produce an invented figure dressed up as fact
- [ ] `python -m evaluation.run_benchmarks` produces a repeatable accuracy report
- [ ] that report includes retrieval performance on a held-out split
- [ ] tests cover calculations, retrieval, permissions, API validation and failure handling
- [ ] the app runs from scratch by following the README

Those matter more than how many screens the frontend has.

## Traceability

Each functional requirement maps to a milestone in the README and to at least one test under
`backend/tests/`.
