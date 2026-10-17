# Architecture

## Where the logic lives

The short version: the model reads the question and writes the answer, and that's all it does. It
never supplies a number and never runs code.

```
                 ┌───────────────────────────────┐
                 │  calculation engine           │
                 │  fixed list of operations     │
                 │  the only thing that does maths│
                 └───────────────┬───────────────┘
                                 │
      ┌──────────────────────────┴──────────────────────────┐
      │                                                     │
┌─────▼──────────────────────┐          ┌───────────────────▼─────────┐
│ benchmark track (measured) │          │ product track (separate)     │
│                            │          │                             │
│ FinQA json                 │          │ uploaded PDF                │
│   pre_text / post_text     │          │   extract text and tables   │
│   table (already parsed)   │          │   (PyMuPDF / pdfplumber)    │
│                            │          │                             │
│ pick the right rows/cells  │          │ chunk into evidence records │
└─────────────┬──────────────┘          └───────────────┬─────────────┘
              │                                         │
              └──────────────────┬──────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ hybrid retrieval       │
                    │ keyword (BM25, built)  │
                    │ + dense (planned)      │
                    │ + rerank (planned)     │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ orchestrator           │
                    │ LLM -> structured plan │
                    │ validated, never code  │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ calculation engine     │
                    │ runs allowed ops only  │
                    │ records what it did    │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ verifier               │
                    │ is every figure in the │
                    │ plan one the report    │
                    │ states? if none is,    │
                    │ refuse and say so      │
                    └────────────┬───────────┘
                                 ▼
                        answer + evidence + trace
```

Two things in that drawing are decisions rather than placeholders. The dense leg is TF-IDF followed
by truncated SVD, run in-process, not a vector database: the dev split is 26,834 evidence units, where
an exact search costs milliseconds and a server costs a deployment. The verifier checks figures rather
than whole answers, and it checks them against the entire report rather than the retrieved rows,
because a retrieval miss must not read as an invention.

Only the keyword leg is built. Everything else on the product track is the plan, and the model card
says which parts exist.

## Two layers of calculation

This one is easy to get wrong, so it's worth writing down.

FinQA's `program` field is a small DSL with a fixed operation set:

| Operation | Args | Meaning |
|---|---|---|
| `add` | 2 | sum |
| `subtract` | 2 | difference |
| `multiply` | 2 | product |
| `divide` | 2 | quotient |
| `exp` | 2 | exponent |
| `greater` | 2 | comparison |
| `table_max` | selection | max over a table selection |
| `table_min` | selection | min over a table selection |
| `table_sum` | selection | sum over a table selection |
| `table_average` | selection | mean over a table selection |

Operands are either constants or `#n`, which means "the result of step n". Running a gold `program`
should reproduce `exe_ans` exactly. That's how the engine gets validated with no model involved at
all.

The tools the finished product exposes to the API are a higher layer built out of those primitives:
`get_financial_value`, `calculate_difference`, `calculate_percentage_change`, `calculate_ratio`,
`compare_values`, `calculate_percentage_point_change`.

Worth noting: FinQA has no percentage-change operation. It's written as
`divide(subtract(a, b), b)`. The percentage-change tool is a wrapper around that pattern, and it's
where the unit rules and the percentage vs percentage-point distinction get enforced.

Keeping the two layers apart means the benchmark-facing half can be proven correct before any
natural-language component exists. That's the order this gets built in.

## Components

| Component | Does | Milestone |
|---|---|---|
| dataset loader | read FinQA splits, validate, expose typed examples | 0 |
| calculation engine | run allowed ops on `Decimal`, record traces | 1 |
| evaluation runner | score predictions, capture latency and cost, write reports | 2 |
| ingestion | accept and validate uploads, job status, private storage | 5 |
| extraction | text and table extraction, normalisation, evidence ids | 5 |
| evidence store | persist chunks with page, section, period, unit, location | 3 |
| retrieval | keyword + semantic search, dedupe, rerank | 3 |
| orchestrator | LLM to validated plan, tool selection | 3 |
| verifier | citation existence, figure agreement, arithmetic, abstention | 4 |
| llm adapter | provider-neutral interface, one provider to start | 2 |
| api | FastAPI routes, typed schemas, auth, job status | 6 |
| frontend | React workspace | 6 |

The milestone column is the order these were planned in, written before any of it existed. For what
actually exists today the parts table in `model_card.md` is the one that is kept current, because a
plan is allowed to be optimistic and a parts table is not.

## Answering a benchmark question

1. Load the example by id. Splits stay as published.
2. Pick candidate rows and cells from the example's table, plus pre/post text.
3. Build an evidence set with stable ids.
4. Orchestrator produces a plan: metric, periods, required evidence, operation, unit.
5. Validate the plan. Identifiers have to resolve to retrieved evidence, and periods and units have
   to be compatible.
6. Run it on the calculation engine, get a value and a trace.
7. Check the trace. Every figure in the program is looked up in the whole report and classified as a
   figure the report states, an arithmetic constant, or neither. The citation list the model wrote is
   compared against the rows the figures were actually found in.
8. Return the answer with citations and an audit trail, or refuse when no figure in the program has a
   home in the report.
9. If the program would not run, or the check refused it, send the engine's reason back to the model
   once and run whatever comes back. That is the whole repair mechanism: one round, no arguing.

## Answering a product question

Same as above, except step 2 is hybrid retrieval over the ingested evidence store, and the retrieved
passages carry page and section metadata so the citations are clickable.

## Storage

PostgreSQL, with full-text search for keywords and a vector extension for embeddings.

That is the plan for the product track, and none of it is built. The benchmark track needs no database
at all: FinQA is 883 records read from one JSON file, retrieval is an in-process BM25 index over
26,834 units, and nothing is persisted between runs. A vector index was considered for the dense leg
and rejected at this size — an exact in-process search is faster than the round trip to a server.

| Table | Holds |
|---|---|
| `users` | accounts |
| `documents` | metadata, owner, checksum, status |
| `document_versions` | storage key, parser version |
| `evidence_chunks` | searchable evidence and its metadata |
| `embeddings` | vectors per evidence record |
| `questions` | question history |
| `answers` | answer text, status, model version, latency |
| `answer_citations` | evidence backing each answer |
| `calculation_traces` | operation, inputs, result, unit |
| `evaluation_runs` | dataset version, config, metrics |

Milestone 3 only needs `documents`, `evidence_chunks`, `questions`, `answers` and
`answer_citations`. The rest arrives as the pipeline grows.

A few rules that matter more than they look: foreign keys and indexes on the hot paths, ownership
checked in the backend rather than only hidden in the UI, raw evidence kept separate from generated
summaries, extraction and embedding versions recorded so re-indexing is reproducible, and uploaded
files in private storage with only references in the database.

## What it deliberately doesn't use

- **No LangChain, at least at first.** The pipeline is plain Python with explicit interfaces, so the
  retrieval, tool-calling and calculation safety stay readable. A framework goes in only if it solves
  something specific.
- **No PyTorch or training.** Pretrained embeddings plus a hosted model cover the first release. It
  comes back only if there's an experiment that needs a trained component.
- **No multi-agent setup.** Complexity gets added where measurement justifies it, not before.
