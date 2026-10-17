# reports

Output of evaluation runs.

Anything quoted from this project, in the README or the model card or a writeup, has to come from a
file in here, produced by a script under `evaluation/`. No numbers from anywhere else.

The runner writes a directory per run:

```
reports/
  2026-10-20T14-22-01_finqa-dev_systemD/
    config.json           dataset version, split, model, prompt version, config hash
    predictions.jsonl     one line per question, with latency and cost
    metrics.json          aggregate scores
    error_analysis.json   category distribution
```

## What is committed, and what is not

`evaluation/reports/*` is git-ignored, so a run directory does not land in the repository by accident.
That leaves a gap, and it is worth naming: the README and the model card cite run directories, and a
citation nobody can open is not a citation.

So the two small files that make a number traceable are committed deliberately, by name, for the runs
worth citing:

```
git add -f evaluation/reports/<run-directory>/config.json \
           evaluation/reports/<run-directory>/metrics.json
```

`config.json` names the split, the question count, the provider, the model, the prompt version, the
tolerance and the trust-layer settings. `metrics.json` holds the aggregates. Between them a reader can
check that a quoted figure came from the run that claims it, and under which settings.

The per-question files stay out, and the reason is size rather than secrecy. A prediction record holds
the model's full reply, the program it wrote and the evidence rows it worked from, which across 883
questions is the bulk of a run. The scorer reads the reference answer from the split file, not from
the prediction, so the two small files really are enough to reproduce the aggregate.
