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

Working run directories are git-ignored. Summary tables get added deliberately, once they're worth
committing.
