# Contributing

This is a solo project, so there is nothing to contribute to yet: pull requests
are closed by default, and the roadmap is whatever is written down in the
README's plan and the evaluation protocol.

That will change if it stops being a solo project, and if it does, the notes
below are what I would owe an outsider.

## Everything runs with no key

The default provider is `fake`, so a fresh checkout answers end to end without
configuration and without touching anyone's quota. real providers are opt-in
configuration in `.env`. If a change requires a paid key to work at all, that
is a bug in the design, not a trade-off to be defended.

## The two rules behind the design

- **No model output is trusted as a number.** Anything that reaches the answer
  path goes through the engine and the audit. A feature that would let a
  reply's figures reach the card unmolested breaks the project's whole premise,
  and is the reason `backend/app/services/briefs.py` re-runs programs instead of
  accepting figures.
- **What you claim, you measure.** New reasoning is benchmarked against the
  committed runs before it ships to the served path; new retrieval is measured
  on the train split and recorded in `evaluation/reports/`.

## Bugs and questions

Issues are welcome, especially the kind that says what you expected instead of
what happened. Security problems go the way described in `SECURITY.md`, not
into a public issue.
