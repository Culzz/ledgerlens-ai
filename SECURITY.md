# Security

## Reporting a problem

Open a private security advisory on the repository
(Security → Advisories → Report a vulnerability) rather than a public issue. If
that is unavailable, email the address on my GitHub profile with the steps to
reproduce it. I will confirm receipt and say whether and when a fix is coming.

There is no bug bounty.

## What this deployment is

A single public demo on Vercel. It has no accounts, no cookies and no uploads,
so there is nothing to log into and nothing to take over.

## What it does hold

- **A model provider key.** Read from an environment variable at run time and
  never sent to the browser. Keys are not in the repository and never have been;
  `.env` is ignored and `.env.example` carries empty placeholders only.
- **Asked questions and the programs that answered them.** Saved briefs are
  stored in Postgres and readable by anyone holding the link. They contain no
  personal data — a brief is a question about a public financial report.
- **The FinQA dev split.** Downloaded at build time from the FinQA authors'
  repository, not committed. See `data/README.md`.

## What it deliberately does not do

- It does not accept file uploads. The PDF ingestion track is not built, so
  there is no upload path to harden.
- It does not execute anything sent to it. The program language is parsed by a
  small grammar in `backend/app/services/calculations/`; there is no `eval` and
  no shell.
- It does not call a model to serve a saved brief, a recompute or a workbook.

## Limits worth knowing

- The share link is unlisted, not access-controlled. Ids are 72 bits of
  randomness, so they are not enumerable, but anyone with the link can read the
  brief. Do not save anything you would not put on a public page.
- A public model endpoint costs money per call. The app paces itself and backs
  off on a rate limit, but a determined caller can still spend the owner's free
  tier. That is the accepted trade for having a demo anyone can try.
