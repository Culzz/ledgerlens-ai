# Verification

Every figure in a program is classified against the whole report: one the report
states, arithmetic the model wrote itself, or neither. That last group is what the
check is for, and it can be measured with no model at all by replaying a finished
run's replies — `audit_grounding` does exactly that, and it is how the check was
measured before it was trusted.

## Measured on run D, replayed

| run D, 883 replies | |
| ---- | ---- |
| `grounded` | 692 |
| `partly_grounded` | 73 |
| `ungrounded`, refused | 42 |
| no program to check | 76 |
| correct answers, check off | 291 of 883 |
| correct answers, check on | 291 of 883 |
| refused answers that were right | 0 |
| refused answers that were already wrong | 42 |

The check cost nothing in accuracy **on this run** — an observation about these
883 replies, not a guarantee about every future answer — and it removed 42 wrong
ones. 26 of the 42 are one behaviour: a program written entirely out of `0`, which
runs cleanly and means nothing.

The 73 `partly_grounded` answers are the more interesting group and they are not
refused. An answer resting on one figure the report does not state is a weaker
answer, not an invented one — 10 of the 73 were right. The audit trail names the
figure either way, so a reader can see which one it was.

## Live measurement in run E

A replay of stored replies never sees the repair pass, because the second attempt
was not part of run D's stored history. Run E — the same system plus the figure
check and one repair pass — is the live measurement, and it puts its own number on
the check: 39 refusals, none of them right. (D refused 42; the runs are neighbours
in configuration and close in count, but they are separate runs.)

## What grounding does and does not mean

Grounding means the report supports the figures the program used. It does *not*
mean the program used the right figures, paired the right years, or picked the
right formula — a fully grounded program can still be the wrong program. The
three-way split (grounding / calculation correctness / answer correctness) is
spelled out in `docs/evaluation.md`; the strict score there is the number that
measures all three together, and it is where the remaining error lives.

What the check buys is that the class of answer built out of numbers the report
never states is refused at the door instead of being scored wrong after the fact —
and on the runs above, every answer it refused was already wrong, so the door cost
nothing.
