// Formatting shared by the panels, so a figure is set the same way wherever it
// turns up.

import type { Answer, Audit } from "./api";

// A cell that holds a figure rather than a label. Currency marks, thousands
// separators, a trailing percent, parentheses for a negative: all the ways a
// statement writes a number.
const FIGURE = /^[([\]]?\s*[$€£¥]?\s*[-+\u2013\u2014]?\s*\d[\d,\s.]*%?\s*[)\]]?$/;

// A lone dash is how a statement says nothing was owed, and it sits flush right
// with the figures it belongs to.
const NIL = /^[\u2013\u2014-]$/;

export function isFigure(value: string): boolean {
  const cell = value.trim();
  if (!cell) return false;
  return NIL.test(cell) || FIGURE.test(cell);
}

export function count(value: number): string {
  return value.toLocaleString("en-US");
}

// Cost is reported as unknown rather than zero when the price of the model is
// not known, and a free tier is named rather than shown as $0.0000, because
// those are three different facts.
export function formatCost(cost: number | null): string {
  if (cost === null) return "cost unknown";
  if (cost === 0) return "free tier";
  return `$${cost.toFixed(4)}`;
}

export function formatSeconds(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(2)} s`;
}

export function plural(value: number, one: string, many = `${one}s`): string {
  return `${value} ${value === 1 ? one : many}`;
}

// The engine works in exact decimal, so a ratio comes back as thirty digits. Six
// significant digits is what a person reads; the untouched string is handed back
// as well so the panel can show it underneath rather than throw it away.
export function formatAnswer(value: string): { shown: string; exact: string | null } {
  const trimmed = value.trim();
  if (!/^-?\d+(\.\d+)?$/.test(trimmed)) return { shown: trimmed, exact: null };

  const exactNumber = Number(trimmed);
  if (!Number.isFinite(exactNumber)) return { shown: trimmed, exact: null };

  const digits = 6;
  const shown = new Intl.NumberFormat("en-US", { maximumSignificantDigits: digits }).format(
    exactNumber,
  );
  const rounded = Number(exactNumber.toPrecision(digits));
  return { shown, exact: rounded === exactNumber ? null : trimmed };
}

// The badge a reader looks at first, and the sentence behind it. The three
// verdicts are three different claims and the wording keeps them apart: "all the
// figures were in the report" is not the same statement as "none of them were".
export interface Trust {
  text: string;
  tone: "good" | "warn" | "bad";
  detail: string;
}

export function trustOf(audit: Audit | null): Trust | null {
  if (!audit) return null;

  const figures = (n: number) => plural(n, "figure");
  switch (audit.verdict) {
    case "grounded":
      return {
        text: "Verified audit",
        tone: "good",
        detail:
          audit.literal_count === 0
            ? "the program names rows and columns rather than writing figures, so there was no figure to check"
            : `all ${figures(audit.sourced_count)} appear in the report`,
      };
    case "partly_grounded":
      return {
        text: "Partly verified",
        tone: "warn",
        detail: `rests on ${figures(audit.unsourced.length)} the report does not state (${audit.unsourced.join(", ")})`,
      };
    default:
      return {
        text: "Unverified",
        tone: "bad",
        detail: `no figure in this program appears in the report (${audit.unsourced.join(", ")})`,
      };
  }
}

// A row's mark and the sentence under it. Shared by the page a reader works on and
// a saved brief opened by link, so the two cannot drift into two vocabularies.
export type Tone = "good" | "warn" | "bad" | "idle";

/** What a served answer and a stored brief line have in common. */
export interface Verdict {
  answer: string | null;
  audit: Audit | null;
  failure: string | null;
}

export function toneOfAnswer(item: Verdict): Tone {
  if (item.answer === null) return "bad";
  return trustOf(item.audit)?.tone ?? "bad";
}

export function noteOfAnswer(item: Verdict): string {
  const trust = trustOf(item.audit);
  if (trust) return `${trust.text} · ${trust.detail}`;
  return item.failure ?? "no answer";
}

// The answer as markdown, with the citations intact. Bounded by a pipe-free form
// of the step, because a row label can contain anything and a table row cannot.
export function toMarkdown(answer: Answer, source: string | null): string {
  const lines: string[] = [
    `**Question.** ${answer.question}`,
    "",
    answer.answer !== null
      ? `**Answer.** ${answer.answer}`
      : `**Answer.** none. ${answer.failure ?? "The program could not be run."}`,
  ];

  const trust = trustOf(answer.audit);
  if (trust) {
    lines.push("", `**${trust.text}.** ${trust.detail}`);
  }
  if (source) {
    lines.push("", `_Source: ${source}._`);
  }
  if (answer.program) {
    lines.push("", "```", answer.program, "```");
  }

  const steps = answer.audit?.steps ?? [];
  if (steps.length > 0) {
    lines.push("", "| # | working |", "| --- | --- |");
    for (const step of steps) {
      lines.push(`| ${step.index + 1} | \`${step.render.replace(/\|/g, "\\|")}\` |`);
    }
  }

  return lines.join("\n");
}
