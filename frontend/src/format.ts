// Formatting shared by the panels, so a figure is set the same way wherever it
// turns up.

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
