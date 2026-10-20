import { ArrowRight, CircleNotch, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

import { fetchBrief, type BriefRow, type SavedBrief as BriefRecord } from "../api";
import { count, formatAnswer, noteOfAnswer, plural, toneOfAnswer, type Verdict } from "../format";
import Claim from "./Claim";
import Mark from "./Mark";

/**
 * A brief opened by link.
 *
 * The page has no controls, and that is deliberate: it is a record, not a document
 * to work on. What it does have is the same verdict wording as the page it was saved
 * from, rendered from the same function, because a shared page that read differently
 * from the page it was shared out of would be worse than no shared page.
 *
 * The figures were recomputed by the server when the link was opened, not carried
 * along with it. Nothing here came from the browser that saved it.
 */

function verdictOf(row: BriefRow): Verdict {
  return {
    answer: row.answer,
    failure: row.note,
    audit: row.verdict
      ? {
          verdict: row.verdict,
          labels: row.labels,
          unsourced: row.unsourced,
          sourced_count: row.sourced_count,
          literal_count: row.literal_count,
          constant_only: false,
          report_unit: null,
          steps: [],
        }
      : null,
  };
}

function savedOn(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

export default function SavedBrief({ id }: { id: string }) {
  const [brief, setBrief] = useState<BriefRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const record = await fetchBrief(id);
        if (!cancelled) setBrief(record);
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  const verdicts = brief ? brief.rows.map(verdictOf) : [];
  const tones = verdicts.map(toneOfAnswer);
  const tally = {
    good: tones.filter((tone) => tone === "good").length,
    warn: tones.filter((tone) => tone === "warn").length,
    bad: tones.filter((tone) => tone === "bad").length,
  };

  return (
    <div className="app app--saved">
      <header className="masthead">
        <a className="brand" href="/">
          <Mark />
          <span>
            Ledger<em>Lens</em>
          </span>
        </a>
        <div className="masthead-end">
          <span className="saved-tag">Saved brief</span>
        </div>
      </header>

      <main className="saved">
        {error ? (
          <div className="saved-missing">
            <WarningCircle size={22} weight="bold" aria-hidden="true" />
            <h1>This brief could not be opened</h1>
            <p>{error}</p>
            <a className="btn" href="/">
              Ask your own questions
            </a>
          </div>
        ) : !brief ? (
          <div className="saved-loading" role="status">
            <CircleNotch className="spin" size={16} weight="bold" aria-hidden="true" />
            <span>Running the programs again</span>
          </div>
        ) : (
          <>
            <span className="label">Brief</span>
            <h1 className="saved-title">{brief.filename}</h1>
            <p className="saved-sub">
              saved {savedOn(brief.created_at)} · {plural(brief.rows.length, "question")}
            </p>

            <div className="tally" aria-label={`${count(brief.rows.length)} questions answered`}>
              <div className="tally-cell">
                <b className="num">{count(brief.rows.length)}</b>
                <span>asked</span>
              </div>
              <div className="tally-cell" data-tone="good">
                <b className="num">{count(tally.good)}</b>
                <span>verified</span>
              </div>
              <div className="tally-cell" data-tone="warn">
                <b className="num">{count(tally.warn)}</b>
                <span>partly</span>
              </div>
              <div className="tally-cell" data-tone="bad">
                <b className="num">{count(tally.bad)}</b>
                <span>no answer</span>
              </div>
            </div>

            <ol className="claims">
              {brief.rows.map((row, at) => (
                <Claim
                  key={`${row.question}-${at}`}
                  question={row.question}
                  figure={row.answer === null ? "\u2014" : formatAnswer(row.answer).shown}
                  note={noteOfAnswer(verdicts[at])}
                  tone={toneOfAnswer(verdicts[at])}
                />
              ))}
            </ol>

            <p className="saved-note">
              Every line was run again from its program against the report as this page loaded. The
              figures are the engine's, and the verdicts are the check's — a link carries the
              programs, not the answers.
            </p>

            <div className="saved-action">
              <a className="btn" href="/">
                Ask your own questions
                <ArrowRight size={14} weight="bold" aria-hidden="true" />
              </a>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
