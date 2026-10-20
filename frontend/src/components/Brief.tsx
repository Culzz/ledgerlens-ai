import { CircleNotch } from "@phosphor-icons/react";
import { useState } from "react";

import { saveBrief, type Answer, type Report } from "../api";
import { count, formatAnswer, noteOfAnswer, plural, toneOfAnswer } from "../format";
import Claim from "./Claim";

interface Props {
  report: Report;
  /** Every question answered about this report, in the order it was answered. */
  asked: Answer[];
  offline: boolean;
  /** Set while the brief is answering the report's questions one at a time. */
  progress: { done: number; total: number } | null;
  onOpen: (answer: Answer) => void;
  onAsk: (question: string) => void;
  onPrepare: () => void;
  onStop: () => void;
}

// The row's mark and colour, and the tally above it, both come from this. A strip
// of numbers that disagreed with the lines it summarises would be the kind of
// summary this whole system exists not to produce.
interface LineProps {
  question: string;
  answer: Answer | undefined;
  onOpen: (answer: Answer) => void;
  onAsk: (question: string) => void;
}

function Line({ question, answer, onOpen, onAsk }: LineProps) {
  if (!answer) {
    return (
      <Claim
        question={question}
        figure="Ask"
        note="not asked yet"
        tone="idle"
        onClick={() => onAsk(question)}
      />
    );
  }

  // The six digits the card shows. The exact figure is on the card, one click
  // away, which is where a reader who wants all thirty digits is going anyway.
  const shown = answer.answer === null ? "\u2014" : formatAnswer(answer.answer).shown;

  return (
    <Claim
      question={question}
      figure={shown}
      note={noteOfAnswer(answer)}
      tone={toneOfAnswer(answer)}
      onClick={() => onOpen(answer)}
    />
  );
}

/**
 * One page about a report: what it is, what has been asked of it, and what the
 * system would and would not say.
 *
 * The questions are the report's own, and a line is only a claim once it has an
 * answer, so this is not a summary of the document - it is a record of the
 * document being asked things, with the check's verdict on every line. That is the
 * briefing this project can honestly produce: a model-written paragraph about a
 * financial statement is the one thing it exists to replace.
 */
export default function Brief({
  report,
  asked,
  offline,
  progress,
  onOpen,
  onAsk,
  onPrepare,
  onStop,
}: Props) {
  // A saved brief is a link to a page that re-runs these programs. It is not a
  // snapshot, so the link is only current while it covers the whole session: ask
  // another question and the button offers to save again rather than letting a
  // stale page look complete.
  const [saved, setSaved] = useState<{ id: string; url: string; count: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  async function save() {
    if (asked.length === 0 || saving) return;
    setSaving(true);
    setSaveError(null);
    setCopied(false);
    try {
      const brief = await saveBrief(
        report.id,
        asked.map((item) => ({ question: item.question, program: item.program })),
      );
      const id = brief.id;
      setSaved({ id, url: `${window.location.origin}/b/${id}`, count: asked.length });
    } catch (cause) {
      setSaveError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  }

  async function copy() {
    if (!saved) return;
    try {
      await navigator.clipboard.writeText(saved.url);
      setCopied(true);
    } catch {
      // No clipboard permission. The link is on screen and can be selected.
    }
  }

  const rows = Math.max(report.table.length - 1, 0);
  const byQuestion = new Map(asked.map((item) => [item.question, item]));
  const remaining = report.questions.filter((question) => !byQuestion.has(question));
  const extras = asked.filter((item) => !report.questions.includes(item.question));

  const tones = asked.map(toneOfAnswer);
  const tally = {
    good: tones.filter((tone) => tone === "good").length,
    warn: tones.filter((tone) => tone === "warn").length,
    bad: tones.filter((tone) => tone === "bad").length,
  };

  return (
    <div className="pane-body">
      <div className="brief">
        <h2 className="brief-title">{report.filename}</h2>
        <p className="brief-sub">
          {count(rows)} rows · {count(report.context.length)} lines
        </p>

        <div className="tally" aria-label={`${count(asked.length)} questions answered`}>
          <div className="tally-cell">
            <b className="num">{count(asked.length)}</b>
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

        <p className="brief-note">
          The model wrote the programs; the engine did the arithmetic. A verified line rests only on
          figures the report states, a partly verified one rests on at least one it does not, and a
          line with no answer is one the check would not let through.
        </p>

        {asked.length > 0 && (
          <div className="brief-save">
            {saved && saved.count === asked.length ? (
              <>
                <a className="brief-share" href={saved.url} target="_blank" rel="noreferrer">
                  {saved.url.replace(/^https?:\/\//, "")}
                </a>
                <button className="btn btn--quiet" onClick={() => void copy()}>
                  {copied ? "Copied" : "Copy link"}
                </button>
              </>
            ) : (
              <button className="btn btn--quiet" onClick={() => void save()} disabled={saving}>
                {saving ? "Saving\u2026" : saved ? "Save again" : "Save this brief"}
              </button>
            )}
            {saveError && <span className="brief-offline">{saveError}</span>}
          </div>
        )}

        {progress ? (
          <div className="brief-running" role="status">
            <CircleNotch className="spin" size={14} weight="bold" aria-hidden="true" />
            <span>
              Answering {count(progress.done)} of {count(progress.total)}
            </span>
            <button onClick={onStop}>Stop</button>
          </div>
        ) : (
          remaining.length > 0 && (
            <div className="brief-action">
              <button className="btn" onClick={onPrepare} disabled={offline}>
                Answer the {plural(remaining.length, "remaining question")}
              </button>
              {offline && <span className="brief-offline">No model is configured.</span>}
            </div>
          )
        )}

        {report.questions.length > 0 ? (
          <>
            <span className="label brief-label">Questions this report came with</span>
            <ol className="claims">
              {report.questions.map((question) => (
                <Line
                  key={question}
                  question={question}
                  answer={byQuestion.get(question)}
                  onOpen={onOpen}
                  onAsk={onAsk}
                />
              ))}
            </ol>
          </>
        ) : (
          <>
            <span className="label brief-label">Questions</span>
            <p className="hint">This report came with none. Ask one from the Ask panel.</p>
          </>
        )}

        {extras.length > 0 && (
          <>
            <span className="label brief-label">Also asked</span>
            <ol className="claims">
              {extras.map((item) => (
                <Line
                  key={item.question}
                  question={item.question}
                  answer={item}
                  onOpen={onOpen}
                  onAsk={onAsk}
                />
              ))}
            </ol>
          </>
        )}
      </div>
    </div>
  );
}
