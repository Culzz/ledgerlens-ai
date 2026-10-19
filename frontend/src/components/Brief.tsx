import { CheckCircle, Circle, CircleNotch, WarningCircle, XCircle } from "@phosphor-icons/react";

import type { Answer, Report } from "../api";
import { count, formatAnswer, plural, trustOf } from "../format";

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

type Tone = "good" | "warn" | "bad";

const MARKS = { good: CheckCircle, warn: WarningCircle, bad: XCircle } as const;

// The row's mark and colour, and the tally above it, both come from this. A strip
// of numbers that disagreed with the lines it summarises would be the kind of
// summary this whole system exists not to produce.
function toneOf(answer: Answer): Tone {
  if (answer.answer === null) return "bad";
  return trustOf(answer.audit)?.tone ?? "bad";
}

// The same words the card uses, because two vocabularies for one verdict is how a
// reader ends up believing there are two verdicts.
function noteOf(answer: Answer): string {
  const trust = trustOf(answer.audit);
  if (trust) return `${trust.text} · ${trust.detail}`;
  return answer.failure ?? "no answer";
}

interface ClaimProps {
  question: string;
  answer: Answer | undefined;
  onOpen: (answer: Answer) => void;
  onAsk: (question: string) => void;
}

function Claim({ question, answer, onOpen, onAsk }: ClaimProps) {
  if (!answer) {
    return (
      <li>
        <button className="claim" data-tone="idle" onClick={() => onAsk(question)}>
          <Circle className="claim-mark" size={15} weight="bold" aria-hidden="true" />
          <span className="claim-q">{question}</span>
          <span className="claim-figure">Ask</span>
          <span className="claim-note">not asked yet</span>
        </button>
      </li>
    );
  }

  const tone = toneOf(answer);
  const Mark = MARKS[tone];
  // The six digits the card shows. The exact figure is on the card, one click
  // away, which is where a reader who wants all thirty digits is going anyway.
  const shown = answer.answer === null ? "\u2014" : formatAnswer(answer.answer).shown;

  return (
    <li>
      <button className="claim" data-tone={tone} onClick={() => onOpen(answer)}>
        <Mark className="claim-mark" size={15} weight="bold" aria-hidden="true" />
        <span className="claim-q">{question}</span>
        <span className="claim-figure num">{shown}</span>
        <span className="claim-note">{noteOf(answer)}</span>
      </button>
    </li>
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
  const rows = Math.max(report.table.length - 1, 0);
  const byQuestion = new Map(asked.map((item) => [item.question, item]));
  const remaining = report.questions.filter((question) => !byQuestion.has(question));
  const extras = asked.filter((item) => !report.questions.includes(item.question));

  const tones = asked.map(toneOf);
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
                <Claim
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
                <Claim
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
