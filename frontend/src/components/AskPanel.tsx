import { ArrowRight, CircleNotch } from "@phosphor-icons/react";
import { useState } from "react";

import type { Answer, Report } from "../api";
import AnswerCard from "./AnswerCard";
import Blank, { NibArt, SheetArt } from "./Blank";

interface Props {
  report: Report | null;
  answer: Answer | null;
  busy: boolean;
  /** True while the brief is working through the report's questions. */
  blocked: boolean;
  offline: boolean;
  onSubmit: (question: string) => void;
  onBrowse: () => void;
}

export default function AskPanel({
  report,
  answer,
  busy,
  blocked,
  offline,
  onSubmit,
  onBrowse,
}: Props) {
  // The draft lives here rather than in App, and App gives this component a key
  // of the report id, so opening another report starts from an empty box without
  // anything having to remember to clear it.
  const [draft, setDraft] = useState("");
  const canAsk = report !== null && !busy && !blocked && draft.trim() !== "";

  function send(event: React.FormEvent) {
    event.preventDefault();
    if (!canAsk) return;
    onSubmit(draft.trim());
    setDraft("");
  }

  return (
    <>
      <div className="pane-head">
        <span className="label">Ask</span>
        {report && <span className="meta">{report.filename}</span>}
      </div>

      {!report ? (
        <Blank
          art={<SheetArt />}
          title="No report open"
          action={
            <button className="btn" onClick={onBrowse}>
              Browse the library
            </button>
          }
        >
          A question has to be about something. Open a report and the questions that came with it
          are listed here.
        </Blank>
      ) : (
        <div className="pane-body">
          <form className="ask-form" onSubmit={send}>
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask your own question about this report"
              aria-label="Your question"
              autoComplete="off"
            />
            <button className="btn" type="submit" disabled={!canAsk}>
              {busy ? <CircleNotch className="spin" size={18} weight="bold" /> : "Ask"}
            </button>
          </form>

          {offline && (
            <div className="notice notice--warn">
              <span>
                No model is configured, so this instance is running on the offline stub. It will
                retrieve the evidence and then decline to answer.
              </span>
            </div>
          )}

          {blocked && (
            <div className="notice">
              <span>
                The brief is working through this report's questions one at a time. This box wakes
                up when it has finished, so a second question does not go out alongside it.
              </span>
            </div>
          )}

          {busy && (
            <div className="block" aria-live="polite">
              <span className="label">Retrieving rows, then writing a program</span>
              <div className="skeleton">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}

          {!busy && answer && (
            <AnswerCard
              // Keyed by the answer, so a what-if does not carry over onto a new
              // question. Moving a figure changes what is on screen, and it should
              // go away the same way the answer it belonged to does.
              key={`${answer.question}|${answer.program ?? ""}`}
              answer={answer}
              documentId={report.id}
              source={report.filename}
              offline={offline}
            />
          )}

          {!busy && !answer && (
            <>
              {report.questions.length > 0 ? (
                <div className="block suggestions">
                  <span className="label">Questions that come with this report</span>
                  <ul>
                    {report.questions.map((question) => (
                      <li key={question}>
                        <button className="suggestion" onClick={() => onSubmit(question)}>
                          <ArrowRight size={13} weight="bold" />
                          <span>{question}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <Blank art={<NibArt />} title="Nothing queued up">
                  This report did not come with any questions. Type one above.
                </Blank>
              )}

              <div className="block">
                <span className="label">How this works</span>
                <p className="blurb">
                  The model never supplies the number. It reads the retrieved rows and lines, writes
                  a program in a small DSL — <code>subtract(1200, 900)</code>,{" "}
                  <code>table_sum(revenue, none)</code> — and the engine runs that against the
                  report.
                </p>
                <p className="blurb">
                  That is why an answer can be shown with its arithmetic instead of asserted, and
                  why a program that names a row the table does not have comes back with no answer
                  rather than a plausible figure.
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
