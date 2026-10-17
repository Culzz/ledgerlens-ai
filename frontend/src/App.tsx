import { useEffect, useState } from "react";

import {
  ask,
  fetchDocuments,
  fetchHealth,
  fetchReport,
  type Answer,
  type Health,
  type Report,
} from "./api";
import AnswerCard from "./components/AnswerCard";
import ReportPicker from "./components/ReportPicker";
import ReportView from "./components/ReportView";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Open a report to begin with, so the page is not an empty form.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [status, listing] = await Promise.all([fetchHealth(), fetchDocuments()]);
        if (cancelled) return;
        setHealth(status);
        if (listing.documents.length > 0) {
          setReport(await fetchReport(listing.documents[0].id));
        }
      } catch (cause) {
        if (!cancelled) setError(describe(cause));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function openReport(id: string) {
    setAnswer(null);
    setError(null);
    setQuestion("");
    try {
      setReport(await fetchReport(id));
    } catch (cause) {
      setError(describe(cause));
    }
  }

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !report || busy) return;

    setQuestion(trimmed);
    setBusy(true);
    setAnswer(null);
    setError(null);
    try {
      setAnswer(await ask(report.id, trimmed));
    } catch (cause) {
      setError(describe(cause));
    } finally {
      setBusy(false);
    }
  }

  const cited = new Set(answer?.evidence.filter((item) => item.cited).map((item) => item.label));
  const offered = new Set(answer?.evidence.map((item) => item.label));
  const offline = health?.provider === "fake";

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>
            Ledger<span>Lens</span>
          </h1>
          <p className="tagline">
            Ask a question about a financial report. The model writes a program, a calculation
            engine runs it against the report's own figures, and the answer arrives with the steps
            and the rows behind it.
          </p>
        </div>
        {health && (
          <div className="badge">
            <strong>{health.provider}</strong>
            {health.model ? ` · ${health.model}` : ""} · {health.documents} reports
          </div>
        )}
      </header>

      {error && <div className="error" style={{ marginBottom: 18 }}>{error}</div>}

      {health && !health.documents_loaded && (
        <div className="error" style={{ marginBottom: 18 }}>
          No documents are loaded. Run <code>python scripts/download_data.py</code> to fetch the
          FinQA split, then reload.
        </div>
      )}

      <div className="layout">
        <div>
          <div className="card">
            <h2>Report</h2>
            <ReportPicker selected={report?.id ?? null} onSelect={openReport} disabled={busy} />
          </div>

          {report && (
            <div className="card">
              <h2>Ask</h2>
              <div className="chips">
                {report.questions.length > 0 ? (
                  report.questions.map((item) => (
                    <button
                      key={item}
                      className="chip"
                      onClick={() => void submit(item)}
                      disabled={busy}
                    >
                      {item}
                    </button>
                  ))
                ) : (
                  <p className="hint">No questions came with this report. Type one below.</p>
                )}
              </div>

              <form
                className="ask-row"
                onSubmit={(event) => {
                  event.preventDefault();
                  void submit(question);
                }}
              >
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Or ask something of your own about this report"
                  aria-label="Your question"
                />
                <button type="submit" disabled={busy || !question.trim()}>
                  {busy ? "Working…" : "Ask"}
                </button>
              </form>
            </div>
          )}

          {report && <ReportView report={report} cited={cited} offered={offered} />}
        </div>

        <div>
          {busy && (
            <div className="card">
              <h2>Answer</h2>
              <p className="blurb">Retrieving the rows, then writing a program…</p>
            </div>
          )}

          {!busy && answer && <AnswerCard answer={answer} offline={offline} />}

          {!busy && !answer && (
            <div className="card">
              <h2>How this works</h2>
              <p className="blurb">
                The model never supplies the number. It reads the retrieved rows and lines and
                writes a program in a small DSL — <code>subtract(1200, 900)</code>,{" "}
                <code>table_sum(revenue, none)</code> — which the engine then runs against the
                report. That is why an answer can be shown with its arithmetic instead of asserted.
              </p>
              <p className="blurb">
                It is also why failure is visible. A program naming a row the table does not have
                returns no answer at all, with the reason, rather than a plausible figure.
              </p>
              <p className="hint">Pick a question on the left to see one.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function describe(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}
