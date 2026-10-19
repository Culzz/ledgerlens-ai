import {
  Article,
  Books,
  ChatCircleDots,
  FileText,
  Info,
  Rows,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";

import {
  ask,
  fetchDocuments,
  fetchHealth,
  fetchReport,
  type Answer,
  type Health,
  type Report,
} from "./api";
import AskPanel from "./components/AskPanel";
import Blank, { SheetArt } from "./components/Blank";
import Brief from "./components/Brief";
import Library from "./components/Library";
import Mark from "./components/Mark";
import ReportReader from "./components/ReportReader";
import { count } from "./format";

// The three regions of the app. Wide screens show all of them as columns; narrow
// ones show one at a time with the tab bar, which is the same app rather than a
// reduced version of it.
type View = "library" | "report" | "ask";

// Spacing only, and only at desk widths - see the compact block in styles.css.
type Density = "comfortable" | "compact";

const DENSITY_KEY = "ledgerlens.density";

const TABS = [
  { id: "library" as const, label: "Library", Icon: Books },
  { id: "report" as const, label: "Report", Icon: FileText },
  { id: "ask" as const, label: "Ask", Icon: ChatCircleDots },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("library");

  // Answers to every question asked about the open report, so the brief is a
  // record of the session rather than only of the last thing asked.
  const [asked, setAsked] = useState<Answer[]>([]);
  // Which half of the report pane is showing: the document, or the page about it.
  const [sheet, setSheet] = useState<"source" | "brief">("source");
  const [preparing, setPreparing] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  // A run stops after the question already in flight. That one cannot be recalled,
  // and pretending otherwise would be a lie about what was spent.
  const stop = useRef(false);
  // Remembered across visits: a reader who wants a dense table wants it tomorrow
  // too, and a setting that resets on reload is a setting nobody uses.
  const [density, setDensity] = useState<Density>(() =>
    localStorage.getItem(DENSITY_KEY) === "compact" ? "compact" : "comfortable",
  );

  useEffect(() => {
    document.documentElement.dataset.density = density;
    localStorage.setItem(DENSITY_KEY, density);
  }, [density]);

  // Open a report to begin with, so a first visit lands on a real table with real
  // questions rather than on three empty panels.
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
    setAsked([]);
    setSheet("source");
    setError(null);
    setView("report");
    try {
      setReport(await fetchReport(id));
    } catch (cause) {
      setError(describe(cause));
    }
  }

  function remember(result: Answer) {
    setAsked((current) => {
      const at = current.findIndex((item) => item.question === result.question);
      if (at === -1) return [...current, result];
      const next = [...current];
      next[at] = result;
      return next;
    });
  }

  // One question, answered and remembered. `where` is the view to land on: asking
  // from the brief still ends with the answer card in front of the reader.
  async function answerQuestion(question: string, where: View = "ask") {
    if (!report || busy || preparing) return;
    setBusy(true);
    setAnswer(null);
    setError(null);
    setView(where);
    try {
      const result = await ask(report.id, question);
      setAnswer(result);
      remember(result);
    } catch (cause) {
      setError(describe(cause));
    } finally {
      setBusy(false);
    }
  }

  // The report's own questions, one at a time, in order. Results land as they
  // arrive, so the brief fills in while it is being written rather than after.
  async function prepareBrief() {
    if (!report || busy || preparing) return;
    const todo = report.questions.filter(
      (question) => !asked.some((item) => item.question === question),
    );
    if (todo.length === 0) return;

    const reportId = report.id;
    stop.current = false;
    setPreparing(true);
    setProgress({ done: 0, total: todo.length });
    setError(null);
    try {
      for (const question of todo) {
        if (stop.current) break;
        remember(await ask(reportId, question));
        setProgress((current) => (current ? { ...current, done: current.done + 1 } : current));
      }
    } catch (cause) {
      // One failure ends the run. A missing key or a rate limit does not fix itself
      // on the next question, and spending the rest of a quota to prove that is not
      // a use of the quota.
      setError(describe(cause));
    } finally {
      setPreparing(false);
      setProgress(null);
    }
  }

  function openAnswer(result: Answer) {
    setAnswer(result);
    setView("ask");
  }

  const cited = new Set(answer?.evidence.filter((item) => item.cited).map((item) => item.label));
  const offered = new Set(answer?.evidence.map((item) => item.label));
  const offline = health?.provider === "fake";

  return (
    <div className="app">
      <header className="masthead">
        <a className="brand" href="/">
          <Mark />
          <span>
            Ledger<em>Lens</em>
          </span>
        </a>
        <div className="masthead-end">
          <button
            className="density"
            onClick={() => setDensity(density === "compact" ? "comfortable" : "compact")}
            aria-label={`Use ${density === "compact" ? "comfortable" : "compact"} spacing`}
            title={
              density === "compact"
                ? "Go back to comfortable spacing"
                : "Fit more rows on screen"
            }
          >
            <Rows size={14} weight="bold" aria-hidden="true" />
            <span>{density === "compact" ? "Comfortable" : "Compact"}</span>
          </button>
          {health && (
            <div className="status">
              <span className="status-dot" data-offline={offline} aria-hidden="true" />
              <span className="sr-only">
                {offline ? "No model configured." : "Model connected."}
              </span>
              <span className="status-narrow">{count(health.documents)} reports</span>
              <span className="status-wide">
                <b>{health.provider}</b>
                {health.model ? ` · ${health.model}` : ""} · {count(health.documents)} reports
              </span>
            </div>
          )}
        </div>
      </header>

      {error && (
        <div className="banner" role="alert">
          <WarningCircle size={16} weight="bold" />
          <span>{error}</span>
          <button className="banner-x" onClick={() => setError(null)} aria-label="Dismiss">
            <X size={14} weight="bold" />
          </button>
        </div>
      )}

      {health && !health.documents_loaded && (
        <div className="banner" role="status">
          <Info size={16} weight="bold" />
          <span>
            No reports are loaded. Run <code>python scripts/download_data.py</code> to fetch the
            FinQA split, then reload.
          </span>
        </div>
      )}

      <main className="panes" data-view={view}>
        <section className="pane pane--library" aria-label="Report library">
          <div className="pane-head">
            <span className="label">Library</span>
            <span className="meta">{health ? count(health.documents) : ""}</span>
          </div>
          <Library
            selected={report?.id ?? null}
            onSelect={(id) => void openReport(id)}
            disabled={busy || preparing}
          />
        </section>

        <div className="stack">
          <section className="pane pane--reader" aria-label="Report">
            <div className="pane-head">
              <span className="label">Report</span>
              {report && (
                <button
                  className="pane-toggle"
                  onClick={() => setSheet(sheet === "source" ? "brief" : "source")}
                  aria-label={
                    sheet === "source" ? "Show the brief for this report" : "Show the report itself"
                  }
                >
                  {sheet === "source" ? <Article size={14} /> : <FileText size={14} />}
                  <span>{sheet === "source" ? "Brief" : "Source"}</span>
                </button>
              )}
            </div>
            {report ? (
              sheet === "brief" ? (
                <Brief
                  report={report}
                  asked={asked}
                  offline={Boolean(offline)}
                  progress={progress}
                  onOpen={openAnswer}
                  onAsk={(question) => void answerQuestion(question)}
                  onPrepare={() => void prepareBrief()}
                  onStop={() => {
                    stop.current = true;
                  }}
                />
              ) : (
                <ReportReader report={report} cited={cited} offered={offered} />
              )
            ) : (
              <Blank
                art={<SheetArt />}
                title="Nothing open yet"
                action={
                  <button className="btn" onClick={() => setView("library")}>
                    Open the library
                  </button>
                }
              >
                Pick a report and its table, its narrative, and the questions that came with it all
                appear here.
              </Blank>
            )}
          </section>

          <section className="pane pane--ask" aria-label="Question">
            <AskPanel
              key={report?.id ?? "none"}
              report={report}
              answer={answer}
              busy={busy}
              blocked={preparing}
              offline={Boolean(offline)}
              onSubmit={(text) => void answerQuestion(text)}
              onBrowse={() => setView("library")}
            />
          </section>
        </div>
      </main>

      <nav className="tabbar" aria-label="Sections">
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            className="tab"
            aria-current={view === id ? "true" : undefined}
            onClick={() => setView(id)}
          >
            <Icon size={19} weight={view === id ? "fill" : "regular"} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

function describe(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}
