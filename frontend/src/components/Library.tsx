import { CaretRight, MagnifyingGlass, XCircle } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

import { fetchDocuments, type DocumentSummary } from "../api";
import { count } from "../format";
import Blank, { LensArt } from "./Blank";

interface Props {
  selected: string | null;
  onSelect: (id: string) => void;
  disabled: boolean;
}

type Status = "loading" | "ready" | "failed";

export default function Library({ selected, onSelect, disabled }: Props) {
  const [query, setQuery] = useState("");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<Status>("loading");
  const [attempt, setAttempt] = useState(0);

  // A few hundred reports is more than anyone wants to scroll, so the box
  // filters rather than the list being capped. The delay keeps a keystroke from
  // becoming a request, and the skeleton is only shown when there is nothing on
  // screen yet - flashing a placeholder over a list that is already there reads
  // as a bug.
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    const timer = setTimeout(() => {
      fetchDocuments(query)
        .then((body) => {
          if (cancelled) return;
          setDocuments(body.documents);
          setTotal(body.total);
          setStatus("ready");
        })
        .catch(() => {
          if (!cancelled) setStatus("failed");
        });
    }, 180);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query, attempt]);

  const firstLoad = status === "loading" && documents.length === 0;

  return (
    <>
      <div className="pane-body">
        <div className="filter">
          <MagnifyingGlass size={16} weight="bold" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter by ticker or year"
            aria-label="Filter reports"
            autoComplete="off"
          />
          {query !== "" && (
            <button className="filter-clear" onClick={() => setQuery("")} aria-label="Clear the filter">
              <XCircle size={17} weight="fill" />
            </button>
          )}
        </div>
      </div>

      {status === "failed" && (
        <div className="pane-body">
          <div className="notice notice--warn">
            <span>
              The report list did not load. The API may still be starting up.
            </span>
            <button className="btn btn--quiet" onClick={() => setAttempt((n) => n + 1)}>
              Try again
            </button>
          </div>
        </div>
      )}

      {firstLoad && (
        <div className="pane-body" aria-hidden="true">
          <div className="skeleton">
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
        </div>
      )}

      {status === "ready" && documents.length === 0 && (
        <Blank
          art={<LensArt />}
          title="Nothing matches"
          action={
            <button className="btn btn--quiet" onClick={() => setQuery("")}>
              Clear the filter
            </button>
          }
        >
          No report in the library carries the term <q>{query}</q>. Clear it to see everything
          again.
        </Blank>
      )}

      {documents.length > 0 && (
        <ul className="records" data-stale={status === "loading"}>
          {documents.map((doc) => (
            <li key={doc.id}>
              <button
                className="record"
                aria-current={doc.id === selected}
                onClick={() => onSelect(doc.id)}
                disabled={disabled}
              >
                <span className="record-main">
                  <span className="record-name">{doc.filename}</span>
                  <span className="record-meta">
                    {doc.table_rows} rows · {doc.context_lines} lines ·{" "}
                    {doc.questions.length === 1 ? "1 question" : `${doc.questions.length} questions`}
                  </span>
                </span>
                <CaretRight className="record-chevron" size={14} weight="bold" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {documents.length > 0 && (
        <p className="records-foot">
          {documents.length < total
            ? `Showing ${count(documents.length)} of ${count(total)}. Narrow it with the filter.`
            : `${count(total)} reports in the library.`}
        </p>
      )}
    </>
  );
}
