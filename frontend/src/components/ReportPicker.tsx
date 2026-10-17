import { useEffect, useState } from "react";

import { fetchDocuments, type DocumentSummary } from "../api";

interface Props {
  selected: string | null;
  onSelect: (id: string) => void;
  disabled: boolean;
}

export default function ReportPicker({ selected, onSelect, disabled }: Props) {
  const [query, setQuery] = useState("");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [total, setTotal] = useState(0);

  // The whole set is a few hundred reports, which is more than anyone wants to
  // scroll, so the box filters rather than the list being capped. The delay keeps
  // a keystroke from becoming a request.
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      fetchDocuments(query)
        .then((body) => {
          if (cancelled) return;
          setDocuments(body.documents);
          setTotal(body.total);
        })
        .catch(() => {
          // The page-level error banner already covers the API being unreachable.
        });
    }, 200);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  return (
    <>
      <input
        className="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Filter reports, e.g. citi or 2008"
        aria-label="Filter reports"
      />

      <ul className="doclist">
        {documents.map((doc) => (
          <li key={doc.id}>
            <button
              className={doc.id === selected ? "doc selected" : "doc"}
              onClick={() => onSelect(doc.id)}
              disabled={disabled}
            >
              <span className="doc-name">{doc.filename}</span>
              <span className="doc-meta">
                {doc.table_rows} rows · {doc.context_lines} lines · {doc.questions.length} question
                {doc.questions.length === 1 ? "" : "s"}
              </span>
            </button>
          </li>
        ))}
        {documents.length === 0 && <li className="hint">Nothing matches that.</li>}
      </ul>

      {documents.length < total && (
        <p className="hint">
          Showing {documents.length} of {total}. Narrow it with the filter.
        </p>
      )}
    </>
  );
}
