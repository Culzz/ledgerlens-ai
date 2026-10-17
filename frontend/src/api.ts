// The shapes the API returns, and the four calls the UI makes. Kept in one file
// so a change on either side has one obvious place to look.

export interface Health {
  status: string;
  provider: string;
  model: string | null;
  documents: number;
  documents_loaded: boolean;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  questions: string[];
  table_rows: number;
  context_lines: number;
}

export interface DocumentList {
  documents: DocumentSummary[];
  total: number;
  limit: number | null;
}

export interface Report {
  id: string;
  filename: string;
  table: string[][];
  context: string[];
  questions: string[];
}

export interface Evidence {
  label: string;
  kind: string;
  text: string;
  cited: boolean;
}

// What one argument of a step turned out to be. A sourced figure cites the rows
// it was read from; an unsourced one is the reason an answer gets refused.
export type OperandRole =
  | "sourced"
  | "constant"
  | "unsourced"
  | "reference"
  | "label"
  | "nested";

export interface Operand {
  text: string;
  role: OperandRole;
  value: string | null;
  labels: string[];
}

export interface AuditStep {
  index: number;
  operation: string;
  result: string;
  /** The step as one line, each figure annotated with where it came from. */
  render: string;
  operands: Operand[];
}

export interface Audit {
  verdict: "grounded" | "partly_grounded" | "ungrounded";
  labels: string[];
  /** Figures the report does not state, as the program wrote them. */
  unsourced: string[];
  sourced_count: number;
  literal_count: number;
  constant_only: boolean;
  /** What the report said its figures were in, when it said. */
  report_unit: string | null;
  steps: AuditStep[];
}

export interface Answer {
  question: string;
  answer: string | null;
  program: string | null;
  audit: Audit | null;
  failure: string | null;
  evidence: Evidence[];
  /** How many times a failed program was sent back before this answer. */
  repair_count: number;
  retrieval_fallback: boolean;
  provider: string;
  model: string | null;
  total_tokens: number;
  cost_usd: number | null;
  latency_seconds: number;
  reference_answer: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    // FastAPI puts the readable reason in `detail`; a bare status code tells the
    // person looking at it nothing.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // Not JSON. The status line is all there is.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function fetchHealth(): Promise<Health> {
  return request<Health>("/api/health");
}

export function fetchDocuments(query = ""): Promise<DocumentList> {
  const trimmed = query.trim();
  const suffix = trimmed ? `?q=${encodeURIComponent(trimmed)}` : "";
  return request<DocumentList>(`/api/documents${suffix}`);
}

export function fetchReport(id: string): Promise<Report> {
  return request<Report>(`/api/documents/${encodeURIComponent(id)}`);
}

export function ask(documentId: string, question: string): Promise<Answer> {
  return request<Answer>("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, question }),
  });
}
