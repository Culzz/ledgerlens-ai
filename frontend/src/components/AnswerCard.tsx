import {
  CaretDown,
  Check,
  Copy,
  MarkdownLogo,
  ShieldCheck,
  Warning,
  WarningCircle,
} from "@phosphor-icons/react";
import { useState } from "react";

import type { Answer, Audit, Operand } from "../api";
import { count, formatAnswer, formatCost, formatSeconds, toMarkdown, trustOf } from "../format";

interface Props {
  answer: Answer;
  /** The report the question was asked about, for the citation line. */
  source: string | null;
  /** True when the app is running without a real model configured. */
  offline: boolean;
}

// How long a clicked citation stays lit. Long enough to find the row it is on,
// short enough that it does not look like a selection.
const FLASH_MS = 1800;

export default function AnswerCard({ answer, source, offline }: Props) {
  const [flashed, setFlashed] = useState<string | null>(null);
  const [copied, setCopied] = useState<"program" | "markdown" | null>(null);

  const trust = trustOf(answer.audit);
  const cited = answer.evidence.filter((item) => item.cited);
  const offered = answer.evidence.filter((item) => !item.cited);
  const answered = answer.answer !== null;
  const figure = answered ? formatAnswer(answer.answer ?? "") : null;

  async function copy(text: string, which: "program" | "markdown") {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      setTimeout(() => setCopied(null), 1400);
    } catch {
      // The clipboard is blocked outside a secure context. Everything copied here
      // is on screen anyway, so there is nothing to recover from.
    }
  }

  function flash(label: string) {
    setFlashed(label);
    setTimeout(() => setFlashed((current) => (current === label ? null : current)), FLASH_MS);
  }

  return (
    <div className="answer">
      {answer.retrieval_fallback && (
        <div className="notice">
          <span>
            Nothing in the question matched anything in the report, so retrieval stood aside and the
            whole report went to the model. Worth knowing when reading the program below: this
            question measured the model, not the retriever.
          </span>
        </div>
      )}

      {trust && (
        <div className="trust" data-tone={trust.tone}>
          <span className="trust-badge">
            {trust.tone === "good" ? (
              <ShieldCheck size={15} weight="fill" />
            ) : trust.tone === "warn" ? (
              <Warning size={15} weight="fill" />
            ) : (
              <WarningCircle size={15} weight="fill" />
            )}
            {trust.text}
          </span>
          <span className="trust-detail">{trust.detail}</span>
        </div>
      )}

      {answer.repair_count > 0 && (
        <div className="notice">
          <span>
            The first program did not run, so it was sent back to the model with the engine's error.
            What you are reading is attempt {answer.repair_count + 1}.
          </span>
        </div>
      )}

      <div className="verdict">
        <span className="label">{answered ? "Computed by the engine" : "No answer"}</span>
        <div className={answered ? "verdict-value num" : "verdict-value num verdict-value--none"}>
          {figure ? figure.shown : "—"}
        </div>
        {figure?.exact && (
          <p className="verdict-exact">
            the engine's exact figure · <code>{figure.exact}</code>
          </p>
        )}
        {!answered && (
          <p className="verdict-note">
            {offline
              ? "No model is configured, so no program was written. Set LLM_PROVIDER and a key to get answers."
              : (answer.failure ?? "The program could not be run against this report.")}
          </p>
        )}
        {answered && answer.reference_answer !== null && (
          <p className="verdict-note">
            FinQA's annotation for this question is <code>{answer.reference_answer}</code>, shown for
            comparison. The figure above came from the document, not from the annotation.
          </p>
        )}
      </div>

      {answer.program && (
        <div className="block">
          <span className="label">Program</span>
          <div className="program">
            <div className="program-tools">
              <button
                className="copy"
                onClick={() => void copy(answer.program ?? "", "program")}
                aria-label="Copy the program"
              >
                {copied === "program" ? <Check size={14} weight="bold" /> : <Copy size={14} />}
              </button>
              <button
                className="copy"
                onClick={() => void copy(toMarkdown(answer, source), "markdown")}
                aria-label="Copy as markdown, with citations"
              >
                {copied === "markdown" ? (
                  <Check size={14} weight="bold" />
                ) : (
                  <MarkdownLogo size={14} />
                )}
              </button>
            </div>
            <pre>{answer.program}</pre>
          </div>
          {answer.audit && <AuditTrail audit={answer.audit} onCite={flash} />}
        </div>
      )}

      {answer.evidence.length > 0 && (
        <div className="block">
          <span className="label">
            Evidence · {cited.length} used of {answer.evidence.length} retrieved
          </span>
          <ul className="evidence">
            {cited.map((item) => (
              <li
                key={item.label}
                id={`evidence-${item.label}`}
                className={flashed === item.label ? "cited is-flash" : "cited"}
              >
                <span className="tag">{item.label}</span>
                <span className="what">{item.text}</span>
              </li>
            ))}
            {offered.map((item) => (
              <li key={item.label} id={`evidence-${item.label}`}>
                <span className="tag">{item.label}</span>
                <span className="what">{item.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="meter">
        <span>
          {answer.provider}
          {answer.model ? ` · ${answer.model}` : ""}
        </span>
        <span>{count(answer.total_tokens)} tokens</span>
        <span>{formatCost(answer.cost_usd)}</span>
        <span>{formatSeconds(answer.latency_seconds)}</span>
      </div>
    </div>
  );
}

// The working, with every figure tied back to the row it was read from. Folded
// away by default: it is the thing that makes an answer checkable, and the
// answer is what a reader wants first.
function AuditTrail({ audit, onCite }: { audit: Audit; onCite: (label: string) => void }) {
  const [open, setOpen] = useState(false);

  // The three counts add up to the literals, and the labels keep them apart: a
  // constant is arithmetic rather than a fault, so it must not be lumped in with a
  // figure the report does not state.
  const unsourced = audit.unsourced.length;
  const arithmetic = audit.literal_count - audit.sourced_count - unsourced;
  const parts = [`${audit.sourced_count} read from the report`];
  if (arithmetic > 0) parts.push(`${arithmetic} arithmetic`);
  if (unsourced > 0) parts.push(`${unsourced} not found`);

  return (
    <div className="audit">
      <button className="audit-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <CaretDown size={13} weight="bold" className={open ? "caret is-open" : "caret"} />
        Audit trail
        <span className="audit-summary">{parts.join(" · ")}</span>
      </button>

      {open && (
        <div className="audit-body">
          {audit.report_unit && (
            <p className="audit-note">
              The report states its figures in {audit.report_unit}, so figures written out in full
              are matched against that.
            </p>
          )}
          {audit.constant_only && (
            <p className="audit-note">
              No figures of the model's own. Every value came out of a table operation.
            </p>
          )}

          <ol className="audit-steps">
            {audit.steps.map((step) => (
              <li key={step.index}>
                <span className="idx num">#{step.index}</span>
                <span className="op">{step.operation}</span>
                <span className="args">
                  {step.operands.map((operand, index) => (
                    <span key={index} className="arg">
                      {index > 0 && <span className="comma">,</span>}
                      <OperandChip operand={operand} onCite={onCite} />
                    </span>
                  ))}
                </span>
                <span className="eq">=</span>
                <span className="res num">{step.result}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

function OperandChip({ operand, onCite }: { operand: Operand; onCite: (label: string) => void }) {
  // A figure is shown as the engine read it, so 0.125 reads the same here as it
  // does in the answer above.
  const face =
    operand.value !== null ? formatAnswer(operand.value).shown : operand.text;

  if (operand.role === "sourced" && operand.labels.length > 0) {
    return (
      <span className="arg-chip num" data-role="sourced">
        {face}
        <span className="arg-cite">
          {operand.labels.map((label) => (
            <button key={label} onClick={() => onCite(label)} title={`Show ${label}`}>
              {label}
            </button>
          ))}
        </span>
      </span>
    );
  }

  const title =
    operand.role === "unsourced"
      ? "this figure appears nowhere in the report"
      : operand.role === "constant"
        ? "part of the arithmetic rather than a figure from the report"
        : operand.role === "reference"
          ? "the result of an earlier step"
          : operand.role === "label"
            ? "a row or column name"
            : undefined;

  return (
    <span className="arg-chip" data-role={operand.role} title={title}>
      {face}
      {operand.role === "unsourced" && <span className="arg-flag">not in report</span>}
    </span>
  );
}
