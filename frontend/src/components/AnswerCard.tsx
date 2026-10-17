import {
  ArrowCounterClockwise,
  CaretDown,
  Check,
  Copy,
  MarkdownLogo,
  ShieldCheck,
  SlidersHorizontal,
  Warning,
  WarningCircle,
} from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";

import type { Answer, Audit, FigureEdit, Operand, OperandRole, Recompute } from "../api";
import { recompute } from "../api";
import { count, formatAnswer, formatCost, formatSeconds, toMarkdown, trustOf } from "../format";

interface Props {
  answer: Answer;
  /** The report the question was about, so a what-if runs against the same one. */
  documentId: string;
  /** The report's filename, for the citation line in copied markdown. */
  source: string | null;
  /** True when the app is running without a real model configured. */
  offline: boolean;
}

// How long a clicked citation stays lit. Long enough to find the row it is on,
// short enough that it does not look like a selection.
const FLASH_MS = 1800;

// A slider fires on every pixel of a drag, and sixty requests a second is not a
// thing to do to a server. The run waits for the hand to pause; the figure on
// screen moves straight away either way.
const SETTLE_MS = 140;

// How far a figure can be moved, either way. Not a limit on what a what-if can be,
// just the range a slider can express as one percent per notch.
const SWING = 50;

// The figures a reader can move: the ones written into the program, as opposed to
// the row names (labels), the results of earlier steps (references) and whole
// calculations (nested calls). That is the set the API will take an edit for.
type FigureRole = Extract<OperandRole, "sourced" | "unsourced" | "constant">;

interface Figure {
  step: number;
  operand: number;
  /** The figure the program used, as the engine read it. */
  base: number;
  role: FigureRole;
  labels: string[];
}

function keyOf(step: number, operand: number): string {
  return `${step}:${operand}`;
}

function figuresIn(audit: Audit | null): Figure[] {
  if (!audit) return [];
  const figures: Figure[] = [];
  for (const step of audit.steps) {
    step.operands.forEach((operand, index) => {
      if (operand.value === null) return;
      if (operand.role !== "sourced" && operand.role !== "unsourced" && operand.role !== "constant") {
        return;
      }
      figures.push({
        step: step.index,
        operand: index,
        base: Number(operand.value),
        role: operand.role,
        labels: operand.labels,
      });
    });
  }
  return figures;
}

// A slider moves in whole percent, and that is the only place floating point can
// get in: 150 * 1.1 is 165.00000000000003 in binary arithmetic, and a figure nobody
// chose turning up in the audit trail is worse than a rounded one. Whatever this
// returns is exactly what goes to the API and exactly what the trail will show.
function asFigure(value: number): string {
  return String(Number(value.toFixed(6)));
}

function describe(item: Figure): string {
  if (item.role === "sourced" && item.labels.length > 0) return `from ${item.labels.join(", ")}`;
  if (item.role === "constant") return "arithmetic";
  return "not in the report";
}

export default function AnswerCard({ answer, documentId, source, offline }: Props) {
  // The answer as the reader has moved it. Nothing here touches the model path: the
  // program is the one the model already wrote, and the engine runs it again.
  const [whatIf, setWhatIf] = useState<Recompute | null>(null);
  const [moved, setMoved] = useState<Record<string, number>>({});
  const [running, setRunning] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [flashed, setFlashed] = useState<string | null>(null);
  const [copied, setCopied] = useState<"program" | "markdown" | null>(null);

  // Drags overlap, so a reply that is no longer the newest one is dropped rather
  // than allowed to overwrite a later figure with an earlier answer.
  const latest = useRef(0);
  const timer = useRef<number | null>(null);

  const figures = figuresIn(answer.audit);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  // The display follows the server, so a what-if shows the arithmetic that actually
  // ran rather than the one that was assumed to.
  const audit = whatIf ? whatIf.audit : answer.audit;
  const program = whatIf ? whatIf.program : answer.program;
  const value = whatIf ? whatIf.answer : answer.answer;
  const failure = whatIf ? whatIf.failure : answer.failure;
  const moved_count = Object.values(moved).filter((percent) => percent !== 0).length;
  const isWhatIf = whatIf !== null;

  const trust = trustOf(audit);
  const cited = answer.evidence.filter((item) => item.cited);
  const offered = answer.evidence.filter((item) => !item.cited);
  const answered = value !== null;
  const figure = answered ? formatAnswer(value ?? "") : null;

  function move(key: string, percent: number) {
    const next = { ...moved, [key]: percent };
    setMoved(next);
    if (timer.current !== null) window.clearTimeout(timer.current);
    const id = ++latest.current;
    timer.current = window.setTimeout(() => void run(id, next), SETTLE_MS);
  }

  function reset() {
    if (timer.current !== null) window.clearTimeout(timer.current);
    // Anything in flight is now older than this, so its reply is discarded.
    latest.current += 1;
    setMoved({});
    setWhatIf(null);
    setProblem(null);
    setRunning(false);
  }

  async function run(id: number, next: Record<string, number>) {
    const edits: FigureEdit[] = figures
      .map((item) => ({ item, percent: next[keyOf(item.step, item.operand)] ?? 0 }))
      .filter(({ percent }) => percent !== 0)
      .map(({ item, percent }) => ({
        step: item.step,
        operand: item.operand,
        value: asFigure(item.base * (1 + percent / 100)),
      }));

    // Every figure put back where it was, so there is nothing to ask and the card
    // goes back to the answer the model gave.
    if (edits.length === 0) {
      setWhatIf(null);
      setProblem(null);
      setRunning(false);
      return;
    }

    setRunning(true);
    try {
      const result = await recompute(documentId, answer.program ?? "", edits);
      if (id !== latest.current) return;
      setWhatIf(result);
      setProblem(null);
    } catch (error) {
      if (id !== latest.current) return;
      setProblem(error instanceof Error ? error.message : "The program could not be run again.");
    } finally {
      if (id === latest.current) setRunning(false);
    }
  }

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

      {isWhatIf && (
        <div className="notice notice--whatif">
          <span>
            <strong>What-if.</strong> You moved {moved_count === 1 ? "a figure" : `${moved_count} figures`}{" "}
            the report states, and the same program was run again — no model involved. The badge above
            is the honest reading of that: the arithmetic is still the engine's, the changed figures
            are yours.
          </span>
          <button className="whatif-reset" onClick={reset}>
            <ArrowCounterClockwise size={13} weight="bold" />
            Back to the report
          </button>
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
        <span className="label">
          {answered
            ? isWhatIf
              ? "What-if, computed by the engine"
              : "Computed by the engine"
            : "No answer"}
        </span>
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
            {offline && !isWhatIf
              ? "No model is configured, so no program was written. Set LLM_PROVIDER and a key to get answers."
              : (failure ?? "The program could not be run against this report.")}
          </p>
        )}
        {answered && !isWhatIf && answer.reference_answer !== null && (
          <p className="verdict-note">
            FinQA's annotation for this question is <code>{answer.reference_answer}</code>, shown for
            comparison. The figure above came from the document, not from the annotation.
          </p>
        )}
      </div>

      {program && (
        <div className="block">
          <span className="label">Program</span>
          <div className="program">
            <div className="program-tools">
              <button
                className="copy"
                onClick={() => void copy(program ?? "", "program")}
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
            <pre>{program}</pre>
          </div>
          {audit && (
            <AuditTrail
              audit={audit}
              figures={figures}
              moved={moved}
              running={running}
              problem={problem}
              onCite={flash}
              onMove={move}
            />
          )}
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
// away by default: it is the thing that makes an answer checkable, and the answer
// is what a reader wants first.
function AuditTrail({
  audit,
  figures,
  moved,
  running,
  problem,
  onCite,
  onMove,
}: {
  audit: Audit;
  figures: Figure[];
  moved: Record<string, number>;
  running: boolean;
  problem: string | null;
  onCite: (label: string) => void;
  onMove: (key: string, percent: number) => void;
}) {
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
            {audit.steps.map((step) => {
              // The same figure the answer card shows, so a reader never sees two
              // different numbers for one quantity. The unrounded value is on hover.
              const result = formatAnswer(step.result);
              return (
                <li key={step.index}>
                  <span className="idx num">#{step.index}</span>
                  <span className="op">{step.operation}</span>
                  <span className="args">
                    {step.operands.map((operand, index) => (
                      <span key={index} className="arg">
                        <OperandChip operand={operand} onCite={onCite} />
                        {index < step.operands.length - 1 && <span className="comma">,</span>}
                      </span>
                    ))}
                  </span>
                  <span className="eq">=</span>
                  <span className="res num" title={result.exact ?? undefined}>
                    {result.shown}
                  </span>
                </li>
              );
            })}
          </ol>

          {figures.length > 0 && (
            <div className="whatif">
              <div className="whatif-head">
                <span className="whatif-title">
                  <SlidersHorizontal size={13} weight="bold" />
                  What if
                </span>
                <span className="whatif-hint">
                  Move a figure and the same program runs again. Nothing is sent to a model.
                </span>
              </div>
              <ul className="whatif-list">
                {figures.map((item) => {
                  const key = keyOf(item.step, item.operand);
                  const percent = moved[key] ?? 0;
                  // The figure on screen is the one that would be sent, so the number
                  // a reader drags to is the number the arithmetic will use.
                  const face = formatAnswer(asFigure(item.base * (1 + percent / 100)));
                  return (
                    <li className="whatif-row" key={key} data-moved={percent !== 0}>
                      <div className="whatif-where">
                        <span className="idx num">#{item.step}</span>
                        <span className="whatif-figure num" title={face.exact ?? undefined}>
                          {face.shown}
                        </span>
                        <span className="whatif-src">{describe(item)}</span>
                      </div>
                      <div className="whatif-slide">
                        <input
                          type="range"
                          min={-SWING}
                          max={SWING}
                          step={1}
                          value={percent}
                          onChange={(event) => onMove(key, Number(event.target.value))}
                          // Named by the figure and its position rather than just the
                          // step, because a step usually has more than one of them and
                          // a screen reader listing "Change the figure at step 0"
                          // twice is a choice between two identical controls.
                          aria-label={`Change the figure ${face.shown} at step ${item.step}, argument ${item.operand + 1}`}
                        />
                        <span className={running ? "whatif-pct num is-running" : "whatif-pct num"}>
                          {percent > 0 ? `+${percent}%` : `${percent}%`}
                        </span>
                      </div>
                    </li>
                  );
                })}
              </ul>
              {problem && <p className="whatif-problem">{problem}</p>}
            </div>
          )}
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
