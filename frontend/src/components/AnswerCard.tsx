import { Check, Copy } from "@phosphor-icons/react";
import { useState } from "react";

import type { Answer } from "../api";
import { count, formatAnswer, formatCost, formatSeconds } from "../format";

interface Props {
  answer: Answer;
  /** True when the app is running without a real model configured. */
  offline: boolean;
}

export default function AnswerCard({ answer, offline }: Props) {
  const [copied, setCopied] = useState(false);
  const cited = answer.evidence.filter((item) => item.cited);
  const offered = answer.evidence.filter((item) => !item.cited);
  const answered = answer.answer !== null;
  const figure = answered ? formatAnswer(answer.answer ?? "") : null;

  async function copyProgram() {
    if (!answer.program) return;
    try {
      await navigator.clipboard.writeText(answer.program);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // The clipboard is blocked outside a secure context. The program is on
      // screen either way, so there is nothing to recover from.
    }
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
            <button className="copy" onClick={() => void copyProgram()} aria-label="Copy the program">
              {copied ? <Check size={14} weight="bold" /> : <Copy size={14} />}
            </button>
            <pre>{answer.program}</pre>
          </div>
          {answer.steps.length > 0 && (
            <ol className="steps">
              {answer.steps.map((step, index) => (
                <li key={index}>
                  <span className="idx">{index + 1}</span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
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
              <li key={item.label} className="cited">
                <span className="tag">{item.label}</span>
                <span className="what">{item.text}</span>
              </li>
            ))}
            {offered.map((item) => (
              <li key={item.label}>
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
