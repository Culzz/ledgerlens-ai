import type { Answer } from "../api";

interface Props {
  answer: Answer;
  /** True when the app is running without a real model configured. */
  offline: boolean;
}

function formatCost(cost: number | null): string {
  if (cost === null) return "cost unknown";
  if (cost === 0) return "free tier";
  return `$${cost.toFixed(4)}`;
}

export default function AnswerCard({ answer, offline }: Props) {
  const cited = answer.evidence.filter((item) => item.cited);

  return (
    <div className="card">
      <h2>Answer</h2>

      {answer.retrieval_fallback && (
        <div className="notice">
          Nothing in the question matched the report, so retrieval stood aside and the whole report
          went to the model. Worth knowing: this question measures the model, not the retriever.
        </div>
      )}

      {answer.answer !== null ? (
        <>
          <div className="answer-label">Computed by the engine</div>
          <div className="answer-value">{answer.answer}</div>
        </>
      ) : (
        <div className="failure">
          <strong>No answer.</strong>{" "}
          {offline
            ? "The app is running without a model configured, so no program was written. Set LLM_PROVIDER and a key to get answers."
            : answer.failure ?? "The program could not be run."}
        </div>
      )}

      {answer.answer !== null && answer.reference_answer !== null && (
        <div className="reference">
          FinQA's annotation for this question is <code>{answer.reference_answer}</code>. The
          comparison is shown, not made: the answer above comes from the document alone.
        </div>
      )}

      {answer.program && (
        <div style={{ marginTop: 18 }}>
          <div className="answer-label">Program</div>
          <div className="program">{answer.program}</div>
          {answer.steps.length > 0 && (
            <ol className="steps">
              {answer.steps.map((step, index) => (
                <li key={index}>{step}</li>
              ))}
            </ol>
          )}
        </div>
      )}

      {cited.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <div className="answer-label">Evidence it read</div>
          <ul className="evidence">
            {cited.map((item) => (
              <li key={item.label} className="cited">
                <span className="label">{item.label}</span>
                <span className="text">{item.text}</span>
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
        <span>{answer.total_tokens.toLocaleString()} tokens</span>
        <span>{formatCost(answer.cost_usd)}</span>
        <span>{answer.latency_seconds.toFixed(2)}s</span>
      </div>
    </div>
  );
}
