import { CheckCircle, Circle, WarningCircle, XCircle } from "@phosphor-icons/react";

import type { Tone } from "../format";

/**
 * One line of a brief: the question, the figure, and what the check made of it.
 *
 * Shared by the page a reader works on and the page a link opens, because the two
 * are the same record seen twice. Two renderings of a verdict would eventually
 * disagree about what a tick means, and the disagreement would be invisible from
 * either side.
 */
const MARKS = { good: CheckCircle, warn: WarningCircle, bad: XCircle, idle: Circle } as const;

interface Props {
  question: string;
  /** The figure, or the word that stands where a figure would be. */
  figure: string;
  note: string;
  tone: Tone;
  /** Absent on a page where there is nothing to open. */
  onClick?: () => void;
}

export default function Claim({ question, figure, note, tone, onClick }: Props) {
  const Mark = MARKS[tone];
  const body = (
    <>
      <Mark className="claim-mark" size={15} weight="bold" aria-hidden="true" />
      <span className="claim-q">{question}</span>
      <span className="claim-figure num">{figure}</span>
      <span className="claim-note">{note}</span>
    </>
  );

  return (
    <li>
      {onClick ? (
        <button className="claim" data-tone={tone} onClick={onClick}>
          {body}
        </button>
      ) : (
        <div className="claim" data-tone={tone} data-static="true">
          {body}
        </div>
      )}
    </li>
  );
}
