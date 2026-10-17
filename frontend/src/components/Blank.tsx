import type { ReactNode } from "react";

// Empty states, drawn rather than greyed out. Each one names what is missing and
// carries the next move as a button, because a panel with nothing in it and
// nothing to do is a dead end - which is what the old version had.

/** A ruled page with a lens over it. Nothing is open. */
export function SheetArt() {
  return (
    <svg width="76" height="76" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="7" y="5" width="22" height="38" rx="1.5" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M13 14h10M13 20h10M13 26h6"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <circle cx="33" cy="32" r="7.5" fill="var(--paper)" stroke="currentColor" strokeWidth="1.7" />
      <path
        className="pen"
        d="M38.6 37.6l4.6 4.6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** A lens over two ruled lines. A search came back empty. */
export function LensArt() {
  return (
    <svg width="76" height="76" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path
        d="M6 15h30M6 22h30M6 29h18"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="24" cy="24" r="10" fill="var(--paper)" stroke="currentColor" strokeWidth="1.8" />
      <path
        className="pen"
        d="M31 31l7.5 7.5"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** A nib. A question has been asked but nothing has been written yet. */
export function NibArt() {
  return (
    <svg width="76" height="76" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path
        d="M24 6c5.5 0 9.5 3.6 9.5 9v15c0 8-4.2 12.5-9.5 12.5S14.5 38 14.5 30V15c0-5.4 4-9 9.5-9z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <circle cx="24" cy="19" r="2.8" stroke="currentColor" strokeWidth="1.5" />
      <path
        className="pen"
        d="M24 26v10"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M11 44h26"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

interface BlankProps {
  art: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}

export default function Blank({ art, title, children, action }: BlankProps) {
  return (
    <div className="blank">
      <div className="blank-art">{art}</div>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}
