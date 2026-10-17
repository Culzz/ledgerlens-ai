// The mark: a ruled page with a lens held over it. The same geometry is in
// public/favicon.svg, which has to be a file rather than a component, so the two
// are kept in step by hand. It is the only thing in the app drawn from scratch.

interface MarkProps {
  size?: number;
}

export default function Mark({ size = 22 }: MarkProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M3.6 3.4h10.6v17.2H3.6z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
      <path
        d="M6.3 7.9h5M6.3 10.9h5M6.3 13.9h3.2"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinecap="round"
      />
      <circle cx="15.5" cy="15.3" r="4.6" fill="var(--paper)" stroke="var(--accent)" strokeWidth="1.4" />
      <path
        d="M18.9 18.7l2.4 2.4"
        stroke="var(--accent)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
