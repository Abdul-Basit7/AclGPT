import type { SVGProps } from "react";

/**
 * Wordmark glyph: stacked pages with a rule through them — documents plus the
 * citation line drawn from them. Deliberately monochrome and geometric; it
 * inherits `currentColor` so it reads as part of the type, not as decoration.
 */
export function Logo({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className={className}
      {...props}
    >
      {/* back page, offset */}
      <path d="M7.5 4.5h6.2L17.5 8.3v8.2" opacity="0.45" />
      {/* front page */}
      <path d="M5 7.5h6.2L15 11.3v7.2a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z" />
      <path d="M11.2 7.5v3.8H15" />
      {/* the cited line */}
      <path d="M7.6 15.4h4.8" />
    </svg>
  );
}
