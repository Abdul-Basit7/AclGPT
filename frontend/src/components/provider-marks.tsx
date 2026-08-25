import type { SVGProps } from "react";

/**
 * Brand marks for the OAuth buttons. Kept as literal paths rather than lucide
 * icons because lucide has no brand glyphs, and providers require their own mark.
 */

export const GoogleMark = (props: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" aria-hidden {...props}>
    <path
      fill="#4285F4"
      d="M23.5 12.3c0-.9-.1-1.5-.2-2.2H12v4.1h6.6c-.1 1.1-.8 2.8-2.4 3.9l-.1.1 3.5 2.7.2.1c2.2-2 3.5-5 3.5-8.7"
    />
    <path
      fill="#34A853"
      d="M12 24c3.2 0 5.9-1.1 7.8-2.9l-3.7-2.9c-1 .7-2.3 1.2-4.1 1.2-3.1 0-5.8-2.1-6.7-5l-.1.1-3.6 2.8v.1C3.5 21.3 7.5 24 12 24"
    />
    <path
      fill="#FBBC05"
      d="M5.3 14.4c-.3-.7-.4-1.5-.4-2.4s.1-1.7.4-2.4V9.5L1.6 6.7l-.1.1C.5 8.3 0 10.1 0 12s.5 3.7 1.4 5.2z"
    />
    <path
      fill="#EA4335"
      d="M12 4.7c2.2 0 3.7.9 4.5 1.7l3.3-3.2C17.9 1.2 15.2 0 12 0 7.5 0 3.5 2.7 1.5 6.7l3.8 2.9c.9-2.9 3.6-4.9 6.7-4.9"
    />
  </svg>
);

export const GitHubMark = (props: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden {...props}>
    <path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1a2.6 2.6 0 0 1 .7-1.6c-2.7-.3-5.5-1.3-5.5-6 0-1.2.5-2.3 1.3-3.1-.2-.4-.6-1.6.1-3.2 0 0 1-.3 3.4 1.2a11.5 11.5 0 0 1 6 0C17.3 4.7 18.3 5 18.3 5c.7 1.6.3 2.8.1 3.2a4.5 4.5 0 0 1 1.2 3.1c0 4.6-2.8 5.6-5.5 5.9.5.4.9 1.2.9 2.4v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3" />
  </svg>
);
