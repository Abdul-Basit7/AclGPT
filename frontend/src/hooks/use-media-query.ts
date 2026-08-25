import { useEffect, useState } from "react";

/**
 * Track a CSS media query in JS.
 *
 * Needed where a Tailwind responsive class is not enough: a Radix dialog mounts
 * a full-screen overlay, so `lg:hidden` on its content would hide the panel but
 * leave the overlay swallowing every click. Mounting has to be conditional.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    const list = window.matchMedia(query);
    const onChange = (event: MediaQueryListEvent) => setMatches(event.matches);
    setMatches(list.matches);
    list.addEventListener("change", onChange);
    return () => list.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/** Matches Tailwind's `lg` breakpoint, where the sources panel can dock. */
export const LG_QUERY = "(min-width: 1024px)";
