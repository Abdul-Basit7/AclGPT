import { useCallback, useEffect, useRef, useState } from "react";

/** Strip markdown so the synthesiser reads prose, not punctuation. */
export function toSpokenText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " code block omitted. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/\|/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Text-to-speech via the browser's built-in SpeechSynthesis.
 * Free, offline in most browsers, and needs no API key.
 */
export function useSpeech() {
  const supported =
    typeof window !== "undefined" && "speechSynthesis" in window;
  const [speakingKey, setSpeakingKey] = useState<string | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const stop = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    utteranceRef.current = null;
    setSpeakingKey(null);
  }, [supported]);

  const speak = useCallback(
    (key: string, text: string) => {
      if (!supported) return;
      const spoken = toSpokenText(text);
      if (!spoken) return;

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(spoken);
      utterance.lang = "en-US";
      utterance.rate = 1.02;
      utterance.pitch = 1;
      utterance.onend = () => setSpeakingKey(null);
      utterance.onerror = () => setSpeakingKey(null);
      utteranceRef.current = utterance;
      setSpeakingKey(key);
      window.speechSynthesis.speak(utterance);
    },
    [supported],
  );

  const toggle = useCallback(
    (key: string, text: string) => {
      if (speakingKey === key) stop();
      else speak(key, text);
    },
    [speakingKey, speak, stop],
  );

  // Never leave an utterance running after the view unmounts.
  useEffect(() => {
    if (!supported) return;
    return () => window.speechSynthesis.cancel();
  }, [supported]);

  return { supported, speakingKey, speak, stop, toggle };
}
