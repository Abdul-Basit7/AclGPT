import { useEffect, useRef, useState } from "react";
import { ArrowUp, Mic, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useVoiceInput } from "@/hooks/useVoiceInput";

interface Props {
  disabled: boolean;
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  placeholder?: string;
}

const MAX_HEIGHT_PX = 200;

export function Composer({ disabled, streaming, onSend, onStop, placeholder }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Text typed before dictation started, so interim results replace cleanly.
  const baseRef = useRef("");

  const voice = useVoiceInput({
    onTranscript: (text, final) => {
      const base = baseRef.current;
      const joined = base ? `${base.replace(/\s+$/, "")} ${text}` : text;
      setValue(joined);
      if (final) baseRef.current = joined;
    },
  });

  // Grow with content, up to a cap.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value]);

  function submit() {
    const text = value.trim();
    if (!text || disabled || streaming) return;
    onSend(text);
    setValue("");
    baseRef.current = "";
    voice.stop();
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  function toggleMic() {
    if (!voice.listening) baseRef.current = value;
    voice.toggle();
  }

  return (
    <div className="bg-background/80 border-t px-4 py-3 backdrop-blur md:px-6">
      <div className="mx-auto max-w-3xl">
        {voice.error ? (
          <p className="text-destructive mb-2 text-xs">{voice.error}</p>
        ) : null}
        {voice.listening ? (
          <p className="text-primary mb-2 flex items-center gap-2 text-xs">
            <span className="relative flex size-2">
              <span className="bg-primary absolute inline-flex size-2 animate-ping rounded-full opacity-75" />
              <span className="bg-primary relative inline-flex size-2 rounded-full" />
            </span>
            Listening… speak now
          </p>
        ) : null}

        <div className="bg-card focus-within:border-ring/60 flex items-end gap-2 rounded-2xl border p-2 transition-colors">
          <Textarea
            ref={textareaRef}
            rows={1}
            value={value}
            disabled={disabled}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder ?? "Ask a question about your documents…"}
            className="max-h-[200px] min-h-0 flex-1 resize-none border-0 bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0 dark:bg-transparent"
          />

          {voice.supported ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleMic}
                  disabled={disabled}
                  aria-label={voice.listening ? "Stop listening" : "Dictate a question"}
                  className={voice.listening ? "bg-primary/15 text-primary" : ""}
                >
                  <Mic />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {voice.listening ? "Stop listening" : "Dictate a question"}
              </TooltipContent>
            </Tooltip>
          ) : null}

          {streaming ? (
            <Button
              variant="outline"
              size="icon"
              onClick={onStop}
              aria-label="Stop generating"
            >
              <Square />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={submit}
              disabled={disabled || !value.trim()}
              aria-label="Send"
            >
              <ArrowUp />
            </Button>
          )}
        </div>

        <p className="text-muted-foreground mt-2 px-1 text-[11px]">
          Enter to send, Shift+Enter for a new line.
          {voice.supported
            ? " Answers are grounded in your uploaded documents."
            : " Voice input needs Chrome or Edge."}
        </p>
      </div>
    </div>
  );
}
