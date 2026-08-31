import { useEffect, useRef, useState } from "react";
import { ArrowUp, Globe, Mic, Sparkles, Square } from "lucide-react";

import type { ModelInfo } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useVoiceInput } from "@/hooks/useVoiceInput";

interface Props {
  disabled: boolean;
  streaming: boolean;
  models: ModelInfo[];
  selectedModel: string;
  /** Questions worth asking next; empty while none have been generated. */
  suggestions: string[];
  suggestionsLoading: boolean;
  onModelChange: (model: string) => void;
  onSend: (text: string) => void;
  onStop: () => void;
  placeholder?: string;
  /** One short sentence on where answers will come from in the current mode. */
  hint?: string;
}

const MAX_HEIGHT_PX = 200;

export function Composer({
  disabled,
  streaming,
  models,
  selectedModel,
  suggestions,
  suggestionsLoading,
  onModelChange,
  onSend,
  onStop,
  placeholder,
  hint,
}: Props) {
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

  function send(text: string) {
    if (!text || disabled || streaming) return;
    onSend(text);
    setValue("");
    baseRef.current = "";
    voice.stop();
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(value.trim());
    }
  }

  function toggleMic() {
    if (!voice.listening) baseRef.current = value;
    voice.toggle();
  }

  // Suggestions are only useful on an idle, empty composer: mid-question they
  // would compete with what the user is already typing.
  const showSuggestions =
    !disabled && !streaming && !value.trim() && suggestions.length > 0;

  return (
    <div className="bg-background/80 shrink-0 border-t px-4 py-2.5 backdrop-blur md:px-6">
      <div className="mx-auto max-w-4xl">
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

        {showSuggestions ? (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => send(suggestion)}
                className="bg-card text-muted-foreground hover:text-foreground hover:border-ring/60 flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition-colors"
              >
                <Sparkles className="size-3 shrink-0 opacity-60" />
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}

        <div className="bg-card focus-within:border-ring/60 rounded-2xl border p-2 transition-colors">
          <Textarea
            ref={textareaRef}
            rows={1}
            value={value}
            disabled={disabled}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder ?? "Ask a question about your documents…"}
            className="max-h-[200px] min-h-0 w-full resize-none border-0 bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0 dark:bg-transparent"
          />

          {/* The model belongs with the question it will answer, not up in the
              title bar next to the workspace controls. */}
          <div className="flex items-center gap-1 pt-1 pl-1">
            {models.length > 0 ? (
              <Select value={selectedModel} onValueChange={onModelChange}>
                <SelectTrigger
                  size="sm"
                  aria-label="Model"
                  className="text-muted-foreground hover:text-foreground h-7 max-w-[12rem] border-0 bg-transparent px-2 text-xs shadow-none dark:bg-transparent"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent align="start">
                  {models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <span className="flex items-center gap-1.5">
                        {model.label}
                        {model.supports_web_search ? (
                          <Globe className="text-muted-foreground size-3" />
                        ) : null}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : null}

            {suggestionsLoading && !showSuggestions ? (
              <span className="text-muted-foreground hidden text-[11px] sm:inline">
                Thinking of follow-ups…
              </span>
            ) : null}

            <div className="ml-auto flex items-center gap-1">
              {voice.supported ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={toggleMic}
                      disabled={disabled}
                      aria-label={
                        voice.listening ? "Stop listening" : "Dictate a question"
                      }
                      className={
                        voice.listening ? "bg-primary/15 text-primary size-8" : "size-8"
                      }
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
                  className="size-8"
                >
                  <Square />
                </Button>
              ) : (
                <Button
                  size="icon"
                  onClick={() => send(value.trim())}
                  disabled={disabled || !value.trim()}
                  aria-label="Send"
                  className="size-8"
                >
                  <ArrowUp />
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Where answers come from changes with the mode, so the caller says so
            rather than this component assuming documents. */}
        <p className="text-muted-foreground mt-1.5 px-1 text-[11px]">
          {hint ?? "Enter to send, Shift+Enter for a new line."}
          {voice.supported ? null : " Voice input needs Chrome or Edge."}
        </p>
      </div>
    </div>
  );
}
