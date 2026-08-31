import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import { Copy, Pencil, Quote, Volume2, VolumeX } from "lucide-react";

import type { Source } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface Props {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  inputTokens?: number | null;
  outputTokens?: number | null;
  durationMs?: number | null;
  createdAt?: string;
  streaming?: boolean;
  speechSupported: boolean;
  speaking: boolean;
  /** False while an answer is in flight: rewinding mid-stream would race it. */
  canEdit?: boolean;
  onToggleSpeech: () => void;
  onShowSources?: () => void;
  onCopy?: () => void;
  onEdit?: (text: string) => void;
}

export function MessageItem({
  role,
  content,
  sources = [],
  inputTokens,
  outputTokens,
  durationMs,
  createdAt,
  streaming = false,
  speechSupported,
  speaking,
  canEdit = false,
  onToggleSpeech,
  onShowSources,
  onCopy,
  onEdit,
}: Props) {
  const [editing, setEditing] = useState(false);

  if (role === "user") {
    if (editing && onEdit) {
      return (
        <MessageEditor
          initial={content}
          onCancel={() => setEditing(false)}
          onSubmit={(text) => {
            setEditing(false);
            onEdit(text);
          }}
        />
      );
    }

    return (
      <div className="group/msg flex items-center justify-end gap-1">
        <div className="flex items-center gap-0.5 opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100">
          {onEdit && canEdit ? (
            <IconAction
              label="Edit question"
              onClick={() => setEditing(true)}
              icon={<Pencil />}
            />
          ) : null}
          {onCopy ? (
            <IconAction label="Copy question" onClick={onCopy} icon={<Copy />} />
          ) : null}
        </div>
        <div className="bg-muted max-w-[85%] rounded-2xl rounded-br-md px-4 py-2.5 text-sm whitespace-pre-wrap">
          {content}
        </div>
      </div>
    );
  }

  const totalTokens = (inputTokens ?? 0) + (outputTokens ?? 0);
  const hasUsage = Boolean(inputTokens || outputTokens);
  const answeredAt = createdAt ? new Date(createdAt) : null;
  const validTime = answeredAt && !Number.isNaN(answeredAt.getTime());

  return (
    <div className="group/msg">
      <div className="prose-answer text-foreground text-sm">
        <Markdown
          remarkPlugins={[remarkGfm]}
          /*
            Models routinely emit inline HTML such as <br> inside table cells.
            rehype-raw parses it so it renders as intended; rehype-sanitize then
            strips anything unsafe. Order matters: raw before sanitize.
          */
          rehypePlugins={[rehypeRaw, rehypeSanitize]}
        >
          {content}
        </Markdown>
        {streaming ? <span className="caret" /> : null}
      </div>

      {!streaming ? (
        <div className="mt-1.5 flex items-center gap-1">
          {sources.length > 0 && onShowSources ? (
            <Button
              variant="ghost"
              size="xs"
              onClick={onShowSources}
              className="text-muted-foreground"
            >
              <Quote />
              {sources.length} source{sources.length === 1 ? "" : "s"}
            </Button>
          ) : null}

          <div className="flex items-center gap-0.5 opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100">
            {onCopy ? (
              <IconAction label="Copy answer" onClick={onCopy} icon={<Copy />} />
            ) : null}

            {speechSupported && content.trim() ? (
              <IconAction
                label={speaking ? "Stop reading" : "Read aloud"}
                onClick={onToggleSpeech}
                active={speaking}
                icon={speaking ? <VolumeX /> : <Volume2 />}
              />
            ) : null}
          </div>

          {validTime || hasUsage ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-muted-foreground ml-auto cursor-default font-mono text-[11px] tabular-nums">
                  {validTime ? formatClock(answeredAt) : null}
                  {validTime && hasUsage ? " · " : null}
                  {hasUsage ? `${totalTokens.toLocaleString()} tok` : null}
                </span>
              </TooltipTrigger>
              <TooltipContent className="space-y-0.5">
                {validTime ? <div>{formatFull(answeredAt)}</div> : null}
                {durationMs ? <div>Generated in {formatDuration(durationMs)}</div> : null}
                {hasUsage ? (
                  <div>
                    {(inputTokens ?? 0).toLocaleString()} in ·{" "}
                    {(outputTokens ?? 0).toLocaleString()} out
                  </div>
                ) : null}
              </TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function IconAction({
  label,
  icon,
  active = false,
  onClick,
}: {
  label: string;
  icon: React.ReactNode;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClick}
          aria-label={label}
          className={`size-7 ${active ? "text-foreground" : "text-muted-foreground"}`}
        >
          {icon}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

/** In-place editor for a question already asked; submitting re-runs the turn. */
function MessageEditor({
  initial,
  onSubmit,
  onCancel,
}: {
  initial: string;
  onSubmit: (text: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    node.focus();
    node.setSelectionRange(node.value.length, node.value.length);
    node.style.height = "auto";
    node.style.height = `${node.scrollHeight}px`;
  }, []);

  function submit() {
    const text = value.trim();
    if (!text) return;
    onSubmit(text);
  }

  return (
    <div className="flex justify-end">
      <div className="bg-muted w-full max-w-[85%] rounded-2xl rounded-br-md p-2">
        <Textarea
          ref={ref}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${e.target.scrollHeight}px`;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              onCancel();
            }
          }}
          className="min-h-0 resize-none border-0 bg-transparent px-2 py-1 text-sm shadow-none focus-visible:ring-0 dark:bg-transparent"
        />
        <div className="mt-1 flex justify-end gap-2 px-1">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button size="sm" onClick={submit} disabled={!value.trim()}>
            Send
          </Button>
        </div>
      </div>
    </div>
  );
}

/** 12-hour clock with am/pm, regardless of what the viewer's locale defaults to. */
function formatClock(value: Date): string {
  return safeFormat(value, {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

/**
 * Formatting a timestamp must never be able to take the page down. Intl throws
 * on an option combination it dislikes, and a thrown error here unmounts the
 * entire conversation, so fall back to something readable instead.
 */
function safeFormat(value: Date, options: Intl.DateTimeFormatOptions): string {
  try {
    return value.toLocaleString(undefined, options);
  } catch {
    return value.toISOString().replace("T", " ").slice(0, 19);
  }
}

function formatFull(value: Date): string {
  // Component options only. Intl forbids mixing `dateStyle`/`timeStyle` with
  // individual fields like `hour`, and throws rather than ignoring the clash --
  // which, with no error boundary above, blanks the whole app.
  return safeFormat(value, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const seconds = ms / 1000;
  return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}
