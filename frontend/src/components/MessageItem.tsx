import Markdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import { Copy, Quote, Volume2, VolumeX } from "lucide-react";

import type { Source } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface Props {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  inputTokens?: number | null;
  outputTokens?: number | null;
  streaming?: boolean;
  speechSupported: boolean;
  speaking: boolean;
  onToggleSpeech: () => void;
  onShowSources?: () => void;
  onCopy?: () => void;
}

export function MessageItem({
  role,
  content,
  sources = [],
  inputTokens,
  outputTokens,
  streaming = false,
  speechSupported,
  speaking,
  onToggleSpeech,
  onShowSources,
  onCopy,
}: Props) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-muted max-w-[85%] rounded-2xl rounded-br-md px-4 py-2.5 text-sm whitespace-pre-wrap">
          {content}
        </div>
      </div>
    );
  }

  const totalTokens = (inputTokens ?? 0) + (outputTokens ?? 0);
  const hasUsage = Boolean(inputTokens || outputTokens);

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
        <div className="mt-2.5 flex items-center gap-1">
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

          <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100">
            {onCopy ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onCopy}
                    aria-label="Copy answer"
                    className="text-muted-foreground size-7"
                  >
                    <Copy />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Copy answer</TooltipContent>
              </Tooltip>
            ) : null}

            {speechSupported && content.trim() ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onToggleSpeech}
                    aria-label={speaking ? "Stop reading" : "Read aloud"}
                    className={`size-7 ${speaking ? "text-foreground" : "text-muted-foreground"}`}
                  >
                    {speaking ? <VolumeX /> : <Volume2 />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {speaking ? "Stop reading" : "Read aloud"}
                </TooltipContent>
              </Tooltip>
            ) : null}
          </div>

          {hasUsage ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-muted-foreground ml-auto cursor-default font-mono text-[11px] tabular-nums">
                  {totalTokens.toLocaleString()} tok
                </span>
              </TooltipTrigger>
              <TooltipContent>
                {(inputTokens ?? 0).toLocaleString()} in ·{" "}
                {(outputTokens ?? 0).toLocaleString()} out
              </TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
