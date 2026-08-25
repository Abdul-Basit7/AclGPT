import { useEffect, useRef } from "react";
import { FileText, PanelRight } from "lucide-react";

import type { Chat, Collection, Message, ModelInfo, Source } from "@/api/types";
import { Composer } from "@/components/Composer";
import { Logo } from "@/components/logo";
import { MessageItem } from "@/components/MessageItem";
import { ModeToggle } from "@/components/mode-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Toggle } from "@/components/ui/toggle";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface Props {
  chat: Chat | null;
  collection: Collection | undefined;
  messages: Message[];
  models: ModelInfo[];
  streamingText: string;
  streaming: boolean;
  loadingMessages: boolean;
  documentCount: number;
  speechSupported: boolean;
  speakingKey: string | null;
  autoSpeak: boolean;
  sourcesOpen: boolean;
  onToggleAutoSpeak: () => void;
  onToggleSpeech: (key: string, text: string) => void;
  onShowSources: (sources: Source[], label: string) => void;
  onToggleSourcesPanel: () => void;
  onCopy: (text: string) => void;
  onSend: (text: string) => void;
  onStop: () => void;
  onModelChange: (model: string) => void;
  onOpenDocuments: () => void;
  onNewChat: () => void;
}

export function ChatView({
  chat,
  collection,
  messages,
  models,
  streamingText,
  streaming,
  loadingMessages,
  documentCount,
  speechSupported,
  speakingKey,
  autoSpeak,
  sourcesOpen,
  onToggleAutoSpeak,
  onToggleSpeech,
  onShowSources,
  onToggleSourcesPanel,
  onCopy,
  onSend,
  onStop,
  onModelChange,
  onOpenDocuments,
  onNewChat,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, streamingText, chat?.id]);

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col">
      <header className="bg-background flex h-14 shrink-0 items-center gap-2 border-b px-3 md:px-4">
        <SidebarTrigger />
        <Separator orientation="vertical" className="mr-1 h-5" />

        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-semibold">
            {chat ? chat.title || "New chat" : "Sourcery"}
          </h1>
          <p className="text-muted-foreground truncate text-xs">
            {collection ? collection.name : "No collection"}
            {documentCount > 0
              ? ` · ${documentCount} document${documentCount === 1 ? "" : "s"}`
              : " · no documents"}
          </p>
        </div>

        {speechSupported ? (
          <Toggle
            pressed={autoSpeak}
            onPressedChange={onToggleAutoSpeak}
            size="sm"
            className="hidden md:inline-flex"
          >
            Auto read
          </Toggle>
        ) : null}

        {chat ? (
          <Select value={chat.model} onValueChange={onModelChange}>
            <SelectTrigger size="sm" className="hidden max-w-[10rem] md:flex" aria-label="Model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : null}

        <ModeToggle />

        <Button variant="outline" size="sm" onClick={onOpenDocuments}>
          <FileText />
          <span className="hidden sm:inline">Documents</span>
          <Badge variant="secondary">{documentCount}</Badge>
        </Button>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleSourcesPanel}
              aria-label="Toggle sources panel"
              aria-pressed={sourcesOpen}
              className={sourcesOpen ? "bg-muted" : ""}
            >
              <PanelRight />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {sourcesOpen ? "Hide sources" : "Show sources"}
          </TooltipContent>
        </Tooltip>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-8 px-4 py-8 md:px-6">
          {!chat ? (
            <EmptyState title="No chat selected">
              Start a new chat to ask questions about the documents in this collection.
              <span className="mt-4 block">
                <Button onClick={onNewChat}>New chat</Button>
              </span>
            </EmptyState>
          ) : null}

          {chat && !loadingMessages && messages.length === 0 && !streaming ? (
            <EmptyState title="Ask your first question">
              {documentCount === 0
                ? "This collection has no documents yet. Upload one and every answer will cite it."
                : "Answers are drawn only from your uploaded documents, with page citations."}
            </EmptyState>
          ) : null}

          {messages.map((message) => (
            <MessageItem
              key={message.id}
              role={message.role}
              content={message.content}
              sources={message.sources}
              inputTokens={message.input_tokens}
              outputTokens={message.output_tokens}
              speechSupported={speechSupported}
              speaking={speakingKey === `message-${message.id}`}
              onToggleSpeech={() =>
                onToggleSpeech(`message-${message.id}`, message.content)
              }
              onShowSources={() =>
                onShowSources(
                  message.sources,
                  message.content.slice(0, 60).replace(/\s+/g, " "),
                )
              }
              onCopy={() => onCopy(message.content)}
            />
          ))}

          {streaming ? (
            streamingText ? (
              <MessageItem
                role="assistant"
                content={streamingText}
                streaming
                speechSupported={false}
                speaking={false}
                onToggleSpeech={() => undefined}
              />
            ) : (
              <p className="text-muted-foreground flex items-center gap-2 text-sm">
                Searching your documents
                <span className="caret" />
              </p>
            )
          ) : null}

          <div ref={bottomRef} />
        </div>
      </div>

      <Composer
        disabled={!chat}
        streaming={streaming}
        onSend={onSend}
        onStop={onStop}
        placeholder={
          documentCount === 0
            ? "Upload a document first, then ask away…"
            : "Ask a question about your documents…"
        }
      />
    </div>
  );
}

function EmptyState({
  title,
  children,
}: {
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <div className="bg-muted text-muted-foreground mb-3 flex size-11 items-center justify-center rounded-xl">
        <Logo className="size-5" />
      </div>
      <h3 className="text-sm font-semibold">{title}</h3>
      {children ? (
        <p className="text-muted-foreground mt-1.5 max-w-sm text-sm">{children}</p>
      ) : null}
    </div>
  );
}
