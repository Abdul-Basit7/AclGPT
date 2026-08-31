import { useEffect, useRef } from "react";
import { AlertCircle, FileText, Globe, PanelRight } from "lucide-react";

import type { Chat, Collection, Message, ModelInfo, Source } from "@/api/types";
import { Composer } from "@/components/Composer";
import { Logo } from "@/components/logo";
import { MessageItem } from "@/components/MessageItem";
import { ModeToggle } from "@/components/mode-toggle";
import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  /** Why the last answer failed, kept visible until the next question. */
  turnError: string | null;
  onDismissError: () => void;
  /** The model in force: the open chat's, or the choice awaiting a first chat. */
  selectedModel: string;
  webSearch: boolean;
  /** A collection exists, so a question can be asked even before a chat does. */
  canCompose: boolean;
  speechSupported: boolean;
  speakingKey: string | null;
  autoSpeak: boolean;
  sourcesOpen: boolean;
  /** Questions worth asking next, shown above the composer. */
  suggestions: string[];
  suggestionsLoading: boolean;
  onToggleAutoSpeak: () => void;
  onToggleSpeech: (key: string, text: string) => void;
  onShowSources: (sources: Source[], label: string) => void;
  onToggleSourcesPanel: () => void;
  onCopy: (text: string) => void;
  onSend: (text: string) => void;
  /** Re-ask an earlier question, discarding it and everything after it. */
  onEditMessage: (messageId: number, text: string) => void;
  onStop: () => void;
  onModelChange: (model: string) => void;
  onToggleWebSearch: () => void;
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
  turnError,
  onDismissError,
  selectedModel,
  webSearch,
  canCompose,
  speechSupported,
  speakingKey,
  autoSpeak,
  sourcesOpen,
  suggestions,
  suggestionsLoading,
  onToggleAutoSpeak,
  onToggleSpeech,
  onShowSources,
  onToggleSourcesPanel,
  onCopy,
  onSend,
  onEditMessage,
  onStop,
  onModelChange,
  onToggleWebSearch,
  onOpenDocuments,
  onNewChat,
}: Props) {
  const activeModel = models.find((model) => model.id === selectedModel);
  // Only Groq's Compound systems can search; the control stays visible but
  // disabled elsewhere, so the capability is discoverable rather than hidden.
  const canSearchWeb = activeModel?.supports_web_search ?? false;
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
    // turnError included so a failure notice scrolls into view like an answer.
  }, [messages.length, streamingText, turnError, chat?.id]);

  return (
    <div className="relative flex h-full min-w-0 flex-1 flex-col">
      <header className="glass-bar absolute inset-x-0 top-0 z-20 flex h-14 items-center gap-2 px-3 md:px-4">
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

        <Tooltip>
          <TooltipTrigger asChild>
            {/* A plain span keeps the tooltip reachable while the toggle is
                disabled: a disabled button emits no pointer events. */}
            <span className="hidden md:inline-flex">
              <Toggle
                pressed={webSearch && canSearchWeb}
                onPressedChange={onToggleWebSearch}
                disabled={!canSearchWeb}
                size="sm"
                aria-label="Web search"
              >
                <Globe />
                <span className="hidden lg:inline">Web</span>
              </Toggle>
            </span>
          </TooltipTrigger>
          <TooltipContent>
            {canSearchWeb
              ? webSearch
                ? "Web search on — answers may cite live pages"
                : "Let this model search the web"
              : `${activeModel?.label ?? "This model"} cannot search the web. Pick a Compound model to enable it.`}
          </TooltipContent>
        </Tooltip>

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
        <div className="mx-auto max-w-4xl space-y-5 px-4 pt-[4.75rem] pb-5 md:px-6">
          {!canCompose ? (
            <EmptyState title="No collection yet">
              Create a collection in the sidebar, then upload documents or ask a
              question.
              <span className="mt-4 block">
                <Button onClick={onNewChat}>New chat</Button>
              </span>
            </EmptyState>
          ) : null}

          {canCompose && !loadingMessages && messages.length === 0 && !streaming ? (
            <EmptyState title="Ask your first question">
              {documentCount === 0
                ? webSearch
                  ? "No documents yet — answers will come from the web, with links."
                  : "This collection has no documents yet. Upload one and every answer will cite it."
                : webSearch
                  ? "Answers draw on your documents first, then the web when they fall short."
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
              durationMs={message.duration_ms}
              createdAt={message.created_at}
              speechSupported={speechSupported}
              speaking={speakingKey === `message-${message.id}`}
              onToggleSpeech={() =>
                onToggleSpeech(`message-${message.id}`, message.content)
              }
              canEdit={!streaming}
              onShowSources={() =>
                onShowSources(
                  message.sources,
                  message.content.slice(0, 60).replace(/\s+/g, " "),
                )
              }
              onCopy={() => onCopy(message.content)}
              onEdit={
                message.role === "user"
                  ? (text) => onEditMessage(message.id, text)
                  : undefined
              }
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

          {turnError && !streaming ? (
            <Alert variant="destructive">
              <AlertCircle />
              <AlertTitle>That answer could not be completed</AlertTitle>
              <AlertDescription>{turnError}</AlertDescription>
              <AlertAction>
                <Button variant="ghost" size="xs" onClick={onDismissError}>
                  Dismiss
                </Button>
              </AlertAction>
            </Alert>
          ) : null}

          <div ref={bottomRef} />
        </div>
      </div>

      <Composer
        disabled={!canCompose}
        streaming={streaming}
        models={models}
        selectedModel={selectedModel}
        suggestions={suggestions}
        suggestionsLoading={suggestionsLoading}
        onModelChange={onModelChange}
        onSend={onSend}
        onStop={onStop}
        placeholder={
          documentCount === 0
            ? webSearch
              ? "Ask anything — the web is searchable…"
              : "Upload a document first, then ask away…"
            : "Ask a question about your documents…"
        }
        hint={
          documentCount === 0
            ? webSearch
              ? "Answers come from the web, with links."
              : "Upload documents and answers will cite them."
            : webSearch
              ? "Your documents first, then the web."
              : "Answers are grounded in your uploaded documents."
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
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
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
