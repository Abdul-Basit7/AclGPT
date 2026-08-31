import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

import { ApiError, api, streamMessage } from "@/api/client";
import type {
  Chat,
  Collection,
  Doc,
  Message,
  ModelInfo,
  Source,
} from "@/api/types";
import { AppSidebar } from "@/components/AppSidebar";
import { AuthPage } from "@/components/AuthPage";
import { ChatView } from "@/components/ChatView";
import { DocumentsPanel } from "@/components/DocumentsPanel";
import { SourcesPanel } from "@/components/SourcesPanel";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useAuth } from "@/hooks/useAuth";
import { useSpeech } from "@/hooks/useSpeech";

const POLL_MS = 1500;

/**
 * The shadcn sidebar writes its collapsed state to a cookie but reads it back
 * server-side, which a Vite app has no chance to do. Read it here so the choice
 * survives a reload.
 */
function sidebarDefaultOpen(): boolean {
  const match = /(?:^|;\s*)sidebar_state=(true|false)/.exec(document.cookie);
  return match ? match[1] === "true" : true;
}

export default function App() {
  const { ready, token, user } = useAuth();

  if (!ready) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center">
        <Loader2 className="size-6 animate-spin" />
      </div>
    );
  }

  if (!token || !user) return <AuthPage />;
  return <Workspace token={token} email={user.email} />;
}

function Workspace({ token, email }: { token: string; email: string }) {
  const { logout } = useAuth();
  const speech = useSpeech();

  const [collections, setCollections] = useState<Collection[]>([]);
  const [activeCollectionId, setActiveCollectionId] = useState<number | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  // Model and web-search choices made before any chat exists, so the controls
  // work from the moment the app opens rather than only after a first message.
  const [pendingModel, setPendingModel] = useState("");
  const [pendingWebSearch, setPendingWebSearch] = useState(false);

  const [loadingMessages, setLoadingMessages] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  // A failed turn leaves nothing in the transcript, so without this the user
  // sees sources and no answer with no explanation -- a toast has long gone by
  // the time they look. Web searches fail often enough for this to matter.
  const [turnError, setTurnError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const [documentsOpen, setDocumentsOpen] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [panelSources, setPanelSources] = useState<Source[]>([]);
  const [panelLabel, setPanelLabel] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const activeChat = useMemo(
    () => chats.find((c) => c.id === activeChatId) ?? null,
    [chats, activeChatId],
  );
  const activeCollection = useMemo(
    () => collections.find((c) => c.id === activeCollectionId),
    [collections, activeCollectionId],
  );

  const onApiError = useCallback(
    (err: unknown, fallback: string) => {
      if (err instanceof ApiError && err.status === 401) {
        logout();
        return;
      }
      toast.error(err instanceof Error ? err.message : fallback);
    },
    [logout],
  );

  // Initial load: collections, chats and the model list.
  useEffect(() => {
    let cancelled = false;
    Promise.all([api.listCollections(token), api.listChats(token), api.models()])
      .then(([nextCollections, nextChats, nextModels]) => {
        if (cancelled) return;
        setCollections(nextCollections);
        setChats(nextChats);
        setModels(nextModels);
        setPendingModel((current) => current || nextModels[0]?.id || "");
        const firstCollection = nextCollections[0]?.id ?? null;
        setActiveCollectionId(firstCollection);
        setActiveChatId(
          nextChats.find((c) => c.collection_id === firstCollection)?.id ?? null,
        );
      })
      .catch((err) => !cancelled && onApiError(err, "Could not load your workspace."));
    return () => {
      cancelled = true;
    };
  }, [token, onApiError]);

  // Messages for the selected chat.
  useEffect(() => {
    if (activeChatId === null) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    setLoadingMessages(true);
    api
      .listMessages(token, activeChatId)
      .then((next) => !cancelled && setMessages(next))
      .catch((err) => !cancelled && onApiError(err, "Could not load this chat."))
      .finally(() => !cancelled && setLoadingMessages(false));
    return () => {
      cancelled = true;
    };
  }, [token, activeChatId, onApiError]);

  const refreshDocuments = useCallback(
    async (collectionId: number) => {
      const [nextDocuments, nextCollections] = await Promise.all([
        api.listDocuments(token, collectionId),
        api.listCollections(token),
      ]);
      setDocuments(nextDocuments);
      setCollections(nextCollections);
      return nextDocuments;
    },
    [token],
  );

  // Documents for the selected collection.
  useEffect(() => {
    if (activeCollectionId === null) {
      setDocuments([]);
      return;
    }
    let cancelled = false;
    refreshDocuments(activeCollectionId).catch(
      (err) => !cancelled && onApiError(err, "Could not load documents."),
    );
    return () => {
      cancelled = true;
    };
  }, [activeCollectionId, refreshDocuments, onApiError]);

  // While anything is still being indexed, poll until it settles.
  useEffect(() => {
    const busy = documents.some(
      (d) => d.status === "pending" || d.status === "processing",
    );
    if (!busy || activeCollectionId === null) return;
    const timer = setInterval(() => {
      refreshDocuments(activeCollectionId).catch(() => undefined);
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [documents, activeCollectionId, refreshDocuments]);

  // What to ask next: follow-ups once a chat has turns, opening questions drawn
  // from the collection before that. Recomputed whenever the ground shifts.
  useEffect(() => {
    if (activeCollectionId === null) {
      setSuggestions([]);
      return;
    }
    // Chips from the previous turn are stale the moment a new one starts.
    if (streaming || loadingMessages) {
      setSuggestions([]);
      return;
    }

    let cancelled = false;
    setSuggestionsLoading(true);
    const request =
      activeChatId !== null && messages.length > 0
        ? api.chatSuggestions(token, activeChatId)
        : api.collectionSuggestions(token, activeCollectionId);

    request
      .then((result) => !cancelled && setSuggestions(result.suggestions))
      // A failed suggestion is not worth a toast; the composer just stays bare.
      .catch(() => !cancelled && setSuggestions([]))
      .finally(() => !cancelled && setSuggestionsLoading(false));

    return () => {
      cancelled = true;
    };
  }, [
    token,
    activeCollectionId,
    activeChatId,
    messages.length,
    documents.length,
    streaming,
    loadingMessages,
  ]);

  const createChat = useCallback(async () => {
    if (activeCollectionId === null) return null;
    const chat = await api.createChat(
      token,
      activeCollectionId,
      pendingModel || undefined,
      pendingWebSearch,
    );
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
    return chat;
  }, [token, activeCollectionId, pendingModel, pendingWebSearch]);

  const handleSend = useCallback(
    async (text: string) => {
      // Sending is what creates the first chat, so the model and web-search
      // choices made on an empty screen are carried into it.
      let chat = activeChat;
      if (!chat) {
        try {
          chat = await createChat();
        } catch (err) {
          onApiError(err, "Could not start a new chat.");
          return;
        }
      }
      if (!chat) return;
      speech.stop();

      const optimistic: Message = {
        id: -Date.now(),
        role: "user",
        content: text,
        sources: [],
        input_tokens: null,
        output_tokens: null,
        duration_ms: null,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, optimistic]);
      setStreaming(true);
      setStreamingText("");
      setTurnError(null);

      const controller = new AbortController();
      abortRef.current = controller;
      let answer = "";

      try {
        await streamMessage(
          token,
          chat.id,
          text,
          (event) => {
            if (event.type === "sources") {
              // Citations land in the right-hand panel as soon as retrieval ends.
              setPanelSources(event.sources);
              setPanelLabel(text.slice(0, 60));
              if (event.sources.length) setSourcesOpen(true);
            } else if (event.type === "token") {
              answer += event.text;
              setStreamingText(answer);
            } else if (event.type === "error") setTurnError(event.detail);
          },
          controller.signal,
        );
      } catch (err) {
        const aborted = err instanceof DOMException && err.name === "AbortError";
        if (!aborted) onApiError(err, "The answer could not be generated.");
      } finally {
        abortRef.current = null;
        setStreaming(false);
        setStreamingText("");

        // Replace the optimistic view with what the server actually stored.
        try {
          const stored = await api.listMessages(token, chat.id);
          setMessages(stored);
          const last = stored[stored.length - 1];
          if (autoSpeak && last?.role === "assistant") {
            speech.speak(`message-${last.id}`, last.content);
          }
        } catch (err) {
          onApiError(err, "Could not refresh this chat.");
        }
        api.listChats(token).then(setChats).catch(() => undefined);
      }
    },
    [activeChat, createChat, token, autoSpeak, speech, onApiError],
  );

  const handleEditMessage = useCallback(
    async (messageId: number, text: string) => {
      if (!activeChat || streaming) return;
      speech.stop();
      try {
        // Optimistic messages carry a negative id and were never stored, so
        // there is nothing on the server to rewind for them.
        if (messageId > 0) {
          await api.deleteMessagesFrom(token, activeChat.id, messageId);
        }
      } catch (err) {
        onApiError(err, "Could not rewind this chat.");
        return;
      }
      setMessages((prev) => {
        const index = prev.findIndex((m) => m.id === messageId);
        return index === -1 ? prev : prev.slice(0, index);
      });
      setPanelSources([]);
      setPanelLabel(null);
      await handleSend(text);
    },
    [token, activeChat, streaming, speech, handleSend, onApiError],
  );

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleShowSources = useCallback((sources: Source[], label: string) => {
    setPanelSources(sources);
    setPanelLabel(label);
    setSourcesOpen(true);
  }, []);

  const handleCopy = useCallback(async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      toast.success("Copied to clipboard");
    } catch {
      toast.error("Could not copy to the clipboard.");
    }
  }, []);

  const handleNewChat = useCallback(async () => {
    try {
      if (await createChat()) setMessages([]);
    } catch (err) {
      onApiError(err, "Could not start a new chat.");
    }
  }, [createChat, onApiError]);

  const handleDeleteChat = useCallback(
    async (chatId: number) => {
      try {
        await api.deleteChat(token, chatId);
        setChats((prev) => {
          const next = prev.filter((c) => c.id !== chatId);
          if (chatId === activeChatId) {
            setActiveChatId(
              next.find((c) => c.collection_id === activeCollectionId)?.id ?? null,
            );
          }
          return next;
        });
      } catch (err) {
        onApiError(err, "Could not delete that chat.");
      }
    },
    [token, activeChatId, activeCollectionId, onApiError],
  );

  const handleSelectCollection = useCallback(
    (collectionId: number) => {
      setActiveCollectionId(collectionId);
      setActiveChatId(chats.find((c) => c.collection_id === collectionId)?.id ?? null);
    },
    [chats],
  );

  const handleCreateCollection = useCallback(
    async (name: string) => {
      try {
        const collection = await api.createCollection(token, name);
        setCollections((prev) => [...prev, collection]);
        setActiveCollectionId(collection.id);
        setActiveChatId(null);
        toast.success(`Created “${collection.name}”`);
      } catch (err) {
        onApiError(err, "Could not create that collection.");
      }
    },
    [token, onApiError],
  );

  const handleDeleteCollection = useCallback(
    async (collectionId: number) => {
      try {
        await api.deleteCollection(token, collectionId);
        const remaining = collections.filter((c) => c.id !== collectionId);
        setCollections(remaining);
        setChats((prev) => prev.filter((c) => c.collection_id !== collectionId));
        setActiveCollectionId(remaining[0]?.id ?? null);
        setActiveChatId(null);
      } catch (err) {
        onApiError(err, "Could not delete that collection.");
      }
    },
    [token, collections, onApiError],
  );

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (activeCollectionId === null) return;
      setUploading(true);
      try {
        const created = await api.uploadDocuments(token, activeCollectionId, files);
        await refreshDocuments(activeCollectionId);
        toast.success(
          created.length === 1
            ? `Indexing “${created[0].filename}”`
            : `Indexing ${created.length} documents`,
        );
      } catch (err) {
        onApiError(err, "Upload failed.");
      } finally {
        setUploading(false);
      }
    },
    [token, activeCollectionId, refreshDocuments, onApiError],
  );

  const handleDeleteDocument = useCallback(
    async (documentId: number) => {
      if (activeCollectionId === null) return;
      try {
        await api.deleteDocument(token, activeCollectionId, documentId);
        await refreshDocuments(activeCollectionId);
      } catch (err) {
        onApiError(err, "Could not delete that document.");
      }
    },
    [token, activeCollectionId, refreshDocuments, onApiError],
  );

  const selectedModel = activeChat?.model ?? pendingModel;
  const webSearchOn = activeChat?.web_search ?? pendingWebSearch;

  const handleModelChange = useCallback(
    async (model: string) => {
      if (!activeChat) {
        setPendingModel(model);
        // Web search cannot carry over to a model that lacks it; the server
        // clears the flag too, so mirror that here instead of showing it on.
        if (!models.find((m) => m.id === model)?.supports_web_search) {
          setPendingWebSearch(false);
        }
        return;
      }
      try {
        const updated = await api.updateChat(token, activeChat.id, { model });
        setChats((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
      } catch (err) {
        onApiError(err, "Could not switch model.");
      }
    },
    [token, activeChat, models, onApiError],
  );

  const handleToggleWebSearch = useCallback(async () => {
    if (!activeChat) {
      setPendingWebSearch((v) => !v);
      return;
    }
    try {
      const updated = await api.updateChat(token, activeChat.id, {
        web_search: !activeChat.web_search,
      });
      setChats((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
    } catch (err) {
      onApiError(err, "Could not change the web search setting.");
    }
  }, [token, activeChat, onApiError]);

  return (
    /* h-svh, not the provider's default min-h-svh: without a fixed height the
       document itself scrolls, carrying the header and composer off-screen. */
    <SidebarProvider defaultOpen={sidebarDefaultOpen()} className="h-svh overflow-hidden">
      <AppSidebar
        email={email}
        collections={collections}
        activeCollectionId={activeCollectionId}
        chats={chats}
        activeChatId={activeChatId}
        onSelectCollection={handleSelectCollection}
        onCreateCollection={handleCreateCollection}
        onDeleteCollection={handleDeleteCollection}
        onNewChat={handleNewChat}
        onSelectChat={setActiveChatId}
        onDeleteChat={handleDeleteChat}
        onLogout={logout}
      />

      <SidebarInset className="flex min-w-0 flex-row overflow-hidden">
        <ChatView
          chat={activeChat}
          collection={activeCollection}
          messages={messages}
          models={models}
          streamingText={streamingText}
          streaming={streaming}
          loadingMessages={loadingMessages}
          documentCount={documents.length}
          turnError={turnError}
          onDismissError={() => setTurnError(null)}
          selectedModel={selectedModel}
          webSearch={webSearchOn}
          canCompose={activeCollectionId !== null}
          speechSupported={speech.supported}
          speakingKey={speech.speakingKey}
          autoSpeak={autoSpeak}
          sourcesOpen={sourcesOpen}
          suggestions={suggestions}
          suggestionsLoading={suggestionsLoading}
          onToggleAutoSpeak={() => {
            setAutoSpeak((v) => !v);
            speech.stop();
          }}
          onToggleSpeech={speech.toggle}
          onShowSources={handleShowSources}
          onToggleSourcesPanel={() => setSourcesOpen((v) => !v)}
          onCopy={handleCopy}
          onSend={handleSend}
          onEditMessage={handleEditMessage}
          onStop={handleStop}
          onModelChange={handleModelChange}
          onToggleWebSearch={handleToggleWebSearch}
          onOpenDocuments={() => setDocumentsOpen(true)}
          onNewChat={handleNewChat}
        />

        <SourcesPanel
          open={sourcesOpen}
          sources={panelSources}
          contextLabel={panelLabel}
          onClose={() => setSourcesOpen(false)}
        />
      </SidebarInset>

      <DocumentsPanel
        open={documentsOpen}
        collectionName={activeCollection?.name ?? ""}
        documents={documents}
        uploading={uploading}
        onOpenChange={setDocumentsOpen}
        onUpload={handleUpload}
        onDelete={handleDeleteDocument}
      />
    </SidebarProvider>
  );
}
