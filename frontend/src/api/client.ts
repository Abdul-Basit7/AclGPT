import type {
  AuthResponse,
  Chat,
  Collection,
  Doc,
  Health,
  Message,
  ModelInfo,
  Provider,
  StreamEvent,
  Suggestions,
  User,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

type Options = {
  method?: string;
  body?: unknown;
  token?: string | null;
  form?: FormData;
};

async function request<T>(path: string, options: Options = {}): Promise<T> {
  const { method = "GET", body, token, form } = options;
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
  });

  if (!response.ok) throw new ApiError(response.status, await readError(response));
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function readError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    const detail = (data as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length) {
      // FastAPI validation errors
      const first = detail[0] as { msg?: string };
      if (first?.msg) return first.msg;
    }
    return JSON.stringify(data);
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

export const api = {
  register: (email: string, password: string) =>
    request<AuthResponse>("/auth/register", { method: "POST", body: { email, password } }),

  login: (email: string, password: string) =>
    request<AuthResponse>("/auth/login", { method: "POST", body: { email, password } }),

  me: (token: string) => request<User>("/auth/me", { token }),

  /** OAuth providers configured on the server; empty when none are set up. */
  providers: () => request<Provider[]>("/auth/providers"),

  /** Full-page navigation: the OAuth dance cannot happen inside fetch. */
  oauthStartUrl: (provider: string) => `${BASE}/auth/oauth/${provider}/start`,

  health: () => request<Health>("/health"),

  models: () => request<ModelInfo[]>("/models"),

  listCollections: (token: string) => request<Collection[]>("/collections", { token }),

  createCollection: (token: string, name: string) =>
    request<Collection>("/collections", { method: "POST", body: { name }, token }),

  renameCollection: (token: string, id: number, name: string) =>
    request<Collection>(`/collections/${id}`, { method: "PATCH", body: { name }, token }),

  deleteCollection: (token: string, id: number) =>
    request<void>(`/collections/${id}`, { method: "DELETE", token }),

  listDocuments: (token: string, collectionId: number) =>
    request<Doc[]>(`/collections/${collectionId}/documents`, { token }),

  uploadDocuments: (token: string, collectionId: number, files: File[]) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<Doc[]>(`/collections/${collectionId}/documents`, {
      method: "POST",
      form,
      token,
    });
  },

  deleteDocument: (token: string, collectionId: number, documentId: number) =>
    request<void>(`/collections/${collectionId}/documents/${documentId}`, {
      method: "DELETE",
      token,
    }),

  listChats: (token: string) => request<Chat[]>("/chats", { token }),

  createChat: (
    token: string,
    collectionId: number,
    model?: string,
    webSearch?: boolean,
  ) =>
    request<Chat>("/chats", {
      method: "POST",
      body: { collection_id: collectionId, model, web_search: webSearch ?? false },
      token,
    }),

  updateChat: (
    token: string,
    id: number,
    patch: { title?: string; model?: string; web_search?: boolean },
  ) =>
    request<Chat>(`/chats/${id}`, { method: "PATCH", body: patch, token }),

  deleteChat: (token: string, id: number) =>
    request<void>(`/chats/${id}`, { method: "DELETE", token }),

  listMessages: (token: string, chatId: number) =>
    request<Message[]>(`/chats/${chatId}/messages`, { token }),

  /** Follow-up questions for an open chat, drawn from its history and documents. */
  chatSuggestions: (token: string, chatId: number) =>
    request<Suggestions>(`/chats/${chatId}/suggestions`, { token }),

  /** Opening questions for a collection, before any chat exists to base them on. */
  collectionSuggestions: (token: string, collectionId: number) =>
    request<Suggestions>(`/collections/${collectionId}/suggestions`, { token }),

  /** Drop a message and every turn after it -- how an edited question rewinds. */
  deleteMessagesFrom: (token: string, chatId: number, messageId: number) =>
    request<void>(`/chats/${chatId}/messages/${messageId}`, {
      method: "DELETE",
      token,
    }),
};

/**
 * POST a question and consume the server-sent event stream.
 * Uses fetch rather than EventSource so the request can carry an auth header.
 */
export async function streamMessage(
  token: string,
  chatId: number,
  content: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE}/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ content }),
    signal,
  });

  if (!response.ok) throw new ApiError(response.status, await readError(response));
  if (!response.body) throw new ApiError(500, "The server returned an empty stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; keep any partial tail for the next read.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      } catch {
        // Ignore malformed frames rather than aborting a partly-received answer.
      }
    }
  }
}
