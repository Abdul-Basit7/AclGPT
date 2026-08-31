export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

/** An OAuth provider the server has credentials for. */
export interface Provider {
  id: string;
  label: string;
}

export interface Collection {
  id: number;
  name: string;
  created_at: string;
  document_count: number;
  ready_count: number;
}

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface Doc {
  id: number;
  collection_id: number;
  filename: string;
  content_type: string;
  size_bytes: number;
  pages: number;
  chunk_count: number;
  chunks_embedded: number;
  status: DocumentStatus;
  error: string | null;
  created_at: string;
}

export interface Chat {
  id: number;
  collection_id: number;
  title: string;
  model: string;
  web_search: boolean;
  created_at: string;
  updated_at: string;
}

export interface Source {
  document_id: number | null;
  filename: string;
  page: number | null;
  snippet: string;
  /** Set for web results; null for a passage from an uploaded document. */
  url: string | null;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  input_tokens: number | null;
  output_tokens: number | null;
  duration_ms: number | null;
  created_at: string;
}

export interface ModelInfo {
  id: string;
  label: string;
  /** False for models the provider cannot web search with. */
  supports_web_search: boolean;
}

export interface Health {
  status: string;
  google_key_configured: boolean;
  groq_key_configured: boolean;
  models: ModelInfo[];
}

export type StreamEvent =
  | { type: "user_message"; id: number }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | {
      type: "usage";
      input_tokens: number | null;
      output_tokens: number | null;
      total_tokens: number | null;
    }
  | { type: "error"; detail: string }
  | {
      type: "done";
      message_id: number | null;
      sources: Source[];
      failed: boolean;
      input_tokens: number | null;
      output_tokens: number | null;
      duration_ms: number | null;
    };

export interface Suggestions {
  suggestions: string[];
}
