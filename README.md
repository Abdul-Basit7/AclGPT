# Sourcery

Ask your documents. Get answers with receipts.

Upload PDFs, spreadsheets, CSV, JSON, Word files, HTML or plain text; Sourcery indexes them, retrieves the relevant passages, and streams back an answer that cites the file and page it came from. Nothing is invented — if the answer is not in your documents, it says so.

Sign in with Google, GitHub or an email address. Voice input and output use the browser's built-in speech APIs, so the voice features cost nothing and need no extra service.

## Features

- **Grounded answers with citations** — every answer is restricted to retrieved passages and cites `filename, page N`. The sources panel shows the exact snippets used.
- **Document upload** — drag and drop PDF, DOCX, XLSX, CSV, TSV, JSON, JSONL, YAML, HTML, XML, Markdown or plain text. Files are chunked, embedded and indexed in the background, with a live progress bar per file.
- **Collections** — group documents into separate knowledge bases, each with its own vector index and chats.
- **Streaming responses** — answers arrive token by token over server-sent events, and can be stopped mid-generation.
- **Token accounting** — every answer shows the tokens it cost, with the input/output split on hover.
- **Docked citations** — sources open in a right-hand panel that stays put while you read, collapsing to a sheet on narrow screens.
- **Collapsible sidebar** — minimise it to an icon rail with the header toggle or ⌘/Ctrl+B; the choice survives a reload.
- **Sign in your way** — Google and GitHub OAuth, or email and password. Signing in with a provider links to an existing account when the email matches and is verified.
- **Light, dark and system themes** — switchable from the app or the sign-in page, remembered per browser, with no flash on load.
- **Voice, both directions** — dictate questions with the microphone, and have answers read aloud, with an optional auto-read.
- **Persistent history** — chats and messages live in SQLite and survive restarts.
- **Model picker** — choose per chat from whichever Groq models your account can actually reach.

## Architecture

```
backend/                    FastAPI + SQLAlchemy + FAISS + Alembic
  app/
    main.py                 app factory, CORS, static frontend mount
    config.py               settings from env / .env
    database.py             engine, session, migration bootstrap
    models.py               User, OAuthAccount, Collection, Document, Chat, Message
    schemas.py              Pydantic request/response models
    security.py             bcrypt hashing, JWT access + OAuth state tokens
    deps.py                 auth and ownership dependencies
    routers/
      auth.py               register, login, me, OAuth start/callback
      collections.py        collection CRUD
      documents.py          upload, list, delete
      chats.py              chat CRUD + SSE message streaming
      meta.py               health, model list
    services/
      extract.py            PDF / DOCX / XLSX / CSV / JSON / YAML / HTML / text
      ingest.py             chunk, paced+retried embedding, progress, rollback
      vectorstore.py        per-collection FAISS index, locked writes
      rag.py                retrieval, prompt assembly, token streaming
      llm.py                Groq model discovery with a static fallback
      oauth.py              provider registry, code exchange, profile fetch
  alembic/versions/         schema migrations
  tests/                    73 tests: auth, OAuth, ingest, formats, RAG, isolation

frontend/                   Vite + React 19 + TypeScript + Tailwind v4 + shadcn/ui
  src/
    api/client.ts           typed API client + SSE reader
    hooks/useAuth.tsx       token storage, session validation, OAuth callback
    hooks/useSpeech.ts      text-to-speech (SpeechSynthesis)
    hooks/useVoiceInput.ts  speech-to-text (SpeechRecognition)
    components/ui/          shadcn primitives (owned, editable)
    components/             AuthPage, AppSidebar, ChatView, Composer, MessageItem,
                            SourcesPanel, DocumentsPanel, mode-toggle
    App.tsx                 workspace state and handlers
```

**How a question is answered.** The frontend POSTs to `/api/chats/{id}/messages`. The backend stores the question, runs MMR retrieval against the collection's FAISS index, builds a system prompt containing only the retrieved passages plus recent history, then streams the model's tokens back as SSE frames (`sources`, then `token`, then `done`). The assistant message and its sources are persisted once the stream finishes.

**Storage.** SQLite holds users, OAuth links, collections, document metadata, chats and messages. Each collection gets its own FAISS index under `backend/data/indexes/<id>`. Writes are serialised per collection and staged through a temp directory, so an interrupted write cannot corrupt an index. Each document records the vector ids it produced, so deleting one document removes exactly its own vectors.

## Requirements

- Python 3.9 or newer
- Node.js 18 or newer
- A **Google API key** for `gemini-embedding-001` document embeddings ([get one](https://aistudio.google.com/apikey)). A paid key is worth it for large documents — see [Indexing large documents](#indexing-large-documents).
- A **Groq API key** for the chat model ([get one](https://console.groq.com/keys))

Both providers have free tiers. Voice needs no key. OAuth sign-in is optional.

## Setup

```bash
git clone <your-repo-url>
cd sourcery
make install
cp backend/.env.example backend/.env
```

Fill in `backend/.env`:

```ini
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=a_long_random_string
```

Generate a real `SECRET_KEY` before deploying — it signs both session and OAuth state tokens:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Optional: OAuth sign-in

Leave a provider's credentials blank and its button simply is not shown — the UI only offers providers the server can actually complete.

**Google** — [Cloud Console → Credentials](https://console.cloud.google.com/apis/credentials) → Create OAuth client ID → Web application. Add this **authorised redirect URI**:

```
http://127.0.0.1:8000/api/auth/oauth/google/callback
```

**GitHub** — [Developer settings → OAuth Apps](https://github.com/settings/developers) → New OAuth App. Set the **authorization callback URL**:

```
http://127.0.0.1:8000/api/auth/oauth/github/callback
```

Then add the pairs you want to `backend/.env`:

```ini
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
GITHUB_OAUTH_CLIENT_ID=...
GITHUB_OAUTH_CLIENT_SECRET=...
```

The redirect URI must match exactly, including the host. In development the Vite proxy forwards with `changeOrigin`, so the backend sees `127.0.0.1:8000` — use that rather than `localhost`. In production set `FRONTEND_URL` and register the public callback URL instead.

## Running

Development, with hot reload on both sides:

```bash
make dev
```

Open <http://localhost:5173>. Vite proxies `/api` to the backend on port 8000, so there is no CORS to configure.

Single process — build the frontend and let FastAPI serve it:

```bash
make build
cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API docs are at `/docs`.

## Migrations

Schema changes are Alembic revisions in `backend/alembic/versions/`, applied automatically on startup — no manual step for normal use.

A database created before Alembic existed has the tables but no version table; startup detects that, stamps it at the initial revision, then upgrades. Existing accounts and chats are preserved.

```bash
cd backend
.venv/bin/alembic revision -m "describe the change"   # after editing models.py
.venv/bin/alembic upgrade head
.venv/bin/alembic downgrade -1
```

SQLite cannot `ALTER COLUMN`, so `render_as_batch` is enabled: column changes rebuild the table and copy rows. Keep using `op.batch_alter_table` for those.

## Testing

```bash
make test     # 73 backend tests
make lint     # frontend typecheck
```

The suite runs against a throwaway data directory with fake embeddings, a fake chat model, stubbed model discovery and a mocked OAuth round trip, so it needs no API keys and makes no network calls. It covers:

- **Auth** — registration, login, token rejection, and per-user isolation of collections, documents and chats.
- **OAuth** — provider advertisement, redirect construction, account creation and linking, refusal of unverified provider emails, and rejection of tampered, expired, cross-provider and replayed state tokens.
- **Ingest** — PDF page metadata is 1-indexed, DOCX/XLSX/CSV/TSV/JSON/JSONL/YAML/HTML/Markdown/TXT extraction, delimiter sniffing, script and style stripping, row truncation, rejection of unsupported and empty files.
- **Large documents** — batching, per-document rate pacing, retry with the provider's suggested delay, fail-fast on non-retryable errors, and rollback of partially embedded chunks.
- **Chat** — SSE streaming with sources, history accumulation, and graceful handling of an LLM outage.

## Voice

Both directions use the Web Speech API built into the browser.

- **Output** (`SpeechSynthesis`) works in every current browser. Markdown is stripped first so the voice reads prose, not punctuation.
- **Input** (`SpeechRecognition`) is implemented in Chrome and Edge. Safari and Firefox generally are not, so the microphone button hides itself and the composer explains why.

Nothing is downloaded and no audio goes to a paid service. For consistent voices everywhere, the alternative is local models — `faster-whisper` for input, Piper for output — behind two new endpoints.

## Configuration

All optional, set in `backend/.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | `dev-secret-change-me` | Signs session and OAuth state tokens. Change it. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | Session length (7 days) |
| `FRONTEND_URL` | `http://localhost:5173` | Where OAuth callbacks redirect back to |
| `OAUTH_STATE_TTL_SECONDS` | `600` | How long a sign-in attempt stays valid |
| `DATA_DIR` | `backend/data` | SQLite file, uploads and indexes |
| `CORS_ORIGINS` | `http://localhost:5173,…` | Allowed origins, comma separated |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Google embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Document splitting |
| `RETRIEVAL_K` / `RETRIEVAL_FETCH_K` | `5` / `20` | Passages used, and the MMR candidate pool |
| `HISTORY_TURNS` | `6` | Conversation turns sent as context |
| `MAX_UPLOAD_MB` | `100` | Per-file upload limit |
| `EMBED_BATCH_SIZE` | `8` | Chunks per embedding request (free tier rejects 16+) |
| `EMBED_REQUESTS_PER_MINUTE` | `60` | Client-side pacing, charged per chunk |
| `EMBED_MAX_RETRIES` | `8` | Retries per batch when rate-limited |

Chat models are discovered from the Groq API at runtime and cached for ten minutes, so a retired model id does not break the app; unknown ids fall back to the first available. `FALLBACK_MODELS` in `backend/app/services/llm.py` is used only when the API cannot be reached.

## Customising the UI

The interface is [shadcn/ui](https://ui.shadcn.com) on Tailwind v4, using the stock neutral palette. Component source lives in `frontend/src/components/ui/` and is yours to edit. Add more with:

```bash
cd frontend && npx shadcn@latest add <component>
```

The colour palette is the `:root` and `.dark` token blocks at the top of `frontend/src/index.css`. Both themes are always defined, so changing a token updates light and dark together. A small inline script in `index.html` applies the stored theme before first paint.

## Indexing large documents

Large files used to fail outright: a 160-page PDF is roughly 1,280 chunks, and
embedding them in one call returned `429` within five seconds.

The constraint is Google's **per-minute** embedding quota on the free tier, and
it counts **documents, not HTTP calls** — a batch of 8 chunks spends 8 units, not
1. Pacing therefore has to be charged per chunk, which is what
`EMBED_REQUESTS_PER_MINUTE` does. Retries also spend quota, since a retried batch
re-sends every chunk in it.

Sourcery now embeds in small batches, paces per chunk, retries rate-limit
failures using the delay the provider asks for, commits progress after every
batch so the UI can show a real percentage, and rolls back partially embedded
chunks if a document ultimately fails — so the index never holds vectors that no
document references.

### The free tier has a hard daily ceiling

There are two separate quotas, and only one of them is something software can
work around:

| Quota | Free-tier limit | Can pacing help? |
| --- | --- | --- |
| `EmbedContentRequestsPerMinute` | ~100 | Yes — this is what pacing and retries are for |
| `EmbedContentRequestsPerDay` | **1,000** | No |

Each chunk is one request, so **a free key can embed roughly 1,000 chunks per
day, in total, across every document.** That is about 125 pages of dense PDF. A
1,280-chunk document simply cannot be indexed on a free key in one day, no matter
how patiently it is paced.

Sourcery tells the two apart via the provider's `quota_id`: a per-minute limit is
retried with backoff, while a daily limit fails immediately with an explanation
rather than burning several minutes in retries that cannot succeed.

For anything beyond occasional small documents, use a paid Google key and raise
`EMBED_REQUESTS_PER_MINUTE`. Other levers: raise `CHUNK_SIZE` so the same
document needs fewer requests, or split the file across days.

## Notes and limits

- **Provider model retirement is a live issue.** `text-embedding-004` was withdrawn during development, and Groq rotates chat model ids. Chat models self-heal via discovery; embeddings do not. If ingest starts failing with a 404 naming the model, list what your key can reach with `curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY"` and set `EMBEDDING_MODEL`. Changing it changes the vector dimensions, so delete `backend/data/indexes/` and re-upload.
- **Very large documents are slow on a free key.** See [Indexing large documents](#indexing-large-documents). A file that exhausts the retry budget is marked `failed`, and its partial vectors are removed so the index never holds chunks no document references.
- **Scanned PDFs need OCR first.** Image-only pages yield no text; the document is marked `failed` with that explanation rather than silently indexing nothing.
- **Uploaded text is sent to Google** for embedding. If that is unacceptable, swap `get_embeddings()` in `backend/app/services/vectorstore.py` for a local model such as `fastembed` — it is the only place embeddings are constructed.
- **Account linking requires a verified email.** A provider reporting an unverified address is refused, so nobody can claim an existing account by registering its address elsewhere.
- **SQLite suits a small deployment.** For many concurrent writers, move to Postgres; the FAISS indexes would then want shared storage or a hosted vector database.
- **Ingest runs in-process** as a FastAPI background task. Fine for modest files; a large corpus wants a real task queue.

## License

MIT — see [LICENSE](LICENSE).
