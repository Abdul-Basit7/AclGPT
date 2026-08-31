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
- **Model picker** — available from the moment the app opens, before any question is sent; choose per chat from whichever Groq models your account can actually reach.
- **Web search where it exists** — the control is enabled only for models the provider can actually search with, and disabled with an explanation elsewhere. Pages consulted appear in the sources panel as links.
- **Local embeddings** — documents are embedded on your own machine by default, so indexing has no quota, needs no key and works offline.
- **Response timing** — every answer shows when it arrived, with generation time on hover.

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
- A **Groq API key** for the chat model ([get one](https://console.groq.com/keys))
- Optionally a **Google API key**, only if you switch embeddings back to `gemini-embedding-001` ([get one](https://aistudio.google.com/apikey)) — see [Embeddings](#embeddings)

Embeddings run locally by default, so the chat model is the only thing that needs a key. Voice needs no key. OAuth sign-in is optional.

## Setup

```bash
git clone <your-repo-url>
cd sourcery
make install
cp backend/.env.example backend/.env
```

Fill in `backend/.env`:

```ini
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=a_long_random_string
# Only needed if you set EMBEDDING_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key
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
| `EMBEDDING_PROVIDER` | `local` | `local` (no key, no quota) or `google` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any model `fastembed` supports |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Google embedding model, when provider is `google` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Document splitting |
| `RETRIEVAL_K` / `RETRIEVAL_FETCH_K` | `5` / `20` | Passages used, and the MMR candidate pool |
| `HISTORY_TURNS` | `6` | Conversation turns sent as context |
| `MAX_UPLOAD_MB` | `100` | Per-file upload limit |
| `LOCAL_EMBED_BATCH_SIZE` | `256` | Chunks per batch when embedding locally |
| `EMBED_BATCH_SIZE` | `8` | Chunks per request when provider is `google` |
| `EMBED_REQUESTS_PER_MINUTE` | `60` | Pacing for `google` only, charged per chunk |
| `EMBED_MAX_RETRIES` | `8` | Retries per batch when rate-limited |

Chat models are discovered from the Groq API at runtime and cached for ten minutes, so a retired model id does not break the app; unknown ids fall back to the first available. `FALLBACK_MODELS` in `backend/app/services/llm.py` is used only when the API cannot be reached.

## Customising the UI

The interface is [shadcn/ui](https://ui.shadcn.com) on Tailwind v4, using the stock neutral palette. Component source lives in `frontend/src/components/ui/` and is yours to edit. Add more with:

```bash
cd frontend && npx shadcn@latest add <component>
```

The colour palette is the `:root` and `.dark` token blocks at the top of `frontend/src/index.css`. Both themes are always defined, so changing a token updates light and dark together. A small inline script in `index.html` applies the stored theme before first paint.

## Embeddings

Retrieval works by comparing your question against vectors built from your
documents. Which model builds those vectors is the single biggest constraint on
how large a document can be indexed.

**Local is the default.** `fastembed` runs `BAAI/bge-small-en-v1.5` as an ONNX
model in the API process: no key, no quota, no network, and the model downloads
once (~67 MB) on first use. Measured on an M-series Mac:

| Model | Dimensions | Speed | 1,280-chunk PDF |
| --- | --- | --- | --- |
| `BAAI/bge-small-en-v1.5` (default) | 384 | ~11 chunks/sec | ~2 min |
| `BAAI/bge-base-en-v1.5` | 768 | ~4 chunks/sec | ~5 min |

bge-base scores about one point higher on retrieval benchmarks for 2.7× the
indexing time, which is why small is the default. Set `LOCAL_EMBEDDING_MODEL` to
any model `fastembed` supports.

**The remote option, and why it is no longer the default.** Setting
`EMBEDDING_PROVIDER=google` uses `gemini-embedding-001`, which does score higher
on retrieval benchmarks. Its free tier has two quotas, and only one can be worked
around:

| Quota | Free-tier limit | Can pacing help? |
| --- | --- | --- |
| `EmbedContentRequestsPerMinute` | ~100 | Yes — this is what pacing and retries are for |
| `EmbedContentRequestsPerDay` | **1,000** | No |

Each chunk is one request, so **a free key can embed roughly 1,000 chunks per
day in total, across every document** — about 125 pages of dense PDF. A
1,280-chunk document cannot be indexed on a free key in one day, however
patiently it is paced. That is the ceiling local embeddings remove entirely.

The remote path still batches, paces per chunk (the quota counts documents, not
HTTP calls, so a batch of 8 spends 8 units), retries per-minute failures using
the delay the provider asks for, and fails a daily-quota error immediately rather
than burning minutes on retries that cannot succeed. Either way, progress is
committed after every batch so the UI shows a real percentage, and a document
that ultimately fails has its partial vectors rolled back, so the index never
holds chunks no document references.

### Changing the model means re-indexing

Vectors are only comparable within one model's vector space. Switching models
does not degrade retrieval, it invalidates it — and because dimensions can happen
to match, the failure is silent. Each index therefore records which model built
it in `embedding.json`, and searching an index built by a different model is
refused with an explanation.

Uploaded files are kept, so re-indexing needs no re-uploading:

```bash
cd backend && .venv/bin/python scripts/reindex.py             # everything
cd backend && .venv/bin/python scripts/reindex.py --collection 3
```

## Web search

Web search is a property of the model, not a setting the app can apply to
anything. Groq performs search and page fetching server side, and only for its
Compound systems — `groq/compound` and `groq/compound-mini`. Every other model
can answer only from its training data.

So the control is enabled for those two and disabled elsewhere, with a tooltip
saying why, and the server clears the flag if you switch to a model that cannot
search. The capability is read from the advertised model list rather than a
second constant, so what the UI offers and what the server enforces cannot drift
apart. Pages actually consulted are reported by the provider in `executed_tools`
and shown in the sources panel as links, rather than being parsed back out of the
answer text.

**On a small Groq tier, expect searches to fail sometimes.** Groq returns `413
Request Entity Too Large` when a *single* request exceeds the account's
per-minute token allowance, and a Compound run pulls whole pages into the prompt
— one long page is enough. Measured on a free tier, `groq/compound-mini`
succeeded on 2 of 3 questions where `groq/compound` managed 1 of 3, because mini
makes fewer tool calls; mini is listed first for that reason. The failure is
translated into an explanation rather than shown as a raw status code, which
would read like an upload-size problem.

## Deploying

One Docker image serves the API and the built frontend together. See
[DEPLOYMENT.md](DEPLOYMENT.md) for the full walkthrough; the short version is
that FAISS indexes and SQLite are files on disk, so the deployment needs a
persistent volume and exactly one instance. Most free tiers give neither.

```bash
docker build -t sourcery:local .
docker run --rm -p 8099:8000 -v sourcery-data:/data \
  -e GROQ_API_KEY=your_key -e SECRET_KEY=local sourcery:local
```

## Notes and limits

- **Provider model retirement is a live issue.** `text-embedding-004` was withdrawn during development, and Groq rotates chat model ids. Chat models self-heal via runtime discovery. Local embeddings are pinned to a downloaded model and cannot be retired underneath you, which is a further argument for the default.
- **Indexing speed is now CPU-bound, not quota-bound.** Roughly 11 chunks/sec, so a very large PDF takes minutes rather than failing. See [Embeddings](#embeddings).
- **Scanned PDFs need OCR first.** Image-only pages yield no text; the document is marked `failed` with that explanation rather than silently indexing nothing.
- **Uploaded text never leaves the machine by default.** Only the retrieved passages go out, in the prompt to Groq. Setting `EMBEDDING_PROVIDER=google` sends document text to Google as well.
- **Web search is unreliable on a small Groq tier.** See [Web search](#web-search).
- **Account linking requires a verified email.** A provider reporting an unverified address is refused, so nobody can claim an existing account by registering its address elsewhere.
- **SQLite suits a small deployment.** For many concurrent writers, move to Postgres; the FAISS indexes would then want shared storage or a hosted vector database.
- **Ingest runs in-process** as a FastAPI background task. Fine for modest files; a large corpus wants a real task queue.

## License

MIT — see [LICENSE](LICENSE).
