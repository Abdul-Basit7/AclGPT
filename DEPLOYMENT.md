# Deploying Sourcery

One Docker image holds both halves: the browser app is built and then served by
the API process, so there is a single deploy, a single URL and no CORS to
configure.

## The constraint that decides everything

Sourcery keeps four things on disk under `DATA_DIR`:

| | |
| --- | --- |
| `sourcery.db` | users, collections, chats, messages |
| `uploads/` | the original files, kept so they can be re-indexed |
| `indexes/` | one FAISS index per collection |

FAISS indexes are held in the process's memory and guarded by in-process locks,
so **exactly one instance may run**, and it needs **persistent storage**. That
rules out most free tiers, which have an ephemeral filesystem: everything above
is wiped on each redeploy.

It also means you cannot scale this horizontally. Two instances would race each
other's writes and answer from different data. Fixing that properly means moving
to Postgres and a hosted vector database — a different project, and unnecessary
for personal use.

## Which host

| Host | Free? | Data survives a redeploy? | Verdict |
| --- | --- | --- | --- |
| **Fly.io** | No, about $2–3/month | **Yes**, with a volume | Use this if you want to keep anything |
| **Render** | Yes | **No** — ephemeral, and free instances cannot mount a disk | Fine for a demo URL, nothing more |
| **Render Starter** | $7/month | Yes, with a disk | Simplest paid option |

Render's free plan also sleeps after inactivity, so the first request after a
quiet spell takes a while to answer.

## Fly.io

```bash
brew install flyctl && fly auth login

fly launch --no-deploy --copy-config          # reads fly.toml
fly volumes create sourcery_data --size 1 --region lhr
fly secrets set GROQ_API_KEY=... SECRET_KEY="$(openssl rand -hex 32)"
fly deploy
```

Then set `FRONTEND_URL` and `CORS_ORIGINS` in `fly.toml` to the real URL and
deploy again. Getting `FRONTEND_URL` wrong does not break the app; it breaks
OAuth sign-in specifically, because that is where the provider redirects back to.

**Do not raise `min_machines_running` above 1**, for the reasons above.

## Render

Push `render.yaml`, then in the dashboard: New → Blueprint → pick the repo. Set
`GROQ_API_KEY` there; never in the file. To keep data, switch the plan to
`starter` and uncomment the `disk` block.

## Environment variables

Only two are required:

| Variable | Required | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | **yes** | The chat model |
| `SECRET_KEY` | **yes** | Signs sign-in tokens. `openssl rand -hex 32`. Changing it logs everybody out. |
| `DATA_DIR` | no | `/data` in the image; point it at your volume |
| `FRONTEND_URL` | for OAuth | Must equal the deployed URL |
| `GOOGLE_API_KEY` | no | Only if `EMBEDDING_PROVIDER=google` |
| `*_OAUTH_CLIENT_ID` / `_SECRET` | no | Omit and that button is simply not shown |

Embeddings run locally, so there is no second API key to obtain and no quota to
exhaust.

## After deploying, register the OAuth redirect URIs

In the Google Cloud and GitHub consoles, add:

```
https://<your-domain>/api/auth/oauth/google/callback
https://<your-domain>/api/auth/oauth/github/callback
```

They must match exactly — scheme, host and path. A mismatch is the most common
OAuth failure, and the error message is unhelpful.

If you deploy behind a proxy that terminates TLS, run uvicorn with
`--proxy-headers`. Without it the callback URL is built as `http://` and the
provider rejects it.

## Running the image locally

Worth doing before deploying, because it catches problems the dev servers hide.

```bash
docker build -t sourcery:local .

docker run --rm -p 8099:8000 \
  -v sourcery-data:/data \
  -e GROQ_API_KEY=your_key \
  -e SECRET_KEY=anything-for-local \
  sourcery:local
```

Open http://localhost:8099. The named volume means data survives
`docker restart`, exactly as it will in production.

## Memory

Measured against a 512MB container, which is what a free host gives you:

| | |
| --- | --- |
| Idle | ~200MB |
| Answering a question | ~400MB |
| Ingesting a 3,000-chunk document | right at the limit |

`LOCAL_EMBED_BATCH_SIZE` is the knob. At the old default of 256 a large ingest is
**OOM-killed**; at 32 it completes, at the same ~10 chunks/sec, because the model
rather than the batch size is the bottleneck. 32 is now the default. Lower it
further if a host kills the process, raise it only where memory is plentiful.

Note the interaction with the stuck-ingest bug below: if the process is killed
mid-ingest, the document is left in `processing` for ever, with no way to retry
from the UI.

## Notes on the image

- **Python 3.11+ with `faiss-cpu` 1.13.** Version 1.9 fails to import on Linux
  arm64 with NumPy 1.x, because it falls back to `numpy.distutils`, which does
  not exist on Python 3.12. It works on macOS, so this only appears once you
  containerise — which is a good argument for building the image early.
- **The embedding model is downloaded at build time** into `/opt/models`, with
  `EMBEDDING_CACHE_DIR` pointing at it. fastembed otherwise caches to a temp
  directory that a container wipes on restart, so every cold start would
  re-download 67 MB and a host without access to Hugging Face would fail.
- **Runs as a non-root user** that owns `DATA_DIR`.
- **One uvicorn worker,** deliberately. See the constraint above.

## What is not production-ready

Honest list, in the order it will matter:

1. **No backups.** The volume is the only copy. `fly ssh console` plus `sqlite3
   .backup` is the crude version.
2. **A restart mid-ingest leaves a document stuck in `processing` forever.**
   Ingest is an in-process background task with no recovery on startup.
3. **No rate limiting on sign-in**, so password guessing is unthrottled.
4. **Sign-in tokens last 7 days and cannot be revoked.** Logging out only clears
   the browser's copy.
5. **The OAuth state is not bound to the browser**, so a sign-in started by
   someone else can be completed by you. Fix before inviting other people.
