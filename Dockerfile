# Build the browser app first, then serve it from the API process. One image and
# one deploy, and because both are same-origin there is no CORS to configure.

FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS api

# Model files live here, baked in at build time. fastembed would otherwise
# default to a temp directory and re-download 67 MB on every cold start.
ENV EMBEDDING_CACHE_DIR=/opt/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Download the embedding model into the image. Doing this here rather than on
# first request means a cold start serves immediately and the container needs no
# access to Hugging Face at runtime.
ARG LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding(model_name='${LOCAL_EMBEDDING_MODEL}', cache_dir='/opt/models')" \
    && chmod -R a+rX /opt/models

COPY backend/ ./backend/
# main.py looks for the built frontend one level up from the backend package.
COPY --from=frontend /build/dist ./frontend/dist

# Uploads, the SQLite file and the FAISS indexes all live here. Mount a volume
# at this path or they are lost on redeploy.
ENV DATA_DIR=/data
RUN mkdir -p /data

# Run as a non-root user; it still needs to write to the data directory.
RUN useradd --create-home --uid 10001 sourcery && chown -R sourcery /data
USER sourcery

EXPOSE 8000
WORKDIR /app/backend

# Single worker on purpose: the FAISS indexes are held in this process's memory
# and guarded by in-process locks, so a second worker would race the first.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
