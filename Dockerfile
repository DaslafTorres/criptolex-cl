# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOME=/app \
    PORT=7860 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

# Dependencies first so code changes do not invalidate the install layer.
# README.md is required here: pyproject.toml declares it as the project readme.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY web ./web

# Bake the corpus, the vector index and the local models into the image.
# Without this the container would download 7 documents plus ~85 MB of ONNX
# weights on the first request, which is exactly when a visitor is waiting.
RUN uv run python -m criptolex.ingest \
    && uv run python -m criptolex.chunk \
    && uv run python -c "from criptolex.config import get_settings; \
from criptolex.retrieve import HybridRetriever, load_chunks; \
r = HybridRetriever(load_chunks(get_settings().processed_dir / 'chunks.jsonl')); \
r.search('warmup', 1); \
print(r.collection.count(), 'chunks indexed')"

RUN useradd --home-dir /home/app --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn criptolex.api:app --host 0.0.0.0 --port ${PORT}"]
