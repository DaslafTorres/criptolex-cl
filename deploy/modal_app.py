"""Modal deployment for criptolex-cl.

The container image is the one GitHub Actions already publishes to GHCR, so the
corpus, the vector index and the reranker weights are baked in exactly as they
are in production. Modal only adds the HTTP front end and the Groq secret, which
keeps this file small and avoids a second, drifting build.

Deploy with:

    modal secret create groq-api-key GROQ_API_KEY=gsk_...
    modal deploy deploy/modal_app.py
"""

from __future__ import annotations

import logging

import modal

logger = logging.getLogger(__name__)

APP_NAME = "criptolex-cl"
SECRET_NAME = "groq-api-key"

app = modal.App(APP_NAME)

# Same image the Dockerfile builds and the Docker workflow pushes. It runs as a
# non-root user and keeps the venv on PATH, which is all Modal needs.
image = modal.Image.from_registry("ghcr.io/daslaftorres/criptolex-cl:latest")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    cpu=2.0,
    memory=4096,
    # Keep the container warm for five minutes after the last request: a demo
    # visitor usually asks two or three questions in a row.
    scaledown_window=300,
    timeout=600,
)
@modal.asgi_app()
def web():
    from criptolex.api import app as api_app
    from criptolex.api import get_retriever

    # Load Chroma, BM25 and the cross-encoder before the container is marked
    # ready, so the first visitor pays the cold start instead of a slow answer.
    try:
        get_retriever().search("warmup", 1)
    except Exception:  # pragma: no cover - warmup must never block the deploy
        logger.warning("retriever warmup failed", exc_info=True)

    return api_app
