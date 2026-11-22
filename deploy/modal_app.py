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

import json
import logging
import urllib.request

import modal

logger = logging.getLogger(__name__)

APP_NAME = "criptolex-cl"
SECRET_NAME = "groq-api-key"
GHCR_REPO = "daslaftorres/criptolex-cl"


def registry_digest(repository: str, tag: str = "latest") -> str:
    """Resolve the digest behind a GHCR tag so every deploy pins an exact image.

    Modal caches images by recipe, so ``:latest`` alone would keep reusing the
    first build it ever saw, silently running stale code after a redeploy.
    """
    scope = f"repository:{repository}:pull"
    token_url = f"https://ghcr.io/token?scope={scope}&service=ghcr.io"
    with urllib.request.urlopen(token_url, timeout=30) as response:  # noqa: S310
        token = json.load(response)["token"]
    request = urllib.request.Request(  # noqa: S310
        f"https://ghcr.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json, "
                "application/vnd.docker.distribution.manifest.list.v2+json"
            ),
        },
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.headers["Docker-Content-Digest"]


app = modal.App(APP_NAME)

try:
    IMAGE_REF = f"ghcr.io/{GHCR_REPO}@{registry_digest(GHCR_REPO)}"
except Exception:  # pragma: no cover - fall back to the tag if GHCR is unreachable
    logger.warning("could not resolve the GHCR digest, falling back to :latest", exc_info=True)
    IMAGE_REF = f"ghcr.io/{GHCR_REPO}:latest"

print(f"deploying {IMAGE_REF}")

# Same image the Dockerfile builds and the Docker workflow pushes. It runs as a
# non-root user and keeps the venv on PATH, which is all Modal needs.
image = modal.Image.from_registry(IMAGE_REF)


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
