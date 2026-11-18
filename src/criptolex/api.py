"""FastAPI service for criptolex-cl.

Endpoints:
    GET  /health  liveness and configuration summary
    POST /ask     { question, k? } -> grounded answer + cited sources
    GET  /        single-page chat UI
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .generate import Answer, GroqClient, Source, extractive_answer, generate_answer
from .retrieve import HybridRetriever, Retrieved, load_chunks

logger = logging.getLogger(__name__)

APP_VERSION = "0.3.0"
AnswerFn = Callable[[str, list[Retrieved]], Answer]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=400)
    k: int = Field(default=5, ge=1, le=10)


class SourceOut(BaseModel):
    n: int
    source_key: str
    title: str
    issuer: str
    year: int
    article: str | None
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    grounded: bool
    refused: bool
    model: str
    latency_ms: int


def _source_out(source: Source) -> SourceOut:
    return SourceOut(
        n=source.n,
        source_key=source.source_key,
        title=source.title,
        issuer=source.issuer,
        year=source.year,
        article=source.article,
        url=source.url,
        snippet=source.snippet,
    )


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    settings = get_settings()
    chunks = load_chunks(settings.processed_dir / "chunks.jsonl")
    return HybridRetriever(chunks)


@lru_cache(maxsize=1)
def get_answerer() -> AnswerFn:
    settings = get_settings()
    if settings.llm_provider == "extractive":
        return extractive_answer
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured. Set it in .env, or run with "
                "LLM_PROVIDER=extractive for a no-LLM demo."
            ),
        )
    client = GroqClient(settings.groq_api_key, settings.llm_model, settings.llm_timeout)
    return lambda question, hits: generate_answer(question, hits, client)


def retriever_dep() -> HybridRetriever:
    return get_retriever()


def answerer_dep() -> AnswerFn:
    return get_answerer()


app = FastAPI(
    title="criptolex-cl",
    description="RAG assistant over Chilean crypto and fintech regulation.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "provider": settings.llm_provider,
        "llm_configured": bool(settings.groq_api_key) or settings.llm_provider == "extractive",
    }


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    retriever: Annotated[HybridRetriever, Depends(retriever_dep)],
    answerer: Annotated[AnswerFn, Depends(answerer_dep)],
) -> AskResponse:
    started = time.perf_counter()
    hits = retriever.search(payload.question, payload.k)

    try:
        answer = answerer(payload.question, hits)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - upstream failures become a clean 502
        logger.exception("generation failed")
        raise HTTPException(
            status_code=502, detail="No se pudo generar la respuesta. Intenta nuevamente."
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    return AskResponse(
        answer=answer.text,
        sources=[_source_out(source) for source in answer.sources],
        grounded=answer.grounded,
        refused=answer.refused,
        model=answer.model,
        latency_ms=latency_ms,
    )


def _web_dir():
    return get_settings().web_dir


if _web_dir().exists():
    app.mount("/static", StaticFiles(directory=_web_dir()), name="static")


@app.get("/", include_in_schema=False)
def index() -> Response:
    index_file = _web_dir() / "index.html"
    if not index_file.exists():
        return JSONResponse({"service": "criptolex-cl", "version": APP_VERSION})
    return FileResponse(index_file)
