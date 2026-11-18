"""Hybrid retrieval over the criptolex corpus.

Two independent rankers run in parallel:

* **BM25** — lexical, exact match on legal terms (``PLAFT``, ``NCG 502``).
* **Dense** — local MiniLM embeddings, semantic paraphrase matching.

Their rankings are merged with Reciprocal Rank Fusion and the merged candidates
are reranked with a cross-encoder (FlashRank). Everything runs locally: no API
keys and no per-query cost.

Usage:
    uv run python -m criptolex.retrieve "¿pago impuestos si vendo bitcoin?"
    uv run python -m criptolex.retrieve "plazo para informar incidentes" --no-rerank
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
import snowballstemmer
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

from .config import get_settings

COLLECTION_NAME = "criptolex_chunks_v2"
RRF_K = 60
DENSE_CANDIDATES = 20
SPARSE_CANDIDATES = 20
RERANK_CANDIDATES = 20

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STEMMER = snowballstemmer.stemmer("spanish")
STOPWORDS = frozenset(
    """
    a al algo algunas algunos ante antes como con contra cual cuando de del desde
    donde durante e el ella ellas ellos en entre era erais eran eras eres es esa
    esas ese eso esos esta estaba estado estamos estan estar estas este esto estos
    fue fueron fui ha haber habia han hasta hay la las le les lo los mas me mi
    mientras muy nada ni no nos nosotros o os otra otras otro otros para pero poco
    por porque que quien se sea segun ser si sin sobre son su sus tambien tanto te
    tiene todo todos tu tus un una uno unos y ya
    """.split()
)


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    text: str
    source_key: str
    article: str | None
    score: float
    stage: str

    @property
    def snippet(self) -> str:
        body = self.text.partition("\n\n")[2] or self.text
        return " ".join(body.split())[:240]

    def label(self) -> str:
        if self.article:
            return f"{self.source_key} · {self.article}"
        return self.source_key


def index_text(text: str) -> str:
    """Body of a chunk, without the repeated citation header.

    Every chunk from a document shares the same header ("NCG 502 — Registro y
    obligaciones…"), so indexing it makes all chunks of that document look alike:
    for a query like "prestadores que deben inscribirse" it drowns the one chunk
    that actually answers. The header is still kept for display and citations.
    """
    return text.partition("\n\n")[2] or text


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, drop stopwords and reduce to Spanish stems.

    Stemming collapses legal morphology so that a query for ``obligaciones``
    also matches ``obligación``, and ``registro`` matches ``registrar``.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    tokens = [
        token for token in TOKEN_RE.findall(ascii_text) if len(token) > 2 and token not in STOPWORDS
    ]
    return STEMMER.stemWords(tokens)


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run python -m criptolex.chunk` first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class HybridRetriever:
    def __init__(self, chunks: list[dict], rebuild: bool = False) -> None:
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        self.by_id = {chunk["id"]: chunk for chunk in chunks}
        self.corpus = [tokenize(index_text(chunk["text"])) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus)
        self._ranker: Ranker | None = None

        settings = get_settings()
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))

        if rebuild:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:  # noqa: BLE001 - collection may not exist yet
                pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ONNXMiniLM_L6_V2(),
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_index()

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            settings = get_settings()
            cache_dir = settings.data_dir / "reranker"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(
                model_name=settings.rerank_model,
                cache_dir=str(cache_dir),
                log_level="ERROR",
            )
        return self._ranker

    def _ensure_index(self) -> None:
        if self.collection.count() == len(self.chunks):
            return

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ONNXMiniLM_L6_V2(),
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 128
        for start in range(0, len(self.chunks), batch_size):
            batch = self.chunks[start : start + batch_size]
            self.collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=[index_text(chunk["text"]) for chunk in batch],
                metadatas=[
                    {
                        "source_key": chunk["source_key"],
                        "article": chunk["article"] or "",
                        "doc_title": chunk["doc_title"],
                        "url": chunk["url"],
                        "year": chunk["year"],
                    }
                    for chunk in batch
                ],
            )

    def _dense(self, query: str, limit: int) -> list[Retrieved]:
        result = self.collection.query(query_texts=[query], n_results=limit)
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        hits: list[Retrieved] = []
        for chunk_id, _document, meta, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            hits.append(
                Retrieved(
                    chunk_id=chunk_id,
                    text=self.by_id[chunk_id]["text"],
                    source_key=meta["source_key"],
                    article=meta["article"] or None,
                    score=1.0 - float(distance),
                    stage="dense",
                )
            )
        return hits

    def _sparse(self, query: str, limit: int) -> list[Retrieved]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:limit]

        hits: list[Retrieved] = []
        for index in order:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[int(index)]
            hits.append(
                Retrieved(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    source_key=chunk["source_key"],
                    article=chunk["article"],
                    score=score,
                    stage="bm25",
                )
            )
        return hits

    @staticmethod
    def _fuse(rankings: list[list[Retrieved]], limit: int) -> list[Retrieved]:
        """Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank))."""
        scores: dict[str, float] = {}
        items: dict[str, Retrieved] = {}

        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                items.setdefault(item.chunk_id, item)

        ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
        return [
            Retrieved(
                chunk_id=chunk_id,
                text=items[chunk_id].text,
                source_key=items[chunk_id].source_key,
                article=items[chunk_id].article,
                score=score,
                stage="fused",
            )
            for chunk_id, score in ordered
        ]

    def _rerank(self, query: str, candidates: list[Retrieved], limit: int) -> list[Retrieved]:
        passages = [
            {"id": item.chunk_id, "text": item.text, "meta": {"source_key": item.source_key}}
            for item in candidates
        ]
        results = self.ranker.rerank(RerankRequest(query=query, passages=passages))

        hits: list[Retrieved] = []
        for result in results[:limit]:
            item = self.by_id[result["id"]]
            hits.append(
                Retrieved(
                    chunk_id=item["id"],
                    text=item["text"],
                    source_key=item["source_key"],
                    article=item["article"],
                    score=float(result["score"]),
                    stage="rerank",
                )
            )
        return hits

    def search(self, query: str, k: int = 5, use_rerank: bool = True) -> list[Retrieved]:
        """Retrieve the top-k chunks for a query.

        The reranker does not replace the fusion ranking: its ordering is fused
        with BM25 and dense rankings. A cross-encoder trained on English web
        passages is a strong signal but not an authority on Spanish legal text —
        fusing keeps a clearly relevant lexical match from being pushed out of
        the context window.
        """
        dense = self._dense(query, DENSE_CANDIDATES)
        sparse = self._sparse(query, SPARSE_CANDIDATES)
        fused = self._fuse([dense, sparse], RERANK_CANDIDATES)

        if not use_rerank:
            return fused[:k]

        reranked = self._rerank(query, fused, RERANK_CANDIDATES)
        return self._fuse([dense, sparse, reranked], k)

    def search_by_stage(self, query: str, k: int = 5) -> dict[str, list[Retrieved]]:
        dense = self._dense(query, DENSE_CANDIDATES)
        sparse = self._sparse(query, SPARSE_CANDIDATES)
        fused = self._fuse([dense, sparse], RERANK_CANDIDATES)
        reranked = self._rerank(query, fused, RERANK_CANDIDATES)
        final = self._fuse([dense, sparse, reranked], k)
        return {
            "bm25": sparse[:k],
            "dense": dense[:k],
            "fused": fused[:k],
            "rerank": reranked[:k],
            "final": final,
        }


def _print_results(title: str, results: list[Retrieved]) -> None:
    print(f"\n{title}")
    if not results:
        print("  (no results)")
        return
    for rank, item in enumerate(results, start=1):
        print(f"  {rank}. [{item.score:+.4f}] {item.label()}")
        print(f"     {item.snippet}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Hybrid retrieval over criptolex-cl")
    parser.add_argument("query", help="Question or keywords, in Spanish")
    parser.add_argument("-k", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip the cross-encoder reranking stage"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the vector index from scratch"
    )
    parser.add_argument("--compare", action="store_true", help="Show every stage side by side")
    args = parser.parse_args(argv)

    settings = get_settings()
    chunks = load_chunks(settings.processed_dir / "chunks.jsonl")
    retriever = HybridRetriever(chunks, rebuild=args.rebuild)

    print(f"Query: {args.query!r}  ({len(chunks)} chunks indexed)")
    if args.compare:
        for stage, results in retriever.search_by_stage(args.query, args.k).items():
            _print_results(f"── {stage.upper()} ──", results)
    else:
        results = retriever.search(args.query, args.k, use_rerank=not args.no_rerank)
        stage = "BM25 + dense + rerank" if not args.no_rerank else "BM25 + dense (fused)"
        _print_results(f"── {stage} ──", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
