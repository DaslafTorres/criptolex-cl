"""Retrieval probe: a small hand-checked set of answerable questions.

This is not the final evaluation harness (that is M4, with citation-level
metrics over generated answers). It exists to make retrieval decisions
reproducible: every question has a verifiable gold chunk, so a change to
chunking, embeddings or fusion can be measured instead of guessed.

Run:
    uv run python eval/probe_retrieval.py
    uv run python eval/probe_retrieval.py --no-rerank
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from criptolex.config import get_settings
from criptolex.retrieve import HybridRetriever, load_chunks

PROBES: list[dict[str, str]] = [
    {
        "id": "inscripcion",
        "question": "¿Qué prestadores deben inscribirse en el registro de la CMF?",
        "source_key": "ley_21521",
        "article": "Artículo 5",
    },
    {
        "id": "vigencia",
        "question": "¿La Ley Fintech entra en vigencia de inmediato?",
        "source_key": "ley_21521",
        "article": "Transitorio primero",
    },
    {
        "id": "definicion",
        "question": "¿Cuál es la definición de activo virtual en la Ley Fintech?",
        "source_key": "ley_21521",
        "article": "Artículo 3",
    },
    {
        "id": "incidentes",
        "question": "¿Cuándo debe informarse un incidente de ciberseguridad a la CMF?",
        "chunk_id": "ncg_502_cmf:0138",
    },
    {
        "id": "mayor_valor",
        "question": "¿Cómo se calcula el mayor valor en la venta de criptomonedas?",
        "source_key": "sii_circular_43_2021",
        "contains": "mayor valor",
    },
    {
        "id": "uaf_reporte",
        "question": "¿Qué operaciones sospechosas debe reportar una casa de cambio a la UAF?",
        "source_key": "circular_62_uaf",
        "contains": "sospechos",
    },
    {
        "id": "sospechosa_def",
        "question": "¿Qué se entiende por operación sospechosa?",
        "chunk_id": "ley_19913:0008",
    },
    {
        "id": "impuestos",
        "question": "¿Las ganancias por vender criptomonedas se declaran en Chile?",
        "source_key": "sii_faq_cripto",
    },
    {
        "id": "sanciones",
        "question": "¿Qué sanciones se aplican por infracciones a la ley de lavado de activos?",
        "source_key": "ley_19913",
        "article": "Artículo 20",
    },
    {
        "id": "dj1964",
        "question": "¿Quiénes están obligados a presentar la declaración jurada de criptoactivos?",
        "chunk_id": "sii_dj1964:0001",
    },
    {
        "id": "requisitos",
        "question": "¿Qué antecedentes se exigen para solicitar la inscripción en el registro?",
        "chunk_id": "ncg_502_cmf:0009",
    },
]


def is_gold(chunk: dict, probe: dict[str, str]) -> bool:
    if "chunk_id" in probe:
        return chunk["id"] == probe["chunk_id"]
    if chunk["source_key"] != probe["source_key"]:
        return False
    if "article" in probe and chunk["article"] != probe["article"]:
        return False
    if "contains" in probe and probe["contains"] not in chunk["text"].lower():
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Retrieval probe over criptolex-cl")
    parser.add_argument("-k", type=int, default=5, help="Cutoff (default 5)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder stage")
    args = parser.parse_args(argv)

    chunks = load_chunks(get_settings().processed_dir / "chunks.jsonl")
    retriever = HybridRetriever(chunks)
    retriever.search("warmup", 1)

    ranks: list[int | None] = []
    latencies: list[float] = []

    print(f"{'probe':14} {'rank':>5}  top-1")
    for probe in PROBES:
        started = time.perf_counter()
        hits = retriever.search(probe["question"], args.k, use_rerank=not args.no_rerank)
        latencies.append((time.perf_counter() - started) * 1000)

        gold = {chunk["id"] for chunk in chunks if is_gold(chunk, probe)}
        rank = next((index for index, hit in enumerate(hits, 1) if hit.chunk_id in gold), None)
        ranks.append(rank)
        print(f"{probe['id']:14} {str(rank):>5}  {hits[0].label()}")

    hit_at_k = sum(1 for rank in ranks if rank and rank <= args.k) / len(ranks)
    mrr = statistics.mean([1 / rank if rank else 0.0 for rank in ranks])
    print(
        f"\nhit@{args.k}={hit_at_k:.2f}  MRR={mrr:.2f}  "
        f"latency={statistics.mean(latencies):.0f} ms/query  n={len(PROBES)}"
    )
    return 0 if hit_at_k == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
