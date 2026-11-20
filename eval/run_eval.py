"""Golden-set evaluation for criptolex-cl.

Measures retrieval and, optionally, generated answers against
``eval/golden.jsonl``. Every question was written from the source text (not from
a retrieval result), and every expected fact is checked against the gold chunk,
so the set cannot silently drift away from the corpus.

    uv run python eval/run_eval.py                  # retrieval only, no API key
    uv run python eval/run_eval.py --generate       # + citation and faithfulness
    uv run python eval/run_eval.py --json eval/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from criptolex.config import get_settings
from criptolex.generate import (
    Answer,
    GroqClient,
    extractive_answer,
    generate_answer,
)
from criptolex.retrieve import HybridRetriever, load_chunks

# Groq list prices for qwen/qwen3.8-27b, USD per million tokens.
INPUT_USD_PER_MTOK = 0.80
OUTPUT_USD_PER_MTOK = 4.00


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(ascii_text.split())


def is_gold(chunk: dict, gold: dict) -> bool:
    if "chunk_ids" in gold:
        return chunk["id"] in gold["chunk_ids"]
    if chunk["source_key"] != gold.get("source_key"):
        return False
    return "article" not in gold or chunk["article"] == gold["article"]


def hit_is_gold(item, gold: dict) -> bool:
    """Same rule as :func:`is_gold`, for a ``Retrieved`` or ``Source`` object."""
    if "chunk_ids" in gold:
        return item.chunk_id in gold["chunk_ids"]
    if item.source_key != gold.get("source_key"):
        return False
    return "article" not in gold or item.article == gold["article"]


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(golden: list[dict], chunks: list[dict]) -> None:
    """Fail loudly if the golden set and the corpus have drifted apart."""
    problems: list[str] = []
    for item in golden:
        gold = item.get("gold")
        if not gold:
            continue
        matched = [chunk for chunk in chunks if is_gold(chunk, gold)]
        if not matched:
            problems.append(f"{item['id']}: no chunk matches {gold}")
            continue
        text = normalize(" ".join(chunk["text"] for chunk in matched))
        for fact in item.get("facts", []):
            if normalize(fact) not in text:
                problems.append(f"{item['id']}: fact not in gold text: {fact!r}")
    if problems:
        raise SystemExit("Golden set invalid:\n  " + "\n  ".join(problems))


def make_answerer():
    settings = get_settings()
    if settings.llm_provider == "extractive":
        return extractive_answer, None
    if not settings.groq_api_key:
        raise SystemExit("No GROQ_API_KEY. Run retrieval-only, or set LLM_PROVIDER=extractive.")
    client = GroqClient(
        settings.groq_api_key,
        settings.llm_model,
        settings.llm_timeout,
        settings.llm_max_tokens,
    )

    def answer(question: str, hits) -> Answer:
        return generate_answer(question, hits, client)

    return answer, client


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Golden-set evaluation")
    parser.add_argument("-k", type=int, default=5, help="Retrieval cutoff (default 5)")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also generate answers and measure citations and facts",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write full results to this JSON file"
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only N questions")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    golden = load_golden(root / "eval" / "golden.jsonl")
    if args.limit:
        golden = golden[: args.limit]

    chunks = load_chunks(get_settings().processed_dir / "chunks.jsonl")
    validate(golden, chunks)
    retriever = HybridRetriever(chunks)
    retriever.search("warmup", 1)

    answerer = None
    if args.generate:
        answerer, client = make_answerer()
    else:
        client = None

    rows: list[dict] = []
    print(f"{'id':24} {'gold':>5} {'refused':>7} {'cites':>5} {'facts':>5}")
    for item in golden:
        started = time.perf_counter()
        hits = retriever.search(item["question"], args.k)
        answer_latency = 0.0

        gold = item.get("gold")
        rank = None
        if gold:
            rank = next(
                (index for index, hit in enumerate(hits, 1) if hit_is_gold(hit, gold)),
                None,
            )

        answer = None
        if answerer is not None:
            answer_started = time.perf_counter()
            answer = answerer(item["question"], hits)
            answer_latency = (time.perf_counter() - answer_started) * 1000

        total_latency = (time.perf_counter() - started) * 1000

        cited = [source.chunk_id for source in answer.sources] if answer else []
        cited_gold = 0
        if gold and answer:
            cited_gold = sum(1 for source in answer.sources if hit_is_gold(source, gold))
        facts = item.get("facts", [])
        fact_hits = 0
        if answer and facts:
            normalized = normalize(answer.text)
            fact_hits = sum(1 for fact in facts if normalize(fact) in normalized)

        row = {
            "id": item["id"],
            "answerable": item["answerable"],
            "gold_rank": rank,
            "refused": answer.refused if answer else None,
            "citations": cited,
            "cited_gold": cited_gold,
            "facts_expected": len(facts),
            "facts_found": fact_hits,
            "latency_ms": round(total_latency),
            "answer_latency_ms": round(answer_latency),
            "answer": answer.text if answer else None,
        }
        rows.append(row)

        refused = "-" if answer is None else ("yes" if answer.refused else "no")
        print(
            f"{item['id']:24} {str(rank):>5} {refused:>7} "
            f"{len(cited):>5} {f'{fact_hits}/{len(facts)}':>5}"
        )

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    ranks = [row["gold_rank"] for row in answerable]
    latencies = [row["latency_ms"] for row in rows]

    summary: dict[str, float | int | None] = {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        f"retrieval_hit@{args.k}": sum(1 for r in ranks if r and r <= args.k) / len(ranks),
        f"retrieval_mrr@{args.k}": statistics.mean(
            [1 / r if r and r <= args.k else 0.0 for r in ranks]
        ),
        "latency_p50_ms": round(percentile(latencies, 0.5)),
        "latency_p95_ms": round(percentile(latencies, 0.95)),
    }

    if answerer is not None:
        generated = [row for row in answerable]
        cited_counts = [len(row["citations"]) for row in generated]
        summary.update(
            {
                "citation_precision": sum(row["cited_gold"] for row in generated)
                / max(1, sum(cited_counts)),
                "citation_recall": sum(1 for row in generated if row["cited_gold"])
                / len(generated),
                "fact_recall": sum(row["facts_found"] for row in generated)
                / max(1, sum(row["facts_expected"] for row in generated)),
                "grounded_rate": sum(1 for row in generated if row["citations"]) / len(generated),
                "false_refusal_rate": sum(1 for row in generated if row["refused"])
                / len(generated),
                "refusal_accuracy": sum(1 for row in unanswerable if row["refused"])
                / max(1, len(unanswerable)),
            }
        )
        if client is not None:
            cost = (
                client.prompt_tokens * INPUT_USD_PER_MTOK
                + client.completion_tokens * OUTPUT_USD_PER_MTOK
            ) / 1_000_000
            summary["tokens_prompt"] = client.prompt_tokens
            summary["tokens_completion"] = client.completion_tokens
            summary["cost_usd_total"] = round(cost, 5)
            summary["cost_usd_per_question"] = round(cost / len(rows), 5)

    print()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:24} {value:.3f}")
        else:
            print(f"{key:24} {value}")

    if args.json:
        payload = {
            "k": args.k,
            "generated": answerer is not None,
            "model": getattr(client, "model", "extractive") if answerer else None,
            "summary": summary,
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
