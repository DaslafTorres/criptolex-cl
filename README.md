# criptolex-cl

RAG assistant over **Chilean crypto and fintech regulation**, with citation-level evaluation.

Ask a question in Spanish, get an answer grounded in official sources — with the document, article and link it came from.

> ⚠️ Informational only. Not legal or tax advice.

## Status

🚧 In development — milestone M1 (ingestion) complete.

**Corpus:** 7 official documents → **543 citable chunks** (article-level for laws).

| Milestone | Scope | Status |
|---|---|---|
| M0 | Scaffold, tooling, CI | ✅ |
| M1 | Ingestion + article-aware chunking | ✅ |
| M2 | Hybrid retrieval (BM25 + vectors) + reranking | ⏳ |
| M3 | API `/ask` + chat UI with source cards | ⏳ |
| M4 | Evaluation harness + deploy | ⏳ |

## Sources

All sources are public and official. Third-party blogs are excluded on purpose.

| Key | Document | Issuer |
|---|---|---|
| `ley_21521` | Ley 21.521 (Fintech) | BCN |
| `ley_19913` | Ley 19.913 (AML / UAF) | BCN |
| `ncg_502_cmf` | NCG 502 | CMF |
| `circular_62_uaf` | Circular 62 | UAF |
| `sii_circular_43_2021` | Circular 43/2021 | SII |
| `sii_faq_cripto` | FAQ criptomonedas | SII |
| `sii_dj1964` | DJ 1964 | SII |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/DaslafTorres/criptolex-cl.git
cd criptolex-cl
uv sync

cp .env.example .env   # add your GROQ_API_KEY

uv run python -m criptolex.ingest   # download + extract official documents
uv run python -m criptolex.chunk    # build data/processed/chunks.jsonl
```

## Architecture

```
official sources (BCN · CMF · UAF · SII)
        │
        ▼
ingest.py ──► data/raw/*.txt
        │
        ▼
chunk.py  ──► article-aware chunks ──► data/processed/chunks.jsonl
        │
        ▼
retrieve.py (BM25 + vectors → rerank)
        │
        ▼
generate.py (Groq, temperature 0, citations required)
        │
        ▼
api.py (FastAPI) ──► /ask  { answer, sources[] }
```

## Why article-aware chunking

Legal text is already segmented by its authors. Splitting on `Artículo N°`
instead of fixed-size windows keeps each chunk self-contained and citable, which
is what makes citation-level evaluation possible.

## Roadmap

- [x] Official source registry
- [x] Ingestion and text extraction
- [x] Article-aware chunking with tests
- [ ] Hybrid retrieval + reranking
- [ ] Citation-enforcing generation
- [ ] Golden evaluation set (25 questions)
- [ ] Metrics: citation accuracy, faithfulness, refusal rate, latency, cost
- [ ] Deploy and demo GIF

## License

MIT © 2026 Daslaf Torres
