# criptolex-cl

RAG assistant over **Chilean crypto and fintech regulation**, with citation-level evaluation.

Ask a question in Spanish, get an answer grounded in official sources — with the document, article and link it came from.

> ⚠️ Informational only. Not legal or tax advice.

## Status

🚧 In development — M1 (ingestion), M2 (retrieval) and M3 (API + UI) complete.

**Corpus:** 7 official documents (831,549 chars) → **628 citable chunks** (article-level for laws).

| Milestone | Scope | Status |
|---|---|---|
| M0 | Scaffold, tooling, CI | ✅ |
| M1 | Ingestion + article-aware chunking | ✅ |
| M2 | Hybrid retrieval (BM25 + vectors) + reranking | ✅ |
| M3 | API `/ask` + chat UI with source cards | ✅ |
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

# ask the retriever directly (no API key needed)
uv run python -m criptolex.retrieve "¿pago impuestos si vendo bitcoin?"
uv run python -m criptolex.retrieve "plazo para informar incidentes" --compare

# API + chat UI
uv run uvicorn criptolex.api:app --reload   # http://127.0.0.1:8000
```

No Groq key yet? Run the API with `LLM_PROVIDER=extractive` to get a no-LLM
demo that quotes the best retrieved passage instead of writing an answer.

## API

```http
POST /ask
{ "question": "¿Las ganancias por vender criptomonedas se declaran en Chile?", "k": 5 }
```

```json
{
  "answer": "Sí. Toda persona domiciliada o residente en Chile debe declarar ... [1]",
  "sources": [
    {
      "n": 1,
      "source_key": "sii_faq_cripto",
      "title": "Preguntas frecuentes — Activos digitales y criptomonedas",
      "issuer": "SII",
      "year": 2025,
      "article": null,
      "url": "https://www.sii.cl/...",
      "snippet": "Pregunta: ¿Las ganancias obtenidas ..."
    }
  ],
  "grounded": true,
  "refused": false,
  "model": "llama-3.3-70b-versatile",
  "latency_ms": 2140
}
```

The citation contract is enforced server-side, not requested politely:

- The model sees numbered extracts and must cite them inline as `[n]`.
- Any citation pointing outside the retrieved set is **removed** before the
  response leaves the API, and `sources` contains only the chunks the answer
  actually cited.
- If the model says the extracts are insufficient, the API returns a fixed
  refusal with `refused: true` and no sources.
- `grounded` is `false` when the answer carries no valid citation, which is how
  the evaluation harness will detect unsupported prose.

`GET /health` reports the provider and whether an LLM key is configured.

## UI

A single static page (`web/`) served by the same FastAPI process: dark,
minimal, English chrome with an EN/ES toggle, and answers stay in Spanish because
the corpus is. Citations in the answer are clickable and scroll to the matching
source card, which links to the official PDF or page.

No build step, no frontend framework — the whole client is one HTML file, one
stylesheet and ~180 lines of JavaScript.

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

Legal text is already segmented by its authors. Splitting on `Artículo N`
instead of fixed-size windows keeps each chunk self-contained and citable, which
is what makes citation-level evaluation possible.

Three details make this harder than a regex, and each one is handled and tested:

1. **Laws that amend other laws** quote the articles of those laws. Ley 21.521
   would appear to have 254 articles; it has 46. Top-level headings are accepted
   only when they continue the sequence (`Artículo 45` → `Artículo 46`), so
   nested headings stay inside the article that quotes them.
2. **Transitory provisions** restart their numbering (`Artículo primero…`, or
   `Artículo 1º…` in older laws). They are detected as a separate section and
   labelled `Transitorio primero`, `Transitorio segundo`, …
3. **PDF extraction noise** — page numbers, running page headers and the
   Constitutional Court certification — is dropped, and no chunk exceeds 2,200
   characters.

Result: 46 articles + 9 transitory provisions for Ley 21.521, and 41 articles +
2 `bis` + 7 transitory provisions for Ley 19.913, with zero duplicate chunks.

## How retrieval works

```
question
   ├── BM25 (Spanish stemming)        → 20 candidates
   └── MiniLM embeddings (Chroma)     → 20 candidates
              └── Reciprocal Rank Fusion → 20 candidates
                       └── cross-encoder rerank → fused again → top 5
```

Three decisions came from measurement, not intuition (the probe script is in
`eval/probe_retrieval.py` and runs with `uv run python eval/probe_retrieval.py`):

1. **Only the body is indexed.** Every chunk of a document shares the same
   citation header (`NCG 502 — Registro y obligaciones…`), so indexing it made
   all 259 chunks of that norm look alike. Keeping the header as metadata raised
   `hit@5` on the probe set from **0.55 to 0.91**.
2. **The reranker is fast, not big.** `ms-marco-MiniLM-L-12-v2` takes ~8 s per
   query on CPU versus ~0.4 s for `ms-marco-TinyBERT-L-2-v2`, with the same
   recall@5. The model is configurable through `RERANK_MODEL`.
3. **Rerank scores are fused, not obeyed.** A cross-encoder trained on English
   web passages is a strong signal on Spanish legal text but not an authority;
   fusing its ranking with BM25 and dense rankings prevented relevant lexical
   matches from being pushed out of the context window (hit@5 **0.82 → 1.00**).

Everything runs locally — embeddings (ONNX MiniLM) and reranker included — so
retrieval needs no API key and costs nothing per query.

**Probe results** (11 questions, each with a hand-checked gold chunk, top-5):

| Metric | Value |
|---|---|
| hit@5 | 1.00 |
| MRR@5 | 0.52 |
| Latency | ~400 ms/query on CPU |

MRR is modest because the first result is sometimes a section heading that
matches the wording but not the answer. That is what M4 measures properly with
citation-level evaluation over generated answers.

## Roadmap

- [x] Official source registry
- [x] Ingestion and text extraction
- [x] Article-aware chunking with tests
- [x] Hybrid retrieval + reranking (hit@5 1.00 on the probe set)
- [x] Citation-enforcing generation
- [x] API `/ask` + chat UI
- [ ] Golden evaluation set (25 questions)
- [ ] Embedding and chunk-size benchmark (English vs multilingual MiniLM)
- [ ] Metrics: citation accuracy, faithfulness, refusal rate, latency, cost
- [ ] Deploy and demo GIF

## License

MIT © 2026 Daslaf Torres
