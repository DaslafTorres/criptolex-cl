# criptolex-cl

[![CI](https://github.com/DaslafTorres/criptolex-cl/actions/workflows/ci.yml/badge.svg)](https://github.com/DaslafTorres/criptolex-cl/actions/workflows/ci.yml)
[![Docker](https://github.com/DaslafTorres/criptolex-cl/actions/workflows/docker.yml/badge.svg)](https://github.com/DaslafTorres/criptolex-cl/actions/workflows/docker.yml)

RAG assistant over **Chilean crypto and fintech regulation**, with citation-level evaluation.

Ask a question in Spanish, get an answer grounded in official sources — with the document, article and link it came from.

> ⚠️ Informational only. Not legal or tax advice.

## Status

🚧 In development — ingestion, retrieval, API and the evaluation harness are done; deployment is next.

**Corpus:** 7 official documents (831,549 chars) → **630 citable chunks** (article-level for laws).

| Milestone | Scope | Status |
|---|---|---|
| M0 | Scaffold, tooling, CI | ✅ |
| M1 | Ingestion + article-aware chunking | ✅ |
| M2 | Hybrid retrieval (BM25 + vectors) + reranking | ✅ |
| M3 | API `/ask` + chat UI with source cards | ✅ |
| M4 | Evaluation harness + generation metrics | ✅ (deploy pending) |

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
  "model": "qwen/qwen3.8-27b",
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

## Deploy

The image is self-contained: corpus, vector index, embedding model and reranker
are baked in at build time, so a cold container answers the first question
without downloading anything.

```bash
docker build -t criptolex-cl .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=gsk_... criptolex-cl
# no key: -e LLM_PROVIDER=extractive
```

- `PORT` is respected (default `7860`, the Hugging Face Spaces convention).
- The Groq key is read at runtime and is never baked into the image.
- Official sites are occasionally slow, so ingestion retries transient failures
  instead of failing the build on one hiccup.

On every push to `main`, GitHub Actions builds the image and pushes it to
`ghcr.io/daslaftorres/criptolex-cl:latest`, which also proves the Dockerfile
still builds (the workflow is green). The public demo is a Hugging Face
**Docker** Space that only pulls that image — the Space itself is two files, in
[`deploy/huggingface/`](deploy/huggingface/), with the setup steps.

The Space is not live yet: it needs a public container package, an
`GROQ_API_KEY` secret and one deploy to verify end to end before a URL is linked
here.

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

Non-legal sources use paragraph windows, except Q&A pages (the SII FAQ): there a
window split left each "Pregunta:" dangling at the end of the *previous* chunk,
detached from its answer. Splitting on question boundaries instead fixed the
"mayor valor" query from rank ≥ 6 to rank 1, because the question line is the
strongest lexical signal in a FAQ.

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

## Evaluation

Two sets live in `eval/`:

- `probe_retrieval.py` — 11 questions used to make the retrieval decisions above.
  Small and fast, meant to catch regressions.
- `golden.jsonl` — 25 questions for the milestone: 20 answerable covering all
  seven sources, plus 5 out-of-scope questions for refusal testing.

The method matters more than the number:

- Questions were written **from the source text**, not from retrieval output, so
  the set does not inherit the retriever's blind spots.
- Every expected fact must appear verbatim (accent-insensitive) in the gold
  chunk. `run_eval.py` validates this before scoring and refuses to run if the
  corpus and the golden set have drifted apart.
- Retrieval is scored at **chunk level**, not "looks relevant".
- The out-of-scope questions are plausible — VAT on streaming, US SEC rules,
  municipal permits — so refusing them is not trivially easy.

```bash
uv run python eval/run_eval.py                 # retrieval only, no API key
uv run python eval/run_eval.py --generate      # + citations, facts, refusals
```

**Results** (20 answerable, 5 out of scope; reproducible on CPU):

| Metric | Value |
|---|---|
| retrieval hit@5 | **0.90** (18/20) |
| retrieval MRR@5 | 0.63 (0.61–0.63 across runs) |
| retrieval latency p50 / p95 | 377 ms / 393 ms |
| citation recall | **0.85** |
| citation precision | 0.40 (strict definition, see below) |
| fact recall | 0.70 |
| grounded rate | 0.95 |
| abstention accuracy | 0.80 (4/5 out-of-scope questions refused) |
| answer latency p50 / p95 | 14.5 s / 21.3 s (free-tier throttle, see below) |
| cost | **$0.051** for 25 questions, ~$0.002 each |

Generation runs on `qwen/qwen3.8-27b` via Groq and used 46.1k prompt / 3.5k
completion tokens in total. Latency and cost are measured end to end, including
retrieval, reranking and the HTTP call.

`hit@5` is stable across runs, but MRR moves by about ±0.02: Chroma's approximate
search and the cross-encoder both order near-ties slightly differently between
processes, which is worth knowing before reading too much into a single run.

The two retrieval misses are honest ranking errors, not missing evidence:

1. *"¿Qué entidades están obligadas a informar operaciones sospechosas?"* — the
   list of obliged entities (Ley 19.913, artículo 3) never enters the top 5;
   Circular 62's procedural section on *reporting* outranks it, because the
   question's wording matches the procedure more strongly than the list.
2. *"¿Quiénes deben presentar la declaración jurada 1964?"* — the gold chunk lands
   at rank 7, behind an SII FAQ paragraph about form **1891**, a different
   declaration. The question shares almost every word with the wrong answer and
   only the form number differs, which is exactly the case where "looks
   relevant" is not good enough.

Both golds sit inside the top 8, so the next things to test are a metadata boost
when a question names a specific form, and query expansion for list questions.

**Citation precision counts a citation as correct only when it points at one of the
1–2 gold chunks for that question.** A typical answer cites 2–4 fragments, and
several of the "incorrect" ones are articles that genuinely support the sentence
but are not in the gold set, so this number understates citation quality. Recall
is the one to watch; document-level precision is on the roadmap.

The 14 s p50 is not model latency. Groq's free tier allows 1,000 output tokens
per minute, so the harness waits between calls to stay under the limit. With a
paid key the same call returns in 0.4–1.2 s.

Two failures are worth naming instead of hiding:

- one false refusal (`roe_definicion`): the gold chunk was retrieved at rank 2 and
  the model still declined, so the prompt is too conservative, not the retrieval;
- one out-of-scope question answered (`fuera_sociedad`): asked how to incorporate
  a closed corporation, it explained the definition instead of refusing.

Both are prompt-level fixes queued for the next iteration.

Reproducing the generation numbers needs a Groq key in `GROQ_API_KEY`. Without
one, `LLM_PROVIDER=extractive` keeps the demo honest: it quotes a retrieved
passage and never pretends to reason. Raw per-question output, including every
citation and refusal, is in [`eval/results.json`](eval/results.json).

## Roadmap

- [x] Official source registry
- [x] Ingestion and text extraction
- [x] Article-aware chunking with tests
- [x] Hybrid retrieval + reranking (hit@5 1.00 on the probe set)
- [x] Citation-enforcing generation
- [x] API `/ask` + chat UI
- [x] Golden evaluation set (25 questions) with fact validation
- [x] Retrieval metrics: hit@5 0.90, MRR@5 0.63
- [x] Generation metrics: citation recall 0.85, fact recall 0.70, abstention 0.80, $0.002/question
- [ ] Embedding and chunk-size benchmark (English vs multilingual MiniLM)
- [ ] Deploy and demo GIF

## License

MIT © 2026 Daslaf Torres
