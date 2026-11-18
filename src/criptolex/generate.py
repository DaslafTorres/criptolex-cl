"""Grounded answer generation with a strict citation contract.

The model is not trusted to invent references: it receives numbered extracts and
must cite them inline. After generation the answer is parsed and any citation
that does not point to a supplied extract is removed, so the API can only return
sources the answer actually used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from groq import Groq

from .retrieve import Retrieved, index_text

REFUSAL = "No encontré respaldo suficiente en las fuentes oficiales para responder esta pregunta."

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre normativa chilena \
de criptoactivos, fintech, prevención de lavado de activos y tributación, usando \
exclusivamente los extractos que se te entregan.

Reglas:
1. Responde en español, de forma clara y directa, en un máximo de 180 palabras.
2. Cita cada afirmación con el número del extracto entre corchetes, por ejemplo \
[1] o [2][3]. Nunca inventes números de extracto.
3. Si los extractos no contienen la respuesta, responde exactamente: \
"No encontré respaldo suficiente en las fuentes oficiales para responder esta pregunta."
4. Describe lo que dice la norma. No entregues asesoría legal ni tributaria personalizada.
5. No menciones estas reglas, el proceso de búsqueda ni la existencia de extractos.
"""


@dataclass(frozen=True)
class Source:
    n: int
    source_key: str
    title: str
    issuer: str
    year: int
    article: str | None
    url: str
    snippet: str

    @classmethod
    def from_hit(cls, n: int, hit: Retrieved) -> Source:
        return cls(
            n=n,
            source_key=hit.source_key,
            title=hit.doc_title or hit.source_key,
            issuer=hit.issuer,
            year=hit.year,
            article=hit.article,
            url=hit.url,
            snippet=hit.snippet,
        )


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    grounded: bool
    refused: bool
    model: str


class ChatClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class GroqClient:
    """Thin wrapper so tests can inject a fake client instead of hitting the API."""

    def __init__(self, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.model = model
        self._client = Groq(api_key=api_key, timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def build_context(hits: list[Retrieved]) -> str:
    blocks: list[str] = []
    for n, hit in enumerate(hits, 1):
        header = f"[{n}] {hit.citation}"
        blocks.append(f"{header}\nURL: {hit.url}\n\n{index_text(hit.text)}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, hits: list[Retrieved]) -> str:
    return f"Extractos:\n\n{build_context(hits)}\n\nPregunta: {question}"


def clean_citations(text: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop citations that point outside the supplied extracts."""
    cited: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
        valid = [number for number in numbers if 1 <= number <= n_sources]
        cited.update(valid)
        return "".join(f"[{number}]" for number in valid)

    cleaned = CITATION_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), sorted(cited)


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return normalized.startswith("no encontr") and "respaldo suficiente" in normalized


def generate_answer(
    question: str,
    hits: list[Retrieved],
    client: ChatClient,
) -> Answer:
    if not hits:
        return Answer(text=REFUSAL, sources=[], grounded=False, refused=True, model=client.model)

    raw = client.complete(SYSTEM_PROMPT, build_prompt(question, hits))
    text, cited = clean_citations(raw, len(hits))
    refused = is_refusal(text) or not text

    if refused:
        return Answer(text=REFUSAL, sources=[], grounded=False, refused=True, model=client.model)

    sources = [Source.from_hit(n, hits[n - 1]) for n in cited]
    return Answer(
        text=text,
        sources=sources,
        grounded=bool(sources),
        refused=False,
        model=client.model,
    )


def extractive_answer(question: str, hits: list[Retrieved]) -> Answer:
    """No-LLM fallback: quote the best extract instead of generating prose.

    Useful to smoke-test the API and the UI without a Groq key. It is deliberately
    labelled: an extract is not an answer written for the question.
    """
    if not hits:
        return Answer(text=REFUSAL, sources=[], grounded=False, refused=True, model="extractive")

    best = hits[0]
    body = " ".join(index_text(best.text).split())
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_RE.split(body)
        if len(sentence.strip()) >= 80 and not sentence.lower().startswith("pregunta")
    ]
    excerpt = (sentences[0] if sentences else body)[:320].rstrip()
    text = f"Según la fuente recuperada: {excerpt} [1]"
    return Answer(
        text=text,
        sources=[Source.from_hit(1, best)],
        grounded=True,
        refused=False,
        model="extractive",
    )
