"""Article-aware chunking for legal and regulatory documents.

Legal texts are split on article boundaries so that every chunk can be cited
as "document + article". Non-legal documents fall back to paragraph windows.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import get_settings
from .sources import SOURCES, Source

ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*(art[íi]culo|art\.)[ \t]*(\d+)[ \t]*[°º]?[ \t]*[.\-–—:]?[ \t]*"
)
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 2200
FALLBACK_CHARS = 1200
MIN_ARTICLES = 3


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_key: str
    doc_title: str
    issuer: str
    url: str
    year: int
    article: str | None


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_articles(text: str) -> list[tuple[str, str]]:
    """Split a legal text into (article_label, body) pairs."""
    matches = list(ARTICLE_RE.finditer(text))
    if len(matches) < MIN_ARTICLES:
        return []

    articles: list[tuple[str, str]] = []
    # Preamble before the first article is usually the law header; keep it
    # out of the article chunks but preserve it as a context chunk.
    preamble = text[: matches[0].start()].strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        articles.append(("Preámbulo", preamble))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        label = f"Artículo {match.group(2)}°"
        if len(body) < MIN_CHUNK_CHARS:
            continue
        articles.append((label, body))
    return articles


def sliding_windows(text: str, size: int = FALLBACK_CHARS) -> list[tuple[str, str]]:
    """Split non-legal text into paragraph windows without cutting sentences."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    windows: list[tuple[str, str]] = []
    current: list[str] = []
    length = 0

    for paragraph in paragraphs:
        if current and length + len(paragraph) > size:
            windows.append(("", "\n\n".join(current)))
            current = [current[-1]] if len(current[-1]) < size // 3 else []
            length = sum(len(item) for item in current)
        current.append(paragraph)
        length += len(paragraph)

    if current:
        windows.append(("", "\n\n".join(current)))
    return windows


def split_long(article: str, body: str) -> list[tuple[str, str]]:
    """Further split an oversized article on paragraph boundaries."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [(article, body)]

    parts: list[tuple[str, str]] = []
    current: list[str] = []
    length = 0

    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and length + len(paragraph) > MAX_CHUNK_CHARS:
            parts.append((article, "\n\n".join(current)))
            current = []
            length = 0
        current.append(paragraph)
        length += len(paragraph)

    if current:
        parts.append((article, "\n\n".join(current)))
    return parts


def chunk_source(source: Source, text: str) -> list[Chunk]:
    text = normalize(text)
    sections = split_articles(text) if source.legal else []
    if not sections:
        sections = sliding_windows(text)

    expanded: list[tuple[str, str]] = []
    for article, body in sections:
        expanded.extend(split_long(article, body))

    chunks: list[Chunk] = []
    for index, (article, body) in enumerate(expanded):
        body = body.strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        header = f"{source.title} ({source.issuer}, {source.year})"
        if article:
            header = f"{header} — {article}"
        chunks.append(
            Chunk(
                id=f"{source.key}:{index:04d}",
                text=f"{header}\n\n{body}",
                source_key=source.key,
                doc_title=source.title,
                issuer=source.issuer,
                url=source.url,
                year=source.year,
                article=article or None,
            )
        )
    return chunks


def build_chunks() -> list[Chunk]:
    settings = get_settings()
    all_chunks: list[Chunk] = []

    for source in SOURCES:
        text_path = settings.raw_dir / f"{source.key}.txt"
        if not text_path.exists():
            print(f"[skip] {source.key}: run `uv run python -m criptolex.ingest` first")
            continue
        chunks = chunk_source(source, text_path.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)
        print(f"[ok] {source.key:<22} {len(chunks):>4} chunks")

    return all_chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main() -> int:
    settings = get_settings()
    chunks = build_chunks()
    if not chunks:
        print("No chunks produced. Did the ingestion step run?")
        return 1
    output = settings.processed_dir / "chunks.jsonl"
    write_chunks(chunks, output)
    print(f"\n{len(chunks):,} chunks -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
