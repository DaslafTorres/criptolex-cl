"""Article-aware chunking for legal and regulatory documents.

Legal texts are split on article boundaries so that every chunk can be cited
as "document + article". Non-legal documents fall back to paragraph windows.

Laws that amend other laws quote the articles of those laws, so a naive regex
over-counts articles. Top-level headings are detected monotonically: a heading is
only accepted when it continues the sequence (``Artículo 46`` -> ``Artículo 47``).
Transitory provisions use their own numbering and are labelled separately.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from re import Match

from .config import get_settings
from .sources import SOURCES, Source

_SEPARATOR = r"[ \t]*(?:[.\-–—:]|$)[ \t]*[-–—]?[ \t]*"

ARTICLE_HEAD_RE = re.compile(
    r"(?im)^[ \t]*art[ií]culo[ \t]+(?P<num>\d+)(?P<bis>[ \t]+bis)?"
    r"[ \t]*[°º]?" + _SEPARATOR
)
TRANSITORIOS_RE = re.compile(
    r"(?im)^[ \t]*(?:disposiciones|art[ií]culos|normas)[ \t]+transitori[oa]s?\.?[ \t]*$"
)
TRANSITORIO_HEAD_RE = re.compile(
    r"(?im)^[ \t]*art[ií]culo[ \t]+(?:(?P<num>\d+)[ \t]*[°º]?|(?P<word>[a-záéíóúñ]+))"
    + _SEPARATOR
)
LAW_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:ley|decreto|resoluci[oó]n)\s+(?:n[uú]m|[nN][°º]|exenta|suprema|\d)"
)
HEADER_NOISE_RE = re.compile(
    r"^(?:biblioteca del congreso nacional|[-–—=]{5,}"
    r"|(?:tipo norma|publicaci[oó]n|promulgaci[oó]n|organismo|t[ií]tulo|versi[oó]n"
    r"|inicio vigencia|id norma|url)\s*:)",
    re.IGNORECASE,
)
TAIL_PATTERNS = (
    re.compile(r"(?im)^[ \t]*habi[eé]ndo(?:se)? cumplido con lo establecido"),
    re.compile(r"(?im)^[ \t]*tribunal constitucional[ \t]*$"),
)
SENTENCE_RE = re.compile(r"(?<=[.;:])\s+")
PAGE_NUMBER_RE = re.compile(r"^-?\s*\d{1,4}\s*-?$")

MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 2200
FALLBACK_CHARS = 1200
MIN_ARTICLES = 3
GAP_MAX = 2
TAIL_WINDOW_CHARS = 6000

ORDINALS = {
    1: "primero",
    2: "segundo",
    3: "tercero",
    4: "cuarto",
    5: "quinto",
    6: "sexto",
    7: "séptimo",
    8: "octavo",
    9: "noveno",
    10: "décimo",
}


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


def strip_header_metadata(text: str) -> str:
    """Drop the BCN catalogue header (``Tipo Norma`` ... ``URL``) at the top."""
    heading = LAW_HEADING_RE.search(text[:4000])
    if heading:
        return text[heading.start() :].strip()

    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or HEADER_NOISE_RE.match(line):
            index += 1
            continue
        break
    return "\n".join(lines[index:]).strip()


def strip_tail(text: str) -> str:
    """Drop the signature and Constitutional Court certification at the end."""
    window_start = max(0, len(text) - TAIL_WINDOW_CHARS)
    tail = text[window_start:]
    cut: int | None = None
    for pattern in TAIL_PATTERNS:
        match = pattern.search(tail)
        if match and (cut is None or match.start() < cut):
            cut = match.start()
    if cut is None:
        return text
    return text[: window_start + cut].rstrip()


def _numbered_articles(text: str) -> list[tuple[str, str]]:
    """Accept only headings that continue the top-level article sequence."""
    accepted: list[Match[str]] = []
    last: int | None = None

    for match in ARTICLE_HEAD_RE.finditer(text):
        number = int(match.group("num"))
        is_bis = bool(match.group("bis"))
        if last is None:
            accepted.append(match)
            last = number
            continue

        keep = number == last if is_bis else last < number <= last + GAP_MAX
        if keep:
            accepted.append(match)
            last = number

    if len(accepted) < MIN_ARTICLES:
        return []

    sections: list[tuple[str, str]] = []
    preamble = text[: accepted[0].start()].strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        sections.append(("Preámbulo", preamble))

    for index, match in enumerate(accepted):
        start = match.end()
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        body = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        label = f"Artículo {match.group('num')}"
        if match.group("bis"):
            label += " bis"
        sections.append((label, body))
    return sections


def _transitorios(text: str) -> list[tuple[str, str]]:
    """Split the transitory section, which restarts its own numbering."""
    matches = list(TRANSITORIO_HEAD_RE.finditer(text))
    sections: list[tuple[str, str]] = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        if match.group("num"):
            ordinal = ORDINALS.get(int(match.group("num")), match.group("num"))
        else:
            ordinal = match.group("word").lower()
        sections.append((f"Transitorio {ordinal}", body))
    return sections


def split_articles(text: str) -> list[tuple[str, str]]:
    """Split a legal text into (article_label, body) pairs."""
    marker = TRANSITORIOS_RE.search(text)
    main_text = text[: marker.start()] if marker else text

    sections = _numbered_articles(main_text)
    if not sections:
        return []
    if marker is not None:
        sections.extend(_transitorios(text[marker.end() :]))
    return sections


def _is_page_number(text: str) -> bool:
    return bool(PAGE_NUMBER_RE.fullmatch(text.strip()))


def _drop_boilerplate(paragraphs: list[str], max_chars: int = 240) -> list[str]:
    """Remove short paragraphs repeated on every page (running headers/footers)."""
    counts = Counter(p for p in paragraphs if len(p) <= max_chars)
    return [
        p for p in paragraphs if not (len(p) <= max_chars and counts[p] >= 3)
    ]


def _looks_like_heading(text: str) -> bool:
    if not text or len(text) > 250:
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    uppercase = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase >= 0.7


def _merge_headings(paragraphs: list[str]) -> list[str]:
    """Glue section headings to the paragraph they introduce."""
    merged: list[str] = []
    pending: list[str] = []

    for paragraph in paragraphs:
        if pending:
            paragraph = "\n\n".join([*pending, paragraph])
            pending = []
        if _looks_like_heading(paragraph):
            pending.append(paragraph)
            continue
        merged.append(paragraph)

    if pending:
        if merged:
            merged[-1] = "\n\n".join([merged[-1], *pending])
        else:
            merged.extend(pending)
    return merged


def sliding_windows(text: str, size: int = FALLBACK_CHARS) -> list[tuple[str, str]]:
    """Split non-legal text into paragraph windows without cutting sentences."""
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip() and not _is_page_number(p)
    ]
    paragraphs = _drop_boilerplate(paragraphs)
    paragraphs = _merge_headings(paragraphs)

    windows: list[str] = []
    current: list[str] = []
    length = 0

    for paragraph in paragraphs:
        if current and length + len(paragraph) > size:
            windows.append("\n\n".join(current))
            current = []
            length = 0
        current.append(paragraph)
        length += len(paragraph)

    if current:
        windows.append("\n\n".join(current))

    # Headings that end up alone are merged forward so they never ship as a chunk.
    merged: list[str] = []
    for body in windows:
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{body}"
        else:
            merged.append(body)
    if len(merged) > 1 and len(merged[-1]) < MIN_CHUNK_CHARS:
        merged[-2] = f"{merged[-2]}\n\n{merged[-1]}"
        merged.pop()
    return [("", body) for body in merged]


def _pack(units: list[str], max_chars: int, glue: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    length = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        extra = len(unit) + (len(glue) if current else 0)
        if current and length + extra > max_chars:
            parts.append(glue.join(current))
            current = [unit]
            length = len(unit)
        else:
            current.append(unit)
            length += extra

    if current:
        parts.append(glue.join(current))
    return parts


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort split on word boundaries so no chunk exceeds the budget."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts or [text]


def _split_unit(text: str, max_chars: int) -> list[str]:
    """Split a single oversized unit, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) > 1:
        parts: list[str] = []
        for part in _pack(sentences, max_chars, " "):
            if len(part) > max_chars:
                parts.extend(_hard_split(part, max_chars))
            else:
                parts.append(part)
        return parts
    return _hard_split(text, max_chars)


def _split_block(text: str, max_chars: int) -> list[str]:
    """Split one oversized block on lines, then sentences, then words."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if not _is_page_number(line)]
    if not lines:
        return []

    pieces: list[str] = []
    for group in _pack(lines, max_chars, "\n"):
        pieces.extend(_split_unit(group, max_chars))
    return pieces


def split_long(
    article: str, body: str, max_chars: int = MAX_CHUNK_CHARS
) -> list[tuple[str, str]]:
    """Split an oversized article on paragraphs, then lines, then sentences."""
    if len(body) <= max_chars:
        return [(article, body)]

    units = [p.strip() for p in re.split(r"\n[ \t]*\n", body) if p.strip()]
    pieces: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if len(current) >= max_chars // 2:
            pieces.append(current)
            current = unit
        else:
            # A short prefix (heading, page number) must travel with the block
            # instead of becoming a chunk of its own.
            current = candidate

        if len(current) > max_chars:
            pieces.extend(_split_block(current, max_chars))
            current = ""

    if current:
        pieces.append(current)

    if len(pieces) > 1 and len(pieces[-1]) < MIN_CHUNK_CHARS:
        merged = f"{pieces[-2]}\n\n{pieces[-1]}"
        if len(merged) <= max_chars:
            pieces = [*pieces[:-2], merged]

    return [(article, piece) for piece in pieces]


def chunk_source(source: Source, text: str) -> list[Chunk]:
    text = normalize(text)
    if source.legal:
        text = strip_header_metadata(text)
    text = strip_tail(text)

    header = f"{source.title} ({source.issuer}, {source.year})"

    sections = split_articles(text) if source.legal else []
    if not sections:
        sections = sliding_windows(text)

    expanded: list[tuple[str, str]] = []
    for article, body in sections:
        section_header = f"{header} — {article}" if article else header
        budget = MAX_CHUNK_CHARS - len(section_header) - 2
        expanded.extend(split_long(article, body, max_chars=budget))

    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for article, body in expanded:
        key = (article, body.strip())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        unique.append((article, body))

    chunks: list[Chunk] = []
    for index, (article, body) in enumerate(unique):
        body = body.strip()
        if not body or not any(char.isalpha() for char in body):
            continue
        section_header = f"{header} — {article}" if article else header
        chunks.append(
            Chunk(
                id=f"{source.key}:{index:04d}",
                text=f"{section_header}\n\n{body}",
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
