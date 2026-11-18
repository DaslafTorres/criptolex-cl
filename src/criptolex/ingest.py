"""Download official sources and extract plain text.

Usage:
    uv run python -m criptolex.ingest            # all sources
    uv run python -m criptolex.ingest sii_faq_cripto ley_21521
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import get_settings
from .sources import BY_KEY, SOURCES, Source

USER_AGENT = "criptolex-cl/0.1 (+https://github.com/DaslafTorres/criptolex-cl)"
RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with retries for timeouts, connection errors, 429 and 5xx.

    Official sites are occasionally slow or briefly unavailable, and a Docker
    build that downloads the corpus should not fail on a single hiccup. Client
    errors (404 and friends) are raised immediately: retrying them achieves
    nothing.
    """
    last_error: httpx.HTTPError | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status < 500 and status != 429:
                raise
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc

        if attempt < RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    assert last_error is not None
    raise last_error


def download(source: Source, raw_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download a source once and cache it under data/raw/."""
    suffix = {"pdf": ".pdf", "bcn_txt": ".source.txt", "bcn_json": ".json"}.get(
        source.kind, ".html"
    )
    dest = raw_dir / f"{source.key}{suffix}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if client is None:
        with httpx.Client(
            follow_redirects=True,
            timeout=90.0,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        ) as own_client:
            return download(source, raw_dir, own_client)

    response = _get(client, source.api_url or source.url)
    dest.write_bytes(response.content)
    return dest


def extract_html(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            html = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - unreachable, latin-1 never fails
        html = raw.decode("utf-8", errors="ignore")

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not text or len(text) < 500:
        raise ValueError(f"Could not extract meaningful text from {path.name}")
    return text


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page)
    if len(text) < 500:
        raise ValueError(f"Could not extract meaningful text from {path.name}")
    return text


def extract_bcn_txt(path: Path) -> str:
    """LeyChile's official TXT export is already plain text."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        text = raw.decode("utf-8", errors="ignore")
    if len(text) < 500:
        raise ValueError(f"Could not extract meaningful text from {path.name}")
    return text


def extract_bcn_json(path: Path) -> str:
    """LeyChile returns JSON with a list of HTML fragments, one per section."""
    data = json.loads(path.read_text(encoding="utf-8"))
    fragments = [item.get("t", "") for item in data.get("html", [])]
    if not fragments:
        raise ValueError(f"No HTML fragments found in {path.name}")
    soup = BeautifulSoup("\n".join(fragments), "html.parser")
    text = soup.get_text("\n")
    if len(text) < 500:
        raise ValueError(f"Could not extract meaningful text from {path.name}")
    return text


def crawl_sii_faq(source: Source, index_path: Path, client: httpx.Client) -> str:
    """The SII FAQ index links to one page per question; fetch them all."""
    index = BeautifulSoup(index_path.read_bytes(), "html.parser")
    anchors = index.select("div#listado-preguntas-por-tema a[href$='.htm']")
    if not anchors:
        raise ValueError("No FAQ question links found")

    parts: list[str] = []
    for anchor in anchors:
        url = urljoin(source.url, anchor.get("href", ""))
        question = anchor.get_text(" ", strip=True)
        response = _get(client, url)

        page = BeautifulSoup(response.content, "html.parser")
        for tag in page(["script", "style"]):
            tag.decompose()

        answer_node = page.select_one("#div-respuesta")
        answer = answer_node.get_text("\n", strip=True) if answer_node else ""
        if not answer:
            continue
        parts.append(f"Pregunta: {question}\n\n{answer}")

    if not parts:
        raise ValueError("FAQ pages contained no answers")
    return "\n\n".join(parts)


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest(keys: list[str] | None = None) -> list[Path]:
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    selected = SOURCES if not keys else [BY_KEY[key] for key in keys]
    written: list[Path] = []

    with httpx.Client(
        follow_redirects=True,
        timeout=90.0,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    ) as client:
        for source in selected:
            local = download(source, settings.raw_dir, client)
            if source.kind == "pdf":
                text = extract_pdf(local)
            elif source.kind == "bcn_txt":
                text = extract_bcn_txt(local)
            elif source.kind == "bcn_json":
                text = extract_bcn_json(local)
            elif source.kind == "sii_faq":
                text = crawl_sii_faq(source, local, client)
            else:
                text = extract_html(local)
            text = clean(text)

            out = settings.raw_dir / f"{source.key}.txt"
            out.write_text(text, encoding="utf-8")
            written.append(out)
            print(f"[ok] {source.key:<22} {len(text):>8,} chars -> {out.name}")

    return written


def main(argv: list[str] | None = None) -> int:
    keys = (argv if argv is not None else sys.argv[1:]) or None
    unknown = [key for key in (keys or []) if key not in BY_KEY]
    if unknown:
        print(f"Unknown source key(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(BY_KEY)}", file=sys.stderr)
        return 2
    ingest(keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
