from criptolex.chunk import Chunk, chunk_source, normalize, split_articles
from criptolex.sources import Source

LEGAL = Source(
    key="demo_ley",
    title="Ley Demo",
    issuer="Demo",
    url="https://example.com",
    kind="html",
    year=2024,
    legal=True,
)


def test_split_articles_finds_boundaries() -> None:
    text = (
        "Ley Demo\n\n"
        "Artículo 1°.- Esta ley regula los activos virtuales en el país y define "
        "sus características principales para efectos de su aplicación.\n\n"
        "Artículo 2°.- Para los efectos de esta ley se entenderá por criptoactivo "
        "toda representación digital de valor que pueda transferirse.\n\n"
        "Artículo 3°.- Los prestadores de servicios deberán inscribirse en el "
        "registro que al efecto lleve la Comisión.\n"
    )
    articles = split_articles(text)
    labels = [label for label, _ in articles]
    assert "Artículo 1°" in labels
    assert "Artículo 2°" in labels
    assert "Artículo 3°" in labels


def test_split_articles_requires_enough_markers() -> None:
    text = "Artículo 1°.- Texto único sin más artículos en el documento completo."
    assert split_articles(text) == []


def test_chunk_source_uses_article_metadata() -> None:
    text = "\n\n".join(
        f"Artículo {i}°.- " + ("Contenido normativo de prueba. " * 8) for i in range(1, 8)
    )
    chunks = chunk_source(LEGAL, text)
    assert len(chunks) >= 7
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.article is not None for chunk in chunks)
    assert "Ley Demo" in chunks[0].text
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_non_legal_source_falls_back_to_windows() -> None:
    source = Source(
        key="demo_faq",
        title="FAQ Demo",
        issuer="Demo",
        url="https://example.com",
        kind="html",
        year=2025,
        legal=False,
    )
    text = "\n\n".join("Pregunta y respuesta de ejemplo. " * 12 for _ in range(10))
    chunks = chunk_source(source, text)
    assert len(chunks) > 1
    assert all(chunk.article is None for chunk in chunks)


def test_normalize_collapses_noise() -> None:
    assert normalize("a\u00a0 b\n\n\n\nc") == "a b\n\nc"
