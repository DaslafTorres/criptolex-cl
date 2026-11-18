from criptolex.retrieve import HybridRetriever, Retrieved, index_text, tokenize


def test_index_text_drops_shared_header() -> None:
    text = "Ley 21.521 — Ley Fintech (BCN, 2023) — Artículo 5\n\nServicios regulados."
    assert index_text(text) == "Servicios regulados."
    assert index_text("sin encabezado") == "sin encabezado"


def test_tokenize_collapses_morphology() -> None:
    assert tokenize("obligaciones") == tokenize("obligación")
    assert tokenize("artículos") == tokenize("articulo")
    assert tokenize("registrar") == tokenize("registro")
    assert tokenize("criptomonedas") == tokenize("criptomoneda")


def test_tokenize_strips_accents_and_stopwords() -> None:
    tokens = tokenize("¿Cuál es el artículo que regula los activos virtuales?")
    assert tokens == tokenize("cual articulo regula activos virtuales")
    assert "que" not in tokens
    assert "los" not in tokens
    assert "el" not in tokens


def test_tokenize_keeps_numbers_and_acronyms() -> None:
    tokens = tokenize("NCG 502 y PLAFT")
    assert "ncg" in tokens
    assert "502" in tokens
    assert "plaft" in tokens


def _item(chunk_id: str, score: float = 0.0) -> Retrieved:
    return Retrieved(
        chunk_id=chunk_id,
        text=f"doc {chunk_id}\n\nbody",
        source_key="demo",
        article=None,
        score=score,
        stage="dense",
    )


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    dense = [_item("a"), _item("b"), _item("c")]
    sparse = [_item("c"), _item("a")]

    fused = HybridRetriever._fuse([dense, sparse], limit=3)
    assert [item.chunk_id for item in fused] == ["a", "c", "b"]
    assert fused[0].stage == "fused"


def test_snippet_drops_the_citation_header() -> None:
    item = Retrieved(
        chunk_id="x",
        text="Ley Demo (Demo, 2024) — Artículo 1\n\nContenido real del artículo.",
        source_key="demo",
        article="Artículo 1",
        score=1.0,
        stage="rerank",
    )
    assert item.snippet == "Contenido real del artículo."
    assert item.label() == "demo · Artículo 1"
