from fastapi.testclient import TestClient

from criptolex.api import answerer_dep, app, retriever_dep
from criptolex.generate import Answer, Source
from criptolex.retrieve import Retrieved

HIT = Retrieved(
    chunk_id="sii_faq_cripto:0003",
    text="FAQ SII\n\nSí, corresponde declarar y pagar impuestos.",
    source_key="sii_faq_cripto",
    article=None,
    score=0.8,
    stage="rerank",
    doc_title="FAQ criptomonedas",
    issuer="SII",
    url="https://example.com/faq",
    year=2025,
)


class FakeRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self.hits = hits
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        self.queries.append((query, k))
        return self.hits


def test_ask_returns_answer_and_sources() -> None:
    retriever = FakeRetriever([HIT])

    def answerer(question: str, hits: list[Retrieved]) -> Answer:
        return Answer(
            text="Sí, se declaran [1].",
            sources=[Source.from_hit(1, hits[0])],
            grounded=True,
            refused=False,
            model="fake",
        )

    app.dependency_overrides[retriever_dep] = lambda: retriever
    app.dependency_overrides[answerer_dep] = lambda: answerer
    try:
        response = TestClient(app).post("/ask", json={"question": "¿Pago impuestos?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Sí, se declaran [1]."
    assert body["grounded"] is True
    assert body["sources"][0]["source_key"] == "sii_faq_cripto"
    assert body["sources"][0]["url"] == "https://example.com/faq"
    assert retriever.queries == [("¿Pago impuestos?", 5)]
    assert isinstance(body["latency_ms"], int)


def test_ask_passes_k() -> None:
    retriever = FakeRetriever([])
    app.dependency_overrides[retriever_dep] = lambda: retriever
    app.dependency_overrides[answerer_dep] = lambda: (
        lambda question, hits: Answer(
            text="sin fuentes", sources=[], grounded=False, refused=True, model="fake"
        )
    )
    try:
        response = TestClient(app).post("/ask", json={"question": "¿Qué dice la norma?", "k": 3})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert retriever.queries == [("¿Qué dice la norma?", 3)]
    assert response.json()["refused"] is True


def test_ask_rejects_short_questions() -> None:
    app.dependency_overrides[retriever_dep] = lambda: FakeRetriever([])
    app.dependency_overrides[answerer_dep] = lambda: (
        lambda question, hits: Answer(
            text="", sources=[], grounded=False, refused=True, model="fake"
        )
    )
    try:
        response = TestClient(app).post("/ask", json={"question": "ab"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


def test_ask_returns_502_when_generation_fails() -> None:
    app.dependency_overrides[retriever_dep] = lambda: FakeRetriever([HIT])

    def boom(question: str, hits: list[Retrieved]) -> Answer:
        raise RuntimeError("upstream down")

    app.dependency_overrides[answerer_dep] = lambda: boom
    try:
        response = TestClient(app).post("/ask", json={"question": "¿Pago impuestos?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "No se pudo generar" in response.json()["detail"]


def test_health_reports_provider() -> None:
    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] in {"groq", "extractive"}
