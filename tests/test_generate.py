from types import SimpleNamespace

import httpx
import pytest
from groq import RateLimitError

from criptolex.generate import (
    REFUSAL,
    SYSTEM_PROMPT,
    GroqClient,
    build_prompt,
    clean_citations,
    extractive_answer,
    generate_answer,
    is_refusal,
    retry_wait,
)
from criptolex.retrieve import Retrieved

HIT = Retrieved(
    chunk_id="ley_21521:0003",
    text=(
        "Ley 21.521 — Ley Fintech (BCN, 2023) — Artículo 5\n\n"
        "Servicios regulados y obligación de inscripción. Sólo podrán dedicarse en "
        "forma profesional a la prestación de servicios de plataforma de "
        "financiamiento colectivo."
    ),
    source_key="ley_21521",
    article="Artículo 5",
    score=0.9,
    stage="rerank",
    doc_title="Ley 21.521 — Ley Fintech",
    issuer="BCN",
    url="https://example.com/ley",
    year=2023,
)


class FakeClient:
    model = "fake-model"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: tuple[str, str] | None = None

    def complete(self, system: str, user: str) -> str:
        self.seen = (system, user)
        return self.reply


def test_prompt_numbers_sources_and_skips_shared_header() -> None:
    prompt = build_prompt("¿Quién debe inscribirse?", [HIT])
    assert "[1] Ley 21.521 — Ley Fintech (BCN, 2023) · Artículo 5" in prompt
    assert "URL: https://example.com/ley" in prompt
    assert "Servicios regulados" in prompt
    assert "Pregunta: ¿Quién debe inscribirse?" in prompt


def test_prompt_keeps_both_sides_of_the_grounding_rule() -> None:
    """Guards the rule that fixed the false refusal and the out-of-scope answer."""
    assert "al menos un extracto contiene el hecho" in SYSTEM_PROMPT
    assert "ningún extracto describe el hecho" in SYSTEM_PROMPT


def test_clean_citations_drops_invented_references() -> None:
    text, cited = clean_citations("Una regla [1] y otra inventada [9].", n_sources=2)
    assert cited == [1]
    assert "[9]" not in text
    assert "[1]" in text


def test_clean_citations_handles_lists_and_duplicates() -> None:
    _, cited = clean_citations("A [1, 2] y B [2].", n_sources=3)
    assert cited == [1, 2]


def test_clean_citations_normalises_fullwidth_brackets() -> None:
    text, cited = clean_citations("Definición 【1】 y ejemplo 【2】.", n_sources=3)
    assert cited == [1, 2]
    assert "[1]" in text and "[2]" in text
    assert "【" not in text


def test_generate_answer_returns_only_cited_sources() -> None:
    client = FakeClient("Los prestadores deben inscribirse [1][9].")
    answer = generate_answer("¿Quién debe inscribirse?", [HIT], client)

    assert answer.grounded is True
    assert answer.refused is False
    assert [source.n for source in answer.sources] == [1]
    assert answer.sources[0].article == "Artículo 5"
    assert answer.model == "fake-model"
    assert client.seen is not None


def test_generate_answer_without_citations_is_not_grounded() -> None:
    answer = generate_answer("¿Quién debe inscribirse?", [HIT], FakeClient("Depende."))
    assert answer.grounded is False
    assert answer.sources == []
    assert answer.refused is False


def test_model_refusal_is_normalised() -> None:
    client = FakeClient("No encontré respaldo suficiente en las fuentes oficiales.")
    answer = generate_answer("¿Cuál es la multa?", [HIT], client)
    assert answer.refused is True
    assert answer.text == REFUSAL
    assert answer.sources == []


def test_no_hits_short_circuits_the_model() -> None:
    client = FakeClient("no debería llamarse")
    answer = generate_answer("¿Cuál es la multa?", [], client)
    assert answer.refused is True
    assert client.seen is None


def test_is_refusal_ignores_case_and_spacing() -> None:
    assert is_refusal("  NO ENCONTRÉ   respaldo suficiente en las fuentes.")


def test_extractive_answer_quotes_the_best_hit() -> None:
    answer = extractive_answer("¿Quién debe inscribirse?", [HIT])
    assert answer.model == "extractive"
    assert answer.grounded is True
    assert answer.sources[0].n == 1
    assert "Sólo podrán dedicarse" in answer.text
    assert answer.text.endswith("[1]")


def rate_limit_error(retry_after: str = "0") -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
    return RateLimitError("rate limited", response=response, body=None)


class FakeCompletions:
    def __init__(self, failures: int = 0, content: str = "ok") -> None:
        self.failures = failures
        self.content = content
        self.calls = 0
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.calls <= self.failures:
            raise rate_limit_error("0.01")
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )


def client_with(completions: FakeCompletions) -> GroqClient:
    client = GroqClient("test-key", "fake-model", max_tokens=64)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client


def test_groq_client_retries_rate_limit_then_succeeds(monkeypatch) -> None:
    completions = FakeCompletions(failures=1)
    client = client_with(completions)
    monkeypatch.setattr("criptolex.generate.time.sleep", lambda _seconds: None)

    assert client.complete("system", "user") == "ok"
    assert completions.calls == 2
    assert (client.prompt_tokens, client.completion_tokens) == (10, 5)


def test_groq_client_gives_up_after_three_rate_limits(monkeypatch) -> None:
    completions = FakeCompletions(failures=99)
    client = client_with(completions)
    monkeypatch.setattr("criptolex.generate.time.sleep", lambda _seconds: None)

    with pytest.raises(RateLimitError):
        client.complete("system", "user")
    assert completions.calls == 3


def test_groq_client_sends_configured_budget() -> None:
    completions = FakeCompletions()
    client_with(completions).complete("system", "user")
    assert completions.kwargs["max_tokens"] == 64
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["model"] == "fake-model"


def test_retry_wait_reads_header_and_caps_it() -> None:
    assert retry_wait(rate_limit_error("2.5")) == 2.5
    assert retry_wait(rate_limit_error("999")) == 60.0
    assert retry_wait(rate_limit_error("not-a-number")) == 5.0
