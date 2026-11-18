import httpx
import pytest

from criptolex import ingest
from criptolex.ingest import download
from criptolex.sources import Source

PDF = b"%PDF-1.4 fake"

SOURCE = Source(
    key="demo",
    title="Demo",
    issuer="Demo",
    url="https://example.com/demo.pdf",
    kind="pdf",
    year=2025,
    legal=False,
)


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_retries_transient_server_errors(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=PDF)

    monkeypatch.setattr(ingest.time, "sleep", lambda _seconds: None)
    with make_client(handler) as client:
        path = download(SOURCE, tmp_path, client)

    assert path.read_bytes() == PDF
    assert calls["n"] == 3


def test_download_does_not_retry_client_errors(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    monkeypatch.setattr(ingest.time, "sleep", lambda _seconds: None)
    with make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            download(SOURCE, tmp_path, client)

    assert calls["n"] == 1


def test_download_gives_up_after_retries(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    monkeypatch.setattr(ingest.time, "sleep", lambda _seconds: None)
    with make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            download(SOURCE, tmp_path, client)

    assert calls["n"] == ingest.RETRIES


def test_cached_download_skips_the_network(tmp_path) -> None:
    (tmp_path / "demo.pdf").write_bytes(PDF)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not hit the network")

    with make_client(handler) as client:
        path = download(SOURCE, tmp_path, client)

    assert path.read_bytes() == PDF
