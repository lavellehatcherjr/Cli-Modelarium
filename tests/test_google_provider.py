"""Tests for cli_modelarium.providers.google_provider.

The google-genai async API is `client.aio.models.generate_content_stream(...)`,
which is a coroutine that resolves to an AsyncIterator. Each chunk has a `.text`
attribute and the final chunk(s) carry `.usage_metadata` with Google-specific
field names (`prompt_token_count`, `candidates_token_count`,
`cached_content_token_count`).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors

from cli_modelarium.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from cli_modelarium.providers.google_provider import GoogleProvider


# ===== fake plumbing =====


class _FakeChunk:
    def __init__(self, text: str | None = None, usage: SimpleNamespace | None = None) -> None:
        self.text = text
        self.usage_metadata = usage


class _FakeAsyncIterator:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> "_FakeAsyncIterator":
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> _FakeChunk:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeAsyncModels:
    def __init__(
        self,
        chunks: list[_FakeChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    async def generate_content_stream(self, **kwargs: Any) -> _FakeAsyncIterator:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return _FakeAsyncIterator(self._chunks or [])


class _FakeAio:
    def __init__(self, models: _FakeAsyncModels) -> None:
        self.models = models


class _FakeClient:
    def __init__(self, models: _FakeAsyncModels) -> None:
        self.aio = _FakeAio(models)


def _build_chunks(
    texts: list[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> list[_FakeChunk]:
    chunks = [_FakeChunk(text=t) for t in texts]
    usage = SimpleNamespace(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens,
        cached_content_token_count=cached_tokens,
    )
    chunks.append(_FakeChunk(text=None, usage=usage))
    return chunks


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[_FakeChunk] | None = None,
    error: Exception | None = None,
) -> tuple[GoogleProvider, _FakeAsyncModels]:
    models = _FakeAsyncModels(chunks=chunks, error=error)

    def fake_client_factory(**_kwargs: Any) -> _FakeClient:
        return _FakeClient(models)

    monkeypatch.setattr(
        "cli_modelarium.providers.google_provider.genai.Client", fake_client_factory
    )
    provider = GoogleProvider(api_key="test-google-api-key-123456789012345")
    return provider, models


# ===== happy path =====


async def test_complete_returns_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _build_chunks(["Hello", ", ", "world!"], input_tokens=10, output_tokens=3)
    provider, _ = _make_provider(monkeypatch, chunks=chunks)

    result = await provider.complete("hi", "gemini-3.1-pro", 0.0)

    assert result.output == "Hello, world!"
    assert result.model == "gemini-3.1-pro"
    assert result.provider == "google"
    assert result.error is None


async def test_complete_captures_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _build_chunks(["x"], input_tokens=42, output_tokens=7, cached_tokens=11)
    provider, _ = _make_provider(monkeypatch, chunks=chunks)

    result = await provider.complete("p", "gemini-3.1-pro", 0.0)

    assert result.input_tokens == 42
    assert result.output_tokens == 7
    assert result.cached_tokens == 11


async def test_complete_calculates_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    # gemini-3.1-pro: $1.25/M input, $5.00/M output.
    chunks = _build_chunks(["x"], input_tokens=1_000_000, output_tokens=1_000_000)
    provider, _ = _make_provider(monkeypatch, chunks=chunks)

    result = await provider.complete("p", "gemini-3.1-pro", 0.0)

    assert result.cost_usd == pytest.approx(6.25)


async def test_system_prompt_in_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google's system prompt goes inside `config["system_instruction"]`."""
    chunks = _build_chunks(["ok"], input_tokens=1, output_tokens=1)
    provider, models = _make_provider(monkeypatch, chunks=chunks)

    await provider.complete(
        "user prompt", "gemini-3.1-pro", 0.0, system_prompt="you are helpful"
    )

    config = models.last_kwargs["config"]
    assert config["system_instruction"] == "you are helpful"
    # And the contents stay just the user prompt.
    assert models.last_kwargs["contents"] == "user prompt"


async def test_no_system_prompt_omits_system_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _build_chunks(["ok"], input_tokens=1, output_tokens=1)
    provider, models = _make_provider(monkeypatch, chunks=chunks)

    await provider.complete("p", "gemini-3.1-pro", 0.0)

    config = models.last_kwargs["config"]
    assert "system_instruction" not in config


async def test_temperature_passed_in_config(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _build_chunks(["x"], input_tokens=1, output_tokens=1)
    provider, models = _make_provider(monkeypatch, chunks=chunks)

    await provider.complete("p", "gemini-3.1-pro", 0.7)

    assert models.last_kwargs["config"]["temperature"] == 0.7


async def test_records_ttft_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _build_chunks(["a", "b"], input_tokens=1, output_tokens=2)
    provider, _ = _make_provider(monkeypatch, chunks=chunks)

    result = await provider.complete("p", "gemini-3.1-pro", 0.0)

    assert result.ttft_ms is not None
    assert result.latency_ms >= (result.ttft_ms or 0)


# ===== stream() iteration =====


async def test_stream_yields_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _build_chunks(["one", " two", " three"], input_tokens=1, output_tokens=3)
    provider, _ = _make_provider(monkeypatch, chunks=chunks)

    collected: list[str] = []
    async for chunk in provider.stream("p", "gemini-3.1-pro", 0.0):
        collected.append(chunk)

    assert collected == ["one", " two", " three"]


# ===== error translation =====


def _client_error(code: int, message: str) -> genai_errors.ClientError:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1/models")
    response = httpx.Response(code, request=request)
    return genai_errors.ClientError(
        code, response_json={"error": {"code": code, "message": message}}, response=response
    )


async def test_401_translated_to_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _client_error(401, "invalid key AIzaSyLeaked1234567890abcdef")
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.complete("p", "gemini-3.1-pro", 0.0)

    assert "Leaked" not in str(exc_info.value)
    assert exc_info.value.provider == "google"


async def test_403_translated_to_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _client_error(403, "forbidden")
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(AuthenticationError):
        await provider.complete("p", "gemini-3.1-pro", 0.0)


async def test_429_translated_to_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _client_error(429, "quota exceeded")
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete("p", "gemini-3.1-pro", 0.0)

    assert exc_info.value.provider == "google"


async def test_500_translated_to_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1/models")
    response = httpx.Response(500, request=request)
    err = genai_errors.ServerError(
        500, response_json={"error": {"code": 500, "message": "internal"}}, response=response
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(ProviderError):
        await provider.complete("p", "gemini-3.1-pro", 0.0)
