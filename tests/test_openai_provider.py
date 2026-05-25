"""Tests for cli_modelarium.providers.openai_provider.

The AsyncOpenAI client is replaced wholesale before instantiation. The fakes
mirror just the surface the provider touches: `client.chat.completions.create()`
returning an async-iterable stream.
"""
from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest

from cli_modelarium.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderOverloadedError,
    RateLimitError,
)
from cli_modelarium.providers._utils import extract_retry_after
from cli_modelarium.providers.openai_provider import OpenAIProvider


# ===== fake client plumbing =====


class _FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    response: Any = None,
    error: Exception | None = None,
) -> tuple[OpenAIProvider, _FakeCompletions]:
    """Construct an OpenAIProvider whose .client is a controllable fake."""
    completions = _FakeCompletions(response=response, error=error)

    def fake_async_openai(**_kwargs: Any) -> _FakeClient:
        return _FakeClient(completions)

    monkeypatch.setattr(
        "cli_modelarium.providers.openai_provider.AsyncOpenAI", fake_async_openai
    )
    provider = OpenAIProvider(api_key="sk-proj-test1234567890abcdefghi")
    return provider, completions


# ===== happy path =====


async def test_complete_returns_full_text(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(
        text_chunks=["Hello", ", ", "world!"],
        input_tokens=10,
        output_tokens=3,
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("hi", "gpt-5.5", 0.0)

    assert result.output == "Hello, world!"
    assert result.model == "gpt-5.5"
    assert result.provider == "openai"
    assert result.temperature == 0.0
    assert result.error is None


async def test_complete_captures_token_counts(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(
        text_chunks=["abc"],
        input_tokens=42,
        output_tokens=7,
        cached_tokens=10,
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("p", "gpt-5.5", 0.0)

    assert result.input_tokens == 42
    assert result.output_tokens == 7
    assert result.cached_tokens == 10


async def test_complete_calculates_cost(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    # gpt-5.5: $5/M input, $30/M output, $0.50/M cached.
    # 1M input (no cache) + 1M output = $5 + $30 = $35
    stream = fake_openai_stream(
        text_chunks=["x"],
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("p", "gpt-5.5", 0.0)

    assert result.cost_usd == pytest.approx(35.0)


async def test_complete_records_ttft_and_latency(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(
        text_chunks=["a", "b"],
        input_tokens=1,
        output_tokens=2,
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("p", "gpt-5.5", 0.0)

    assert result.ttft_ms is not None
    assert result.ttft_ms >= 0
    assert result.latency_ms >= 0
    assert result.latency_ms >= (result.ttft_ms or 0)


async def test_system_prompt_prepended(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(text_chunks=["ok"], input_tokens=1, output_tokens=1)
    provider, completions = _make_provider(monkeypatch, response=stream)

    await provider.complete(
        "user prompt", "gpt-5.5", 0.0, system_prompt="you are helpful"
    )

    messages = completions.last_kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "user prompt"},
    ]


async def test_no_system_prompt_omits_system_message(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(text_chunks=["ok"], input_tokens=1, output_tokens=1)
    provider, completions = _make_provider(monkeypatch, response=stream)

    await provider.complete("user prompt", "gpt-5.5", 0.0)

    messages = completions.last_kwargs["messages"]
    assert messages == [{"role": "user", "content": "user prompt"}]


async def test_temperature_passed_through(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(text_chunks=["x"], input_tokens=1, output_tokens=1)
    provider, completions = _make_provider(monkeypatch, response=stream)

    await provider.complete("p", "gpt-5.5", 0.7)

    assert completions.last_kwargs["temperature"] == 0.7


async def test_include_usage_requested_in_stream_options(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(text_chunks=["x"], input_tokens=1, output_tokens=1)
    provider, completions = _make_provider(monkeypatch, response=stream)

    await provider.complete("p", "gpt-5.5", 0.0)

    assert completions.last_kwargs["stream"] is True
    assert completions.last_kwargs["stream_options"] == {"include_usage": True}


# ===== stream() iteration =====


async def test_stream_yields_text_chunks(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    stream = fake_openai_stream(
        text_chunks=["one", " two", " three"], input_tokens=1, output_tokens=3
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    collected: list[str] = []
    async for chunk in provider.stream("p", "gpt-5.5", 0.0):
        collected.append(chunk)

    assert collected == ["one", " two", " three"]


# ===== error translation =====


def _build_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status, request=request, headers=headers or {})


async def test_authentication_error_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    err = openai.AuthenticationError(
        "invalid key sk-proj-leakedkey1234567890abcdefghi",
        response=_build_response(401),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.complete("p", "gpt-5.5", 0.0)

    # The leaked key string must be redacted.
    assert "leakedkey" not in str(exc_info.value)
    assert exc_info.value.provider == "openai"


async def test_rate_limit_error_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    err = openai.RateLimitError(
        "too many requests",
        response=_build_response(429, headers={"retry-after": "5"}),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete("p", "gpt-5.5", 0.0)

    assert exc_info.value.retry_after == 5.0
    assert exc_info.value.provider == "openai"


async def test_rate_limit_without_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    err = openai.RateLimitError(
        "too many requests",
        response=_build_response(429),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete("p", "gpt-5.5", 0.0)

    assert exc_info.value.retry_after is None


async def test_529_overloaded_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    err = openai.APIStatusError(
        "service overloaded",
        response=_build_response(529),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(ProviderOverloadedError):
        await provider.complete("p", "gpt-5.5", 0.0)


async def test_generic_api_error_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    err = openai.APIStatusError(
        "server error",
        response=_build_response(500),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(ProviderError):
        await provider.complete("p", "gpt-5.5", 0.0)


# ===== retry-after extraction helper =====


class TestExtractRetryAfter:
    def test_extracts_numeric_value(self) -> None:
        err = openai.RateLimitError(
            "rate", response=_build_response(429, headers={"retry-after": "10"}), body=None
        )
        assert extract_retry_after(err) == 10.0

    def test_extracts_case_insensitive(self) -> None:
        err = openai.RateLimitError(
            "rate", response=_build_response(429, headers={"Retry-After": "5"}), body=None
        )
        assert extract_retry_after(err) == 5.0

    def test_returns_none_when_missing(self) -> None:
        err = openai.RateLimitError("rate", response=_build_response(429), body=None)
        assert extract_retry_after(err) is None

    def test_returns_none_for_non_numeric(self) -> None:
        err = openai.RateLimitError(
            "rate",
            response=_build_response(429, headers={"retry-after": "Wed, 21 Oct"}),
            body=None,
        )
        assert extract_retry_after(err) is None


# ===== base_url for OpenAI-compatible subclasses =====


def test_base_url_forwarded_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> _FakeClient:
        captured.update(kwargs)
        return _FakeClient(_FakeCompletions())

    monkeypatch.setattr(
        "cli_modelarium.providers.openai_provider.AsyncOpenAI", capture
    )

    OpenAIProvider(api_key="sk-proj-test1234567890abcdefghi", base_url="https://example.invalid/v1")

    assert captured["api_key"] == "sk-proj-test1234567890abcdefghi"
    assert captured["base_url"] == "https://example.invalid/v1"


def test_no_base_url_omitted_from_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> _FakeClient:
        captured.update(kwargs)
        return _FakeClient(_FakeCompletions())

    monkeypatch.setattr(
        "cli_modelarium.providers.openai_provider.AsyncOpenAI", capture
    )

    OpenAIProvider(api_key="sk-proj-test1234567890abcdefghi")

    assert "base_url" not in captured


def test_transform_model_default_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _make_provider(monkeypatch)
    assert provider._transform_model("gpt-5.5") == "gpt-5.5"
