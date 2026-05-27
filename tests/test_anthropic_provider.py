"""Tests for cli_modelarium.providers.anthropic_provider.

Anthropic's streaming API exposes an async context manager whose inner stream
has `.text_stream` (an async iterator of strings) and `.get_final_message()`
(an awaitable returning a Message with `.usage`). The fakes here implement
that protocol exactly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from cli_modelarium.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderOverloadedError,
    RateLimitError,
)
from cli_modelarium.providers.anthropic_provider import (
    DEFAULT_MAX_TOKENS,
    AnthropicProvider,
)

# ===== fake stream plumbing =====


class _FakeInnerStream:
    """The object yielded inside `async with messages.stream(...) as stream:`."""

    def __init__(
        self,
        text_chunks: list[str],
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        self._chunks = text_chunks
        self._usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cached_tokens,
        )

    @property
    def text_stream(self) -> Any:
        async def gen() -> Any:
            for t in self._chunks:
                yield t

        return gen()

    async def get_final_message(self) -> Any:
        return SimpleNamespace(usage=self._usage)


class _FakeStreamManager:
    """The object returned by messages.stream(); it's an async context manager."""

    def __init__(self, inner: _FakeInnerStream) -> None:
        self._inner = inner

    async def __aenter__(self) -> _FakeInnerStream:
        return self._inner

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeMessages:
    def __init__(
        self, inner: _FakeInnerStream | None = None, error: Exception | None = None
    ) -> None:
        self._inner = inner
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    def stream(self, **kwargs: Any) -> _FakeStreamManager:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        assert self._inner is not None
        return _FakeStreamManager(self._inner)


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    text_chunks: list[str] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    error: Exception | None = None,
) -> tuple[AnthropicProvider, _FakeMessages]:
    inner = (
        _FakeInnerStream(
            text_chunks or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
        if error is None
        else None
    )
    messages = _FakeMessages(inner=inner, error=error)

    def fake_async_anthropic(**_kwargs: Any) -> _FakeClient:
        return _FakeClient(messages)

    monkeypatch.setattr(
        "cli_modelarium.providers.anthropic_provider.AsyncAnthropic", fake_async_anthropic
    )
    provider = AnthropicProvider(api_key="sk-ant-api03-test1234567890abcdefghi")
    return provider, messages


# ===== happy path =====


async def test_complete_returns_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _make_provider(
        monkeypatch, text_chunks=["Hello", ", ", "world!"], input_tokens=10, output_tokens=3
    )

    result = await provider.complete("hi", "claude-opus-4-7", 0.0)

    assert result.output == "Hello, world!"
    assert result.model == "claude-opus-4-7"
    assert result.provider == "anthropic"
    assert result.error is None


async def test_complete_captures_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _make_provider(
        monkeypatch, text_chunks=["x"], input_tokens=42, output_tokens=7, cached_tokens=11
    )

    result = await provider.complete("p", "claude-opus-4-7", 0.0)

    assert result.input_tokens == 42
    assert result.output_tokens == 7
    assert result.cached_tokens == 11


async def test_complete_calculates_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    # claude-opus-4-7: $5/M input, $25/M output (user-corrected pricing).
    provider, _ = _make_provider(
        monkeypatch, text_chunks=["x"], input_tokens=1_000_000, output_tokens=1_000_000
    )

    result = await provider.complete("p", "claude-opus-4-7", 0.0)

    assert result.cost_usd == pytest.approx(30.0)


async def test_complete_records_ttft_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _make_provider(
        monkeypatch, text_chunks=["a", "b"], input_tokens=1, output_tokens=2
    )

    result = await provider.complete("p", "claude-opus-4-7", 0.0)

    assert result.ttft_ms is not None
    assert result.ttft_ms >= 0
    assert result.latency_ms >= (result.ttft_ms or 0)


# ===== system prompt + max_tokens contracts =====


async def test_system_prompt_passed_at_top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 3 critical contract: system goes top-level, NOT in messages array."""
    provider, messages = _make_provider(
        monkeypatch, text_chunks=["ok"], input_tokens=1, output_tokens=1
    )

    await provider.complete("user prompt", "claude-opus-4-7", 0.0, system_prompt="you are helpful")

    # `system` must be a top-level kwarg, NOT in the messages array.
    assert messages.last_kwargs["system"] == "you are helpful"
    msgs = messages.last_kwargs["messages"]
    assert msgs == [{"role": "user", "content": "user prompt"}]
    for m in msgs:
        assert m["role"] != "system"


async def test_no_system_prompt_omits_system_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, messages = _make_provider(
        monkeypatch, text_chunks=["ok"], input_tokens=1, output_tokens=1
    )

    await provider.complete("p", "claude-opus-4-7", 0.0)

    assert "system" not in messages.last_kwargs


async def test_max_tokens_required_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_tokens MUST be set on every call (Anthropic rejects without it)."""
    provider, messages = _make_provider(
        monkeypatch, text_chunks=["ok"], input_tokens=1, output_tokens=1
    )

    await provider.complete("p", "claude-opus-4-7", 0.0)

    assert messages.last_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS


async def test_temperature_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, messages = _make_provider(
        monkeypatch, text_chunks=["x"], input_tokens=1, output_tokens=1
    )

    await provider.complete("p", "claude-opus-4-7", 0.7)

    assert messages.last_kwargs["temperature"] == 0.7


# ===== stream() iteration =====


async def test_stream_yields_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _make_provider(
        monkeypatch, text_chunks=["one", " two", " three"], input_tokens=1, output_tokens=3
    )

    collected: list[str] = []
    async for chunk in provider.stream("p", "claude-opus-4-7", 0.0):
        collected.append(chunk)

    assert collected == ["one", " two", " three"]


# ===== error translation =====


def _build_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status, request=request, headers=headers or {})


async def test_authentication_error_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    err = anthropic.AuthenticationError(
        "invalid key sk-ant-leakedkey1234567890abcdefghi",
        response=_build_response(401),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.complete("p", "claude-opus-4-7", 0.0)

    # Key string must be scrubbed.
    assert "leakedkey" not in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


async def test_rate_limit_error_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    err = anthropic.RateLimitError(
        "too many requests",
        response=_build_response(429, headers={"retry-after": "7"}),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete("p", "claude-opus-4-7", 0.0)

    assert exc_info.value.retry_after == 7.0
    assert exc_info.value.provider == "anthropic"


async def test_529_overloaded_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic's 529 'overloaded_error' is distinct from 429."""
    err = anthropic.APIStatusError(
        "overloaded",
        response=_build_response(529),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(ProviderOverloadedError):
        await provider.complete("p", "claude-opus-4-7", 0.0)


async def test_generic_api_error_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    err = anthropic.APIStatusError(
        "server error",
        response=_build_response(500),
        body=None,
    )
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(ProviderError):
        await provider.complete("p", "claude-opus-4-7", 0.0)
