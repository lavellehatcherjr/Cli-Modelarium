"""Tests for cli_modelarium.providers.mistral_provider.

Mistral's streaming API: `client.chat.stream_async(...)` is a coroutine that
resolves to an AsyncIterator of CompletionEvent objects. Each event's
`.data.choices[0].delta.content` is the text delta; the final event's
`.data.usage` carries the OpenAI-style `{prompt,completion}_tokens`.

Errors raise SDKError, which wraps an httpx response we read the status code from.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mistralai.client.errors.sdkerror import SDKError

from cli_modelarium.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from cli_modelarium.providers.mistral_provider import MistralProvider


# ===== fake plumbing =====


def _make_event(text: str | None = None, usage: SimpleNamespace | None = None) -> SimpleNamespace:
    """Build a CompletionEvent shape: event.data.choices[0].delta.content + event.data.usage."""
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta)
    data = SimpleNamespace(choices=[choice], usage=usage)
    return SimpleNamespace(data=data)


class _FakeAsyncIterator:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self) -> "_FakeAsyncIterator":
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeChat:
    def __init__(
        self, events: list[Any] | None = None, error: Exception | None = None
    ) -> None:
        self._events = events
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    async def stream_async(self, **kwargs: Any) -> _FakeAsyncIterator:
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return _FakeAsyncIterator(self._events or [])


class _FakeClient:
    def __init__(self, chat: _FakeChat) -> None:
        self.chat = chat


def _build_events(
    texts: list[str], input_tokens: int = 0, output_tokens: int = 0
) -> list[Any]:
    events = [_make_event(text=t) for t in texts]
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    events.append(_make_event(text=None, usage=usage))
    return events


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    events: list[Any] | None = None,
    error: Exception | None = None,
) -> tuple[MistralProvider, _FakeChat]:
    chat = _FakeChat(events=events, error=error)

    def fake_mistral_factory(**_kwargs: Any) -> _FakeClient:
        return _FakeClient(chat)

    monkeypatch.setattr(
        "cli_modelarium.providers.mistral_provider.Mistral", fake_mistral_factory
    )
    provider = MistralProvider(api_key="abc123XYZ456DEF789ghi0jkl")
    return provider, chat


# ===== happy path =====


async def test_complete_returns_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _build_events(["Hello", ", ", "world!"], input_tokens=10, output_tokens=3)
    provider, _ = _make_provider(monkeypatch, events=events)

    result = await provider.complete("hi", "mistral-large-latest", 0.0)

    assert result.output == "Hello, world!"
    assert result.model == "mistral-large-latest"
    assert result.provider == "mistral"
    assert result.error is None


async def test_complete_captures_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _build_events(["x"], input_tokens=42, output_tokens=7)
    provider, _ = _make_provider(monkeypatch, events=events)

    result = await provider.complete("p", "mistral-large-latest", 0.0)

    assert result.input_tokens == 42
    assert result.output_tokens == 7


async def test_complete_calculates_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    # mistral-large-latest: $0.50/M input, $1.50/M output.
    events = _build_events(["x"], input_tokens=1_000_000, output_tokens=1_000_000)
    provider, _ = _make_provider(monkeypatch, events=events)

    result = await provider.complete("p", "mistral-large-latest", 0.0)

    assert result.cost_usd == pytest.approx(2.0)


async def test_system_prompt_in_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mistral uses OpenAI-style: system prompt is a role:system message in the array."""
    events = _build_events(["ok"], input_tokens=1, output_tokens=1)
    provider, chat = _make_provider(monkeypatch, events=events)

    await provider.complete(
        "user prompt", "mistral-large-latest", 0.0, system_prompt="you are helpful"
    )

    messages = chat.last_kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "user prompt"},
    ]


async def test_no_system_prompt_omits_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _build_events(["ok"], input_tokens=1, output_tokens=1)
    provider, chat = _make_provider(monkeypatch, events=events)

    await provider.complete("p", "mistral-large-latest", 0.0)

    assert chat.last_kwargs["messages"] == [{"role": "user", "content": "p"}]


async def test_temperature_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _build_events(["x"], input_tokens=1, output_tokens=1)
    provider, chat = _make_provider(monkeypatch, events=events)

    await provider.complete("p", "mistral-large-latest", 0.7)

    assert chat.last_kwargs["temperature"] == 0.7


async def test_records_ttft_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _build_events(["a", "b"], input_tokens=1, output_tokens=2)
    provider, _ = _make_provider(monkeypatch, events=events)

    result = await provider.complete("p", "mistral-large-latest", 0.0)

    assert result.ttft_ms is not None
    assert result.latency_ms >= (result.ttft_ms or 0)


# ===== stream() iteration =====


async def test_stream_yields_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _build_events(["one", " two", " three"], input_tokens=1, output_tokens=3)
    provider, _ = _make_provider(monkeypatch, events=events)

    collected: list[str] = []
    async for chunk in provider.stream("p", "mistral-large-latest", 0.0):
        collected.append(chunk)

    assert collected == ["one", " two", " three"]


# ===== error translation =====


def _sdk_error(status_code: int, message: str = "error") -> SDKError:
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return SDKError(message=message, raw_response=response, body=None)


async def test_401_translated_to_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _sdk_error(401, "invalid key abc123XYZ456leaked")
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.complete("p", "mistral-large-latest", 0.0)

    assert exc_info.value.provider == "mistral"


async def test_429_translated_to_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _sdk_error(429, "rate limited")
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete("p", "mistral-large-latest", 0.0)

    assert exc_info.value.provider == "mistral"


async def test_500_translated_to_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _sdk_error(500, "internal")
    provider, _ = _make_provider(monkeypatch, error=err)

    with pytest.raises(ProviderError):
        await provider.complete("p", "mistral-large-latest", 0.0)
