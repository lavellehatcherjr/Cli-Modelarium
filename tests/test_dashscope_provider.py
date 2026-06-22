"""Tests for cli_modelarium.providers.dashscope_provider.

DashScope is a thin OpenAIProvider subclass pointed at Alibaba's
International/Singapore endpoint. It overrides `_extra_create_kwargs()` to send
`enable_thinking=False` so usage (and therefore cost) reflects the non-thinking
rates stored in the pricing table.
"""

from __future__ import annotations

from typing import Any

import pytest

from cli_modelarium.providers.dashscope_provider import DashScopeProvider
from cli_modelarium.providers.openai_provider import OpenAIProvider

# ===== fake client plumbing (mirrors test_openai_provider.py) =====


class _FakeCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return self._response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, response: Any
) -> tuple[DashScopeProvider, _FakeCompletions]:
    completions = _FakeCompletions(response)

    def fake_async_openai(**_kwargs: Any) -> _FakeClient:
        return _FakeClient(completions)

    monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", fake_async_openai)
    provider = DashScopeProvider(api_key="sk-test1234567890abcdefghi")
    return provider, completions


# ===== thinking-mode hook =====


def test_extra_create_kwargs_disables_thinking() -> None:
    p = DashScopeProvider.__new__(DashScopeProvider)
    assert p._extra_create_kwargs() == {"extra_body": {"enable_thinking": False}}


def test_base_extra_create_kwargs_is_empty() -> None:
    # The shared base defaults to a no-op so existing providers are unaffected.
    p = OpenAIProvider.__new__(OpenAIProvider)
    assert p._extra_create_kwargs() == {}


# ===== cost =====


async def test_complete_calculates_cost(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    # qwen3.7-max: $2.50/M input, $7.50/M output.
    # 1000 input + 500 output = $0.0025 + $0.00375 = $0.00625
    stream = fake_openai_stream(
        text_chunks=["hi"],
        input_tokens=1000,
        output_tokens=500,
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("p", "qwen3.7-max", 0.0)

    assert result.provider == "dashscope"
    assert result.model == "qwen3.7-max"
    assert result.input_tokens == 1000
    assert result.output_tokens == 500
    assert result.cost_usd == pytest.approx(0.00625)


async def test_enable_thinking_false_sent_to_create(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    """The hook must actually reach chat.completions.create() via complete()."""
    stream = fake_openai_stream(text_chunks=["x"], input_tokens=1, output_tokens=1)
    provider, completions = _make_provider(monkeypatch, response=stream)

    await provider.complete("p", "qwen3.7-max", 0.0)

    assert completions.last_kwargs["extra_body"] == {"enable_thinking": False}
