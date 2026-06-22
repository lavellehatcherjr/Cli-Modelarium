"""Tests for cli_modelarium.providers.zai_provider.

Z.AI / GLM is a thin OpenAIProvider subclass pointed at Z.AI's OpenAI-compatible
overseas endpoint. Unlike DashScope it adds no request-param overrides - GLM
streams and reports usage like any other OpenAI-compatible provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from cli_modelarium.models_registry import get_provider_for_model
from cli_modelarium.providers.openai_provider import OpenAIProvider
from cli_modelarium.providers.zai_provider import ZAIProvider

# ===== fake client plumbing (mirrors test_dashscope_provider.py) =====


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
) -> tuple[ZAIProvider, _FakeCompletions]:
    completions = _FakeCompletions(response)

    def fake_async_openai(**_kwargs: Any) -> _FakeClient:
        return _FakeClient(completions)

    monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", fake_async_openai)
    provider = ZAIProvider(api_key="zai-test1234567890abcdefghi")
    return provider, completions


# ===== identity / wiring =====


def test_provider_identity() -> None:
    assert ZAIProvider.name == "zai"
    assert ZAIProvider.BASE_URL == "https://api.z.ai/api/paas/v4/"


def test_subclasses_openai_provider() -> None:
    assert issubclass(ZAIProvider, OpenAIProvider)


def test_glm_models_route_to_zai() -> None:
    # Routing is data-driven from PRICING["provider"]; GLM ids resolve to zai.
    assert get_provider_for_model("glm-5.2") == "zai"
    assert get_provider_for_model("glm-4-32b-0414-128k") == "zai"


def test_no_extra_create_kwargs() -> None:
    # Z.AI is plain - it inherits the base no-op (no thinking-toggle).
    p = ZAIProvider.__new__(ZAIProvider)
    assert p._extra_create_kwargs() == {}


# ===== cost =====


async def test_complete_calculates_cost(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    # glm-5.2: $1.40/M input, $4.40/M output.
    # 1000 input + 500 output = $0.0014 + $0.0022 = $0.0036
    stream = fake_openai_stream(
        text_chunks=["hi"],
        input_tokens=1000,
        output_tokens=500,
    )
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("p", "glm-5.2", 0.0)

    assert result.provider == "zai"
    assert result.model == "glm-5.2"
    assert result.cost_usd == pytest.approx(0.0036)


async def test_free_glm_model_costs_zero(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    # glm-4.7-flash is $0 across the board - no divide-by-zero, cost is $0.
    stream = fake_openai_stream(text_chunks=["x"], input_tokens=1000, output_tokens=500)
    provider, _ = _make_provider(monkeypatch, response=stream)

    result = await provider.complete("p", "glm-4.7-flash", 0.0)

    assert result.cost_usd == 0.0


async def test_no_thinking_param_sent(
    monkeypatch: pytest.MonkeyPatch, fake_openai_stream: Any
) -> None:
    """Plain provider: no extra_body / thinking-toggle reaches create()."""
    stream = fake_openai_stream(text_chunks=["x"], input_tokens=1, output_tokens=1)
    provider, completions = _make_provider(monkeypatch, response=stream)

    await provider.complete("p", "glm-5.2", 0.0)

    assert "extra_body" not in completions.last_kwargs
