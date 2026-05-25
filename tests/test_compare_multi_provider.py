"""Integration tests for cli.py:_run_compare across multiple providers.

These tests don't mock individual SDKs - they replace `_get_provider_instance`
itself with a factory returning fakes, then verify the orchestration code:

    * routes each model to its correct provider
    * reuses one provider instance for multiple models from the same provider
    * runs all calls in parallel
    * one provider failure does not kill the others
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from cli_modelarium.cli import _run_compare
from cli_modelarium.exceptions import ProviderError
from cli_modelarium.providers.base import BaseProvider, CompletionResult


class _RecordingProvider(BaseProvider):
    """Fake provider that returns a preset CompletionResult or raises a preset error."""

    def __init__(self, name: str, result_or_error: CompletionResult | Exception) -> None:
        self.name = name
        self._result_or_error = result_or_error
        self.call_count = 0
        self.calls: list[tuple[str, str, float, str | None]] = []

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""
        raise NotImplementedError("stream not exercised in these tests")

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        self.call_count += 1
        self.calls.append((prompt, model, temperature, system_prompt))
        if isinstance(self._result_or_error, Exception):
            raise self._result_or_error
        return replace(self._result_or_error, model=model, temperature=temperature)


async def test_runs_three_providers_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = {
        "openai": _RecordingProvider("openai", CompletionResult(output="oa out", provider="openai")),
        "anthropic": _RecordingProvider(
            "anthropic", CompletionResult(output="ant out", provider="anthropic")
        ),
        "google": _RecordingProvider(
            "google", CompletionResult(output="google out", provider="google")
        ),
    }
    monkeypatch.setattr("cli_modelarium.cli._get_provider_instance", lambda name: fakes[name])

    results = await _run_compare(
        prompt="test prompt",
        models=["gpt-5.5", "claude-opus-4-7", "gemini-3.1-pro"],
        temperatures=[0.0],
        system_prompt=None,
    )

    assert len(results) == 3
    outputs = {r.output for r in results}
    assert outputs == {"oa out", "ant out", "google out"}
    for f in fakes.values():
        assert f.call_count == 1


async def test_one_failure_does_not_kill_others(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = {
        "openai": _RecordingProvider("openai", CompletionResult(output="ok", provider="openai")),
        "anthropic": _RecordingProvider(
            "anthropic", ProviderError("server died", provider="anthropic")
        ),
        "google": _RecordingProvider(
            "google", CompletionResult(output="ok2", provider="google")
        ),
    }
    monkeypatch.setattr("cli_modelarium.cli._get_provider_instance", lambda name: fakes[name])

    results = await _run_compare(
        prompt="p",
        models=["gpt-5.5", "claude-opus-4-7", "gemini-3.1-pro"],
        temperatures=[0.0],
        system_prompt=None,
    )

    assert len(results) == 3
    errors = [r for r in results if r.error]
    successes = [r for r in results if r.error is None]
    assert len(errors) == 1
    assert errors[0].model == "claude-opus-4-7"
    assert errors[0].provider == "anthropic"
    assert len(successes) == 2


async def test_one_provider_instance_per_provider_not_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If multiple models come from one provider, only ONE instance is created."""
    instantiations: dict[str, int] = {}
    fakes: dict[str, _RecordingProvider] = {}

    def make(name: str) -> _RecordingProvider:
        instantiations[name] = instantiations.get(name, 0) + 1
        provider = _RecordingProvider(
            name, CompletionResult(output=f"{name} ok", provider=name)
        )
        fakes[name] = provider
        return provider

    monkeypatch.setattr("cli_modelarium.cli._get_provider_instance", make)

    # Two OpenAI models + one Anthropic model = 2 unique providers, 3 model calls
    results = await _run_compare(
        prompt="p",
        models=["gpt-5.5", "gpt-5.4", "claude-opus-4-7"],
        temperatures=[0.0],
        system_prompt=None,
    )

    assert len(results) == 3
    assert instantiations == {"openai": 1, "anthropic": 1}
    assert fakes["openai"].call_count == 2  # both openai models routed through same instance
    assert fakes["anthropic"].call_count == 1


async def test_multiple_temperatures_fan_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """N models x M temperatures = N*M tasks, all using one client per provider."""
    fake = _RecordingProvider("openai", CompletionResult(output="x", provider="openai"))
    monkeypatch.setattr("cli_modelarium.cli._get_provider_instance", lambda _: fake)

    results = await _run_compare(
        prompt="p",
        models=["gpt-5.5", "gpt-5.4"],
        temperatures=[0.0, 0.5, 1.0],
        system_prompt=None,
    )

    assert len(results) == 6  # 2 models x 3 temperatures
    assert fake.call_count == 6
    temps_seen = sorted(t for _, _, t, _ in fake.calls)
    assert temps_seen == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


async def test_system_prompt_threaded_through(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _RecordingProvider("openai", CompletionResult(output="x", provider="openai"))
    monkeypatch.setattr("cli_modelarium.cli._get_provider_instance", lambda _: fake)

    await _run_compare(
        prompt="user",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompt="you are helpful",
    )

    assert fake.calls[0][3] == "you are helpful"


async def test_unexpected_exception_caught_as_result_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected non-ModelariumError should become an error CompletionResult,
    not propagate and kill the whole run.
    """
    fakes = {
        "openai": _RecordingProvider("openai", RuntimeError("kaboom")),
        "anthropic": _RecordingProvider(
            "anthropic", CompletionResult(output="ok", provider="anthropic")
        ),
    }
    monkeypatch.setattr("cli_modelarium.cli._get_provider_instance", lambda name: fakes[name])

    results = await _run_compare(
        prompt="p",
        models=["gpt-5.5", "claude-opus-4-7"],
        temperatures=[0.0],
        system_prompt=None,
    )

    assert len(results) == 2
    errors = [r for r in results if r.error]
    assert len(errors) == 1
    assert "kaboom" in (errors[0].error or "")
