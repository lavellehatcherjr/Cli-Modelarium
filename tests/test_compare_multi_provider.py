"""Integration tests for run_streaming_comparison across multiple providers.

These tests don't mock individual SDKs - they pass a `provider_factory` of
fakes directly, then verify the orchestration code:

    * routes each model to its correct provider
    * reuses one provider instance for multiple models from the same provider
    * runs all calls in parallel
    * one provider failure does not kill the others
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

from cli_modelarium.exceptions import ProviderError
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk
from cli_modelarium.streaming import run_streaming_comparison


class _RecordingProvider(BaseProvider):
    """Fake provider that returns a preset CompletionResult or raises a preset error.

    When `on_chunk` is supplied, the preset output is delivered through it
    so the orchestrator's state.text gets populated the same way it would
    with a real provider's streaming.
    """

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
        *,
        on_chunk: OnChunk | None = None,
    ) -> CompletionResult:
        self.call_count += 1
        self.calls.append((prompt, model, temperature, system_prompt))
        if isinstance(self._result_or_error, Exception):
            raise self._result_or_error
        result = replace(self._result_or_error, model=model, temperature=temperature)
        if on_chunk is not None and result.output:
            on_chunk(result.output)
        return result


async def test_runs_three_providers_in_parallel() -> None:
    fakes = {
        "openai": _RecordingProvider(
            "openai", CompletionResult(output="oa out", provider="openai")
        ),
        "anthropic": _RecordingProvider(
            "anthropic", CompletionResult(output="ant out", provider="anthropic")
        ),
        "google": _RecordingProvider(
            "google", CompletionResult(output="google out", provider="google")
        ),
    }

    states = await run_streaming_comparison(
        prompt="test prompt",
        models=["gpt-5.5", "claude-opus-4-7", "gemini-3.1-pro-preview"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda name: fakes[name],
        live_display=False,
    )

    assert len(states) == 3
    outputs = {s.text for s in states}
    assert outputs == {"oa out", "ant out", "google out"}
    for f in fakes.values():
        assert f.call_count == 1


async def test_one_failure_does_not_kill_others() -> None:
    fakes = {
        "openai": _RecordingProvider("openai", CompletionResult(output="ok", provider="openai")),
        "anthropic": _RecordingProvider(
            "anthropic", ProviderError("server died", provider="anthropic")
        ),
        "google": _RecordingProvider("google", CompletionResult(output="ok2", provider="google")),
    }

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "claude-opus-4-7", "gemini-3.1-pro-preview"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda name: fakes[name],
        live_display=False,
    )

    assert len(states) == 3
    errors = [s for s in states if s.error]
    successes = [s for s in states if s.error is None]
    assert len(errors) == 1
    assert errors[0].model == "claude-opus-4-7"
    assert errors[0].provider_name == "anthropic"
    assert len(successes) == 2


async def test_one_provider_instance_per_provider_not_per_model() -> None:
    """If multiple models come from one provider, the factory is called only ONCE for it."""
    instantiations: dict[str, int] = {}
    fakes: dict[str, _RecordingProvider] = {}

    def make(name: str) -> _RecordingProvider:
        instantiations[name] = instantiations.get(name, 0) + 1
        provider = _RecordingProvider(name, CompletionResult(output=f"{name} ok", provider=name))
        fakes[name] = provider
        return provider

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "gpt-5.4", "claude-opus-4-7"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=make,
        live_display=False,
    )

    assert len(states) == 3
    assert instantiations == {"openai": 1, "anthropic": 1}
    assert fakes["openai"].call_count == 2  # both openai models routed through one instance
    assert fakes["anthropic"].call_count == 1


async def test_multiple_temperatures_fan_out() -> None:
    """N models x M temperatures = N*M tasks, all using one client per provider."""
    fake = _RecordingProvider("openai", CompletionResult(output="x", provider="openai"))

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "gpt-5.4"],
        temperatures=[0.0, 0.5, 1.0],
        system_prompts=[None],
        provider_factory=lambda _: fake,
        live_display=False,
    )

    assert len(states) == 6  # 2 models x 3 temperatures
    assert fake.call_count == 6
    temps_seen = sorted(t for _, _, t, _ in fake.calls)
    assert temps_seen == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


async def test_system_prompt_threaded_through() -> None:
    fake = _RecordingProvider("openai", CompletionResult(output="x", provider="openai"))

    await run_streaming_comparison(
        prompt="user",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=["you are helpful"],
        provider_factory=lambda _: fake,
        live_display=False,
    )

    assert fake.calls[0][3] == "you are helpful"


async def test_unexpected_exception_caught_as_state_error() -> None:
    """An unexpected non-ModelariumError should become an error StreamState,
    not propagate and kill the whole run.
    """
    fakes = {
        "openai": _RecordingProvider("openai", RuntimeError("kaboom")),
        "anthropic": _RecordingProvider(
            "anthropic", CompletionResult(output="ok", provider="anthropic")
        ),
    }

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "claude-opus-4-7"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda name: fakes[name],
        live_display=False,
    )

    assert len(states) == 2
    errors = [s for s in states if s.error]
    assert len(errors) == 1
    assert "kaboom" in (errors[0].error or "")
