"""Tests for the runs parameter on run_streaming_comparison."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk
from cli_modelarium.streaming import (
    AUTO_COLLAPSE_TASK_THRESHOLD,
    run_streaming_comparison,
)


class _CountingProvider(BaseProvider):
    """Counts how many times `complete()` was called per model+temp."""

    def __init__(self, name: str = "openai") -> None:
        self.name = name
        self.call_count: int = 0
        self.concurrent_calls = 0
        self.peak_concurrent_calls = 0

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""
        raise NotImplementedError

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
        self.concurrent_calls += 1
        self.peak_concurrent_calls = max(self.peak_concurrent_calls, self.concurrent_calls)
        try:
            await asyncio.sleep(0)
            if on_chunk is not None:
                on_chunk("ok")
            return CompletionResult(
                output="ok",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0001,
                latency_ms=1.0,
                model=model,
                provider=self.name,
                temperature=temperature,
            )
        finally:
            self.concurrent_calls -= 1


class _FailingProvider(BaseProvider):
    """Half-success: errors on odd run_indices."""

    def __init__(self, name: str = "openai") -> None:
        self.name = name
        self.call_count = 0

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""
        raise NotImplementedError

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
        if self.call_count % 2 == 0:
            raise RuntimeError("boom")
        if on_chunk is not None:
            on_chunk("ok")
        return CompletionResult(
            output="ok",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0001,
            latency_ms=1.0,
            model=model,
            provider=self.name,
            temperature=temperature,
        )


def test_runs_constructs_correct_task_count() -> None:
    """N x M x T x S task count matches the runs parameter."""
    provider = _CountingProvider()

    async def go() -> None:
        states = await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0, 0.7],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            live_display=False,
            runs=5,
        )
        assert len(states) == 10  # 1 model * 2 temps * 1 sp * 5 runs
        assert provider.call_count == 10

    asyncio.run(go())


def test_runs_assigns_unique_run_indices() -> None:
    """Within each cell, run_index goes 0..N-1."""
    provider = _CountingProvider()

    async def go() -> None:
        states = await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            live_display=False,
            runs=5,
        )
        assert [s.run_index for s in states] == [0, 1, 2, 3, 4]

    asyncio.run(go())


def test_runs_default_one_preserves_existing_behavior() -> None:
    """Without `runs=`, behavior is identical to v0.1.0."""
    provider = _CountingProvider()

    async def go() -> None:
        states = await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            live_display=False,
        )
        assert len(states) == 1
        assert states[0].run_index == 0

    asyncio.run(go())


def test_runs_preserves_per_provider_semaphore() -> None:
    """With runs > 1, concurrency cap still kicks in."""
    provider = _CountingProvider()

    async def go() -> None:
        await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            concurrency=2,
            live_display=False,
            runs=10,
        )
        assert provider.peak_concurrent_calls <= 2

    asyncio.run(go())


def test_runs_handles_partial_failures() -> None:
    """When some runs fail, the orchestrator returns all states with errors recorded."""
    provider = _FailingProvider()

    async def go() -> None:
        states = await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            live_display=False,
            runs=4,
        )
        assert len(states) == 4
        # 2 of the 4 calls fail (every even call).
        errors = [s for s in states if s.error is not None]
        successes = [s for s in states if s.error is None]
        assert len(errors) == 2
        assert len(successes) == 2

    asyncio.run(go())


def test_auto_collapse_above_threshold() -> None:
    """When task count exceeds threshold, live_display is silently disabled."""
    provider = _CountingProvider()

    async def go() -> None:
        states = await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            live_display=True,
            runs=AUTO_COLLAPSE_TASK_THRESHOLD + 1,
        )
        # Tasks still ran; the auto-collapse only affects display, not work.
        assert len(states) == AUTO_COLLAPSE_TASK_THRESHOLD + 1

    asyncio.run(go())


def test_show_all_runs_overrides_auto_collapse() -> None:
    """show_all_runs=True keeps the Live display on even past the threshold."""
    provider = _CountingProvider()

    async def go() -> None:
        # We pass show_all_runs=True; the threshold check should not flip
        # live_display off. We can't easily assert the Live being on without
        # mocking, but we CAN verify the run completes with the correct
        # number of tasks.
        states = await run_streaming_comparison(
            prompt="q",
            models=["gpt-5.5"],
            temperatures=[0.0],
            system_prompts=[None],
            provider_factory=lambda _name: provider,
            live_display=True,
            runs=AUTO_COLLAPSE_TASK_THRESHOLD + 1,
            show_all_runs=True,
        )
        assert len(states) == AUTO_COLLAPSE_TASK_THRESHOLD + 1

    asyncio.run(go())
