"""Tests for cli_modelarium.streaming.

Covers the four orchestrator behaviors that Phase 4 added:

    * TTFT measured at the orchestrator level (post-semaphore, not from scheduling)
    * Per-provider semaphore caps in-flight calls
    * 429 RateLimitError triggers retry with exponential backoff (respects retry-after)
    * 529 ProviderOverloadedError triggers retry with a longer base backoff
    * One slow stream does not block other streams making progress
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from cli_modelarium.exceptions import (
    ProviderError,
    ProviderOverloadedError,
    RateLimitError,
)
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk
from cli_modelarium.streaming import (
    OVERLOADED_BASE_DELAY_SECONDS,
    RATE_LIMIT_BASE_DELAY_SECONDS,
    StreamingDisplay,
    StreamState,
    run_streaming_comparison,
)

# ===== generic fake provider =====


class _ChunkingProvider(BaseProvider):
    """Fake provider that delivers preset text chunks with optional per-chunk delay."""

    def __init__(
        self,
        name: str,
        chunks: list[str],
        chunk_delay: float = 0.0,
        first_chunk_delay: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.name = name
        self._chunks = chunks
        self._chunk_delay = chunk_delay
        self._first_chunk_delay = first_chunk_delay
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost = cost_usd
        self.concurrent_calls = 0
        self.peak_concurrent_calls = 0
        self.call_order: list[str] = []

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
        self.concurrent_calls += 1
        self.peak_concurrent_calls = max(self.peak_concurrent_calls, self.concurrent_calls)
        self.call_order.append(model)
        try:
            if self._first_chunk_delay:
                await asyncio.sleep(self._first_chunk_delay)
            for c in self._chunks:
                if on_chunk is not None:
                    on_chunk(c)
                if self._chunk_delay:
                    await asyncio.sleep(self._chunk_delay)
            return CompletionResult(
                output="".join(self._chunks),
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cost_usd=self._cost,
                model=model,
                provider=self.name,
                temperature=temperature,
            )
        finally:
            self.concurrent_calls -= 1


class _FailingProvider(BaseProvider):
    """Provider that raises a preset error on its first N calls, then succeeds."""

    def __init__(
        self,
        name: str,
        errors: list[Exception],
        success_result: CompletionResult,
    ) -> None:
        self.name = name
        self._errors = list(errors)
        self._success_result = success_result
        self.attempt_count = 0

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
        self.attempt_count += 1
        if self._errors:
            raise self._errors.pop(0)
        from dataclasses import replace as _replace

        result = _replace(self._success_result, model=model, temperature=temperature)
        if on_chunk is not None and result.output:
            on_chunk(result.output)
        return result


@pytest.fixture
def fake_sleep() -> Any:
    """Sleep function that records calls but does not actually sleep."""
    calls: list[float] = []

    async def _sleep(seconds: float) -> None:
        calls.append(seconds)

    _sleep.calls = calls  # type: ignore[attr-defined]
    return _sleep


# ===== TTFT =====


async def test_ttft_recorded_for_each_state() -> None:
    provider = _ChunkingProvider(
        "openai",
        chunks=["hi", " there"],
        first_chunk_delay=0.01,
        input_tokens=1,
        output_tokens=2,
    )

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
    )

    # TTFT must be captured on the first chunk and be non-negative. We do NOT
    # assert a minimum magnitude: on Windows, asyncio sleep granularity and OS
    # timer coalescing can make a 10ms artificial delay measure as ~0ms, so a
    # magnitude floor is platform-fragile. The "captured on first chunk"
    # contract is covered here (is not None) and by
    # test_ttft_set_on_first_chunk_via_state_callback.
    assert states[0].ttft_ms is not None
    assert states[0].ttft_ms >= 0


async def test_ttft_set_on_first_chunk_via_state_callback() -> None:
    """Direct unit test of StreamState.append_text: TTFT is set on FIRST chunk only."""
    state = StreamState(model="m", provider_name="p", temperature=0.0)
    state.mark_started()

    state.append_text("first")
    first_ttft = state.ttft_ms
    assert first_ttft is not None

    await asyncio.sleep(0.01)
    state.append_text("second")
    # Second chunk must not overwrite the TTFT.
    assert state.ttft_ms == first_ttft
    assert state.text == "firstsecond"


# ===== semaphore =====


async def test_semaphore_caps_concurrent_calls_per_provider() -> None:
    """concurrency=2 means at most 2 in-flight calls per provider at any time."""
    provider = _ChunkingProvider(
        "openai",
        chunks=["x"],
        first_chunk_delay=0.02,
        input_tokens=1,
        output_tokens=1,
    )

    await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4o"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        concurrency=2,
    )

    assert provider.peak_concurrent_calls == 2


async def test_semaphores_are_per_provider_not_global() -> None:
    """openai and anthropic each get their own semaphore so they don't share the budget."""
    openai_p = _ChunkingProvider(
        "openai", chunks=["o"], first_chunk_delay=0.02, input_tokens=1, output_tokens=1
    )
    anthropic_p = _ChunkingProvider(
        "anthropic", chunks=["a"], first_chunk_delay=0.02, input_tokens=1, output_tokens=1
    )

    def factory(name: str) -> BaseProvider:
        return openai_p if name == "openai" else anthropic_p

    await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "gpt-5.4", "claude-opus-4-7", "claude-sonnet-4-6"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=factory,
        live_display=False,
        concurrency=2,
    )

    # Each provider should hit its 2-call cap independently.
    assert openai_p.peak_concurrent_calls == 2
    assert anthropic_p.peak_concurrent_calls == 2


# ===== parallelism =====


async def test_slow_stream_does_not_block_fast_one() -> None:
    """A slow provider must not prevent a fast provider from completing first."""
    fast = _ChunkingProvider(
        "openai",
        chunks=["fast"],
        first_chunk_delay=0.01,
        input_tokens=1,
        output_tokens=1,
    )
    slow = _ChunkingProvider(
        "anthropic",
        chunks=["slow"],
        first_chunk_delay=0.10,
        input_tokens=1,
        output_tokens=1,
    )

    def factory(name: str) -> BaseProvider:
        return fast if name == "openai" else slow

    import time

    start = time.monotonic()
    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5", "claude-opus-4-7"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=factory,
        live_display=False,
    )
    elapsed = time.monotonic() - start

    # If sequential, elapsed would be >= 0.11. Parallel should be ~0.10.
    assert elapsed < 0.15
    assert all(s.status == "complete" for s in states)


# ===== 429 retry =====


async def test_rate_limit_retries_then_succeeds(fake_sleep: Any) -> None:
    success = CompletionResult(output="ok after retry", provider="openai")
    provider = _FailingProvider(
        "openai",
        errors=[RateLimitError("limit", provider="openai", retry_after=None)],
        success_result=success,
    )

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        sleep=fake_sleep,
    )

    assert provider.attempt_count == 2  # 1 failure + 1 success
    assert states[0].error is None
    assert states[0].text == "ok after retry"
    assert states[0].attempts == 1  # retry was attempted
    assert fake_sleep.calls == [RATE_LIMIT_BASE_DELAY_SECONDS]


async def test_rate_limit_respects_retry_after_header(fake_sleep: Any) -> None:
    """When the provider returns a retry-after, sleep for THAT, not the default backoff."""
    success = CompletionResult(output="ok", provider="openai")
    provider = _FailingProvider(
        "openai",
        errors=[RateLimitError("limit", provider="openai", retry_after=15.0)],
        success_result=success,
    )

    await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        sleep=fake_sleep,
    )

    assert fake_sleep.calls == [15.0]


async def test_rate_limit_exponential_backoff_across_attempts(fake_sleep: Any) -> None:
    """Without retry-after headers, the orchestrator doubles the delay each retry."""
    success = CompletionResult(output="finally", provider="openai")
    provider = _FailingProvider(
        "openai",
        errors=[
            RateLimitError("limit", provider="openai", retry_after=None),
            RateLimitError("limit", provider="openai", retry_after=None),
            RateLimitError("limit", provider="openai", retry_after=None),
        ],
        success_result=success,
    )

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        max_retries=3,
        sleep=fake_sleep,
    )

    # 3 retries -> 3 sleeps: 1, 2, 4 (exponential doubling)
    assert fake_sleep.calls == [
        RATE_LIMIT_BASE_DELAY_SECONDS,
        RATE_LIMIT_BASE_DELAY_SECONDS * 2,
        RATE_LIMIT_BASE_DELAY_SECONDS * 4,
    ]
    assert states[0].error is None


async def test_rate_limit_gives_up_after_max_retries(fake_sleep: Any) -> None:
    success = CompletionResult(output="never", provider="openai")
    provider = _FailingProvider(
        "openai",
        errors=[RateLimitError("limit", provider="openai") for _ in range(10)],
        success_result=success,
    )

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        max_retries=3,
        sleep=fake_sleep,
    )

    # max_retries=3 means 4 attempts total (1 initial + 3 retries), then give up.
    assert provider.attempt_count == 4
    assert states[0].error is not None
    assert states[0].status == "error"


# ===== 529 backoff =====


async def test_529_overloaded_uses_longer_base_delay_than_429(fake_sleep: Any) -> None:
    success = CompletionResult(output="finally", provider="anthropic")
    provider = _FailingProvider(
        "anthropic",
        errors=[ProviderOverloadedError("overloaded", provider="anthropic")],
        success_result=success,
    )

    await run_streaming_comparison(
        prompt="p",
        models=["claude-opus-4-7"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        sleep=fake_sleep,
    )

    # 529's base delay is configured to be longer than 429's.
    assert OVERLOADED_BASE_DELAY_SECONDS > RATE_LIMIT_BASE_DELAY_SECONDS
    assert fake_sleep.calls == [OVERLOADED_BASE_DELAY_SECONDS]


async def test_529_backoff_doubles_across_retries(fake_sleep: Any) -> None:
    success = CompletionResult(output="finally", provider="anthropic")
    provider = _FailingProvider(
        "anthropic",
        errors=[
            ProviderOverloadedError("overloaded", provider="anthropic"),
            ProviderOverloadedError("overloaded", provider="anthropic"),
        ],
        success_result=success,
    )

    await run_streaming_comparison(
        prompt="p",
        models=["claude-opus-4-7"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        max_retries=3,
        sleep=fake_sleep,
    )

    assert fake_sleep.calls == [
        OVERLOADED_BASE_DELAY_SECONDS,
        OVERLOADED_BASE_DELAY_SECONDS * 2,
    ]


# ===== non-retryable errors =====


async def test_non_retryable_error_propagates_immediately(fake_sleep: Any) -> None:
    """A ProviderError (not 429/529) must not trigger a retry."""
    provider = _FailingProvider(
        "openai",
        errors=[ProviderError("auth failed", provider="openai")],
        success_result=CompletionResult(output="ok"),
    )

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        sleep=fake_sleep,
    )

    assert provider.attempt_count == 1  # no retry
    assert fake_sleep.calls == []
    assert states[0].status == "error"


# ===== StreamingDisplay rendering smoke test =====


def test_streaming_display_renders_panels() -> None:
    states = [
        StreamState(
            model="gpt-5.5", provider_name="openai", temperature=0.0, status="streaming", text="hi"
        ),
        StreamState(
            model="claude-opus-4-7",
            provider_name="anthropic",
            temperature=0.0,
            status="complete",
            text="done",
        ),
        StreamState(
            model="grok-4.3", provider_name="xai", temperature=0.0, status="error", error="boom"
        ),
        StreamState(
            model="gemini-3.1-pro",
            provider_name="google",
            temperature=0.0,
            status="retrying",
            retry_message="rate limited",
        ),
        StreamState(
            model="mistral-large-latest", provider_name="mistral", temperature=0.0, status="pending"
        ),
    ]
    display = StreamingDisplay(states)

    group = display.__rich__()
    # Should produce one panel per state.
    assert len(list(group.renderables)) == 5


def test_complete_panel_shows_free_for_local_models() -> None:
    """Local models must display 'Free' in the complete panel, not '$0.000000'."""
    from rich.console import Console

    local = StreamState(
        model="local/llama-3.3-70b",
        provider_name="local",
        temperature=0.0,
        status="complete",
        text="done",
        ttft_ms=120.0,
        cost_usd=0.0,
    )
    cloud = StreamState(
        model="gpt-5.5",
        provider_name="openai",
        temperature=0.0,
        status="complete",
        text="done",
        ttft_ms=120.0,
        cost_usd=0.001234,
    )

    console = Console(record=True, width=120, color_system=None)
    console.print(StreamingDisplay([local, cloud]).__rich__())
    output = console.export_text()

    assert "Free" in output
    assert "$0.000000" not in output
    assert "$0.001234" in output


# ===== redaction in error state =====


async def test_provider_error_message_redacted_in_state(fake_sleep: Any) -> None:
    provider = _FailingProvider(
        "openai",
        errors=[ProviderError("leaked sk-proj-abc1234567890XYZdef", provider="openai")],
        success_result=CompletionResult(output="ok"),
    )

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        sleep=fake_sleep,
    )

    assert states[0].error is not None
    assert "abc1234567890XYZ" not in states[0].error
    assert "REDACTED" in states[0].error


# ===== partial text cleared on retry =====


async def test_partial_text_cleared_when_retrying(fake_sleep: Any) -> None:
    """If a stream starts emitting text but then rate-limits, the partial text
    must be cleared before retry so the user sees the retry's output cleanly.
    """

    class _RateLimitAfterChunks(BaseProvider):
        def __init__(self) -> None:
            self.name = "openai"
            self.attempts = 0

        async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
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
            self.attempts += 1
            if self.attempts == 1:
                if on_chunk:
                    on_chunk("partial-")
                    on_chunk("output")
                raise RateLimitError("limited", provider="openai", retry_after=None)
            if on_chunk:
                on_chunk("clean retry")
            return CompletionResult(output="clean retry", provider="openai")

    provider = _RateLimitAfterChunks()

    states = await run_streaming_comparison(
        prompt="p",
        models=["gpt-5.5"],
        temperatures=[0.0],
        system_prompts=[None],
        provider_factory=lambda _: provider,
        live_display=False,
        sleep=fake_sleep,
    )

    assert states[0].text == "clean retry"
    assert "partial-output" not in states[0].text
