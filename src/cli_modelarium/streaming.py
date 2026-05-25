"""Parallel streaming orchestration for the compare command.

This module owns the per-call lifecycle for a comparison run:

    * one semaphore per provider (rate-limit hygiene)
    * 429 retry with exponential backoff (respects `retry-after`)
    * 529 (Anthropic overloaded) retry with a longer backoff than 429
    * live token-by-token display via Rich Live
    * accurate TTFT measurement (timestamps captured at the orchestrator
      level, after the semaphore has been acquired - so "queued behind other
      tasks" does not pollute the metric)

`run_streaming_comparison()` is the only public entry point. cli.py routes
both `--stream` and `--no-stream` through it - the only difference is
whether the Live display is enabled (`live_display=False` for --no-stream).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from cli_modelarium.exceptions import (
    ModelariumError,
    ProviderOverloadedError,
    RateLimitError,
)
from cli_modelarium.models_registry import get_provider_for_model
from cli_modelarium.pricing import is_local_model
from cli_modelarium.providers.base import BaseProvider, CompletionResult
from cli_modelarium.security import redact_secrets

DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_RETRIES = 3

# Backoff schedule for 429 rate limits. Used when the response has no
# retry-after header. When the header is present, we honor it verbatim and
# this schedule still advances for the next attempt.
RATE_LIMIT_BASE_DELAY_SECONDS = 1.0

# Backoff schedule for Anthropic 529 (service overloaded). Retrying
# immediately does not help while the upstream is over-capacity, so the
# base delay is longer than for 429.
OVERLOADED_BASE_DELAY_SECONDS = 5.0


# Possible state.status values:
#   pending     - task created, not yet entered
#   waiting     - semaphore acquired, request in flight, no chunks yet
#   streaming   - at least one chunk has arrived
#   retrying    - 429/529 hit; sleeping before retry
#   complete    - call finished successfully
#   error       - call failed (after all retries)
Status = str


@dataclass
class StreamState:
    """Per-task state surfaced live in the streaming display.

    Mutated in place by `_run_one()` so the Rich Live renderable can read
    the current state on every refresh.
    """

    model: str
    provider_name: str
    temperature: float
    status: Status = "pending"
    text: str = ""
    ttft_ms: float | None = None
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    retry_message: str | None = None
    attempts: int = 0  # number of retries attempted (0 == first call only)
    _start: float | None = field(default=None, repr=False)

    def mark_started(self) -> None:
        self._start = time.monotonic()
        self.status = "waiting"

    def append_text(self, chunk: str) -> None:
        """Callback hook passed as `on_chunk=` to provider.complete()."""
        if self.ttft_ms is None and self._start is not None:
            self.ttft_ms = (time.monotonic() - self._start) * 1000
        self.status = "streaming"
        self.text += chunk

    def mark_complete(self, result: CompletionResult) -> None:
        self.status = "complete"
        self.input_tokens = result.input_tokens
        self.output_tokens = result.output_tokens
        self.cached_tokens = result.cached_tokens
        self.cost_usd = result.cost_usd
        self.latency_ms = result.latency_ms
        # Prefer the provider's authoritative TTFT (measured around the
        # actual SDK call) when our orchestrator-level estimate is missing.
        if self.ttft_ms is None and result.ttft_ms is not None:
            self.ttft_ms = result.ttft_ms

    def mark_error(self, message: str) -> None:
        self.status = "error"
        self.error = redact_secrets(message)

    def mark_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        self.status = "retrying"
        self.attempts = attempt + 1
        self.retry_message = f"{reason}, retry in {delay_s:.1f}s (attempt {attempt + 1})"


class StreamingDisplay:
    """Rich renderable. `__rich__` is re-evaluated on every Live refresh."""

    def __init__(self, states: list[StreamState]) -> None:
        self.states = states

    def __rich__(self) -> Group:
        return Group(*[self._panel(s) for s in self.states])

    @staticmethod
    def _panel(state: StreamState) -> Panel:
        if state.status == "pending":
            status_text = "[dim]queued[/dim]"
        elif state.status == "waiting":
            status_text = "[cyan]connecting...[/cyan]"
        elif state.status == "streaming":
            ttft_text = (
                f"TTFT {state.ttft_ms / 1000:.2f}s" if state.ttft_ms is not None else ""
            )
            status_text = f"[cyan]streaming[/cyan]  [dim]{ttft_text}[/dim]"
        elif state.status == "retrying":
            status_text = f"[yellow]{state.retry_message or 'retrying'}[/yellow]"
        elif state.status == "complete":
            parts: list[str] = []
            if state.ttft_ms is not None:
                parts.append(f"TTFT {state.ttft_ms / 1000:.2f}s")
            # Local models show "Free" - a visual distinction from $0.000000.
            if is_local_model(state.model):
                parts.append("Free")
            else:
                parts.append(f"${state.cost_usd:.6f}")
            status_text = f"[green]done[/green]  [dim]{'  '.join(parts)}[/dim]"
        elif state.status == "error":
            status_text = "[red]error[/red]"
        else:
            status_text = state.status

        title = f"[bold]{state.model}[/bold] @ {state.temperature:.1f}   {status_text}"

        if state.error:
            body = f"[red]{state.error}[/red]"
        elif state.status == "streaming":
            body = state.text + "[cyan]▌[/cyan]"
        elif state.status in ("pending", "waiting"):
            body = "[dim]...[/dim]"
        else:
            body = state.text or "[dim](no output)[/dim]"

        border_style = {
            "complete": "green",
            "error": "red",
            "retrying": "yellow",
            "streaming": "cyan",
        }.get(state.status, "dim")

        return Panel(body, title=title, title_align="left", border_style=border_style)


async def _call_with_retry(
    provider: BaseProvider,
    state: StreamState,
    prompt: str,
    system_prompt: str | None,
    max_retries: int,
    sleep: Callable[[float], "asyncio.Future[None]"] = asyncio.sleep,
) -> CompletionResult:
    """Run `provider.complete()` with 429/529 retry, updating state.

    `sleep` is parameterised so tests can monkeypatch it to make backoff
    instantaneous.
    """
    rate_limit_delay = RATE_LIMIT_BASE_DELAY_SECONDS
    overloaded_delay = OVERLOADED_BASE_DELAY_SECONDS

    for attempt in range(max_retries + 1):
        try:
            return await provider.complete(
                prompt=prompt,
                model=state.model,
                temperature=state.temperature,
                system_prompt=system_prompt,
                on_chunk=state.append_text,
            )
        except RateLimitError as e:
            if attempt >= max_retries:
                raise
            wait = e.retry_after if e.retry_after is not None else rate_limit_delay
            state.mark_retry(attempt, wait, reason="rate limited")
            # Reset partial text so retries don't show duplicated output.
            state.text = ""
            state.ttft_ms = None
            await sleep(wait)
            rate_limit_delay *= 2
        except ProviderOverloadedError:
            if attempt >= max_retries:
                raise
            wait = overloaded_delay
            state.mark_retry(attempt, wait, reason="provider overloaded")
            state.text = ""
            state.ttft_ms = None
            await sleep(wait)
            overloaded_delay *= 2

    # Loop exits via `return` or `raise`, never falls through.
    raise RuntimeError("unreachable: retry loop did not return or raise")


async def _run_one(
    provider: BaseProvider,
    state: StreamState,
    prompt: str,
    system_prompt: str | None,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    sleep: Callable[[float], "asyncio.Future[None]"] = asyncio.sleep,
) -> None:
    """Run a single task to completion: semaphore -> retry loop -> state update."""
    async with semaphore:
        state.mark_started()
        try:
            result = await _call_with_retry(
                provider=provider,
                state=state,
                prompt=prompt,
                system_prompt=system_prompt,
                max_retries=max_retries,
                sleep=sleep,
            )
            state.mark_complete(result)
        except ModelariumError as e:
            state.mark_error(str(e))
        except Exception as e:  # noqa: BLE001 - converted to error state, not re-raised
            state.mark_error(f"unexpected: {e}")


async def run_streaming_comparison(
    *,
    prompt: str,
    models: list[str],
    temperatures: list[float],
    system_prompt: str | None,
    provider_factory: Callable[[str], BaseProvider],
    console: Console | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    live_display: bool = True,
    sleep: Callable[[float], "asyncio.Future[None]"] = asyncio.sleep,
) -> list[StreamState]:
    """Run every (model x temperature) call in parallel.

    Returns the list of `StreamState` in the same order as the cartesian
    product: for each model in order, all temperatures in order.

    With `live_display=True` (the default) wraps the work in a Rich `Live`
    using `transient=True`, so the streaming panels disappear when the call
    completes and the caller can render a clean final summary.

    `provider_factory(name)` is called once per unique provider name to
    obtain the SDK client instance. Reused across all calls for that
    provider.
    """
    if console is None:
        console = Console()

    states: list[StreamState] = []
    for model in models:
        provider_name = get_provider_for_model(model)
        for temperature in temperatures:
            states.append(
                StreamState(
                    model=model, provider_name=provider_name, temperature=temperature
                )
            )

    # One instance per provider, one semaphore per provider.
    provider_names = sorted({s.provider_name for s in states})
    instances: dict[str, BaseProvider] = {
        name: provider_factory(name) for name in provider_names
    }
    semaphores: dict[str, asyncio.Semaphore] = {
        name: asyncio.Semaphore(concurrency) for name in provider_names
    }

    tasks_coro = asyncio.gather(
        *[
            _run_one(
                provider=instances[s.provider_name],
                state=s,
                prompt=prompt,
                system_prompt=system_prompt,
                semaphore=semaphores[s.provider_name],
                max_retries=max_retries,
                sleep=sleep,
            )
            for s in states
        ]
    )

    if live_display:
        display = StreamingDisplay(states)
        # transient=True clears the streaming panels on exit so the final
        # summary table the caller prints isn't shoved below leftover panels.
        with Live(display, console=console, refresh_per_second=10, transient=True):
            await tasks_coro
    else:
        await tasks_coro

    return states
