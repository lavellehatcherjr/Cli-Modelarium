"""Provider abstraction: every concrete provider implements this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass


@dataclass
class CompletionResult:
    """The full result of a single completion call.

    All fields default-initialized so partial results (on error, mid-stream
    failure, or sync collection) remain valid dataclasses.
    """

    output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    model: str = ""
    provider: str = ""
    temperature: float = 0.0
    error: str | None = None


# Type alias for the per-chunk callback that the streaming orchestrator passes in.
OnChunk = Callable[[str], None]


class BaseProvider(ABC):
    """Abstract base class for every provider integration.

    Subclasses implement `stream()` as an async generator that yields token
    chunks. The default `complete()` collects those chunks but cannot capture
    usage data - subclasses should override `complete()` to read the final
    usage payload from the provider's response.
    """

    name: str = "base"

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield response token chunks as they arrive from the provider."""
        ...

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
        *,
        on_chunk: OnChunk | None = None,
    ) -> CompletionResult:
        """Run a non-streaming completion.

        Default implementation collects chunks from `stream()` but cannot
        capture token usage. Subclasses should override to read usage data
        from the final API response.

        If `on_chunk` is provided it is invoked with each text chunk before
        the chunk is appended to the accumulator. This lets the streaming
        orchestrator surface partial output live without changing the public
        return type.
        """
        chunks: list[str] = []
        async for chunk in self.stream(prompt, model, temperature, system_prompt):
            if on_chunk is not None:
                on_chunk(chunk)
            chunks.append(chunk)
        return CompletionResult(
            output="".join(chunks),
            model=model,
            provider=self.name,
            temperature=temperature,
        )
