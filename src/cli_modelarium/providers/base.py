"""Provider abstraction: every concrete provider implements this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
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
    ) -> CompletionResult:
        """Run a non-streaming completion.

        Default implementation collects chunks from `stream()` but cannot
        capture token usage. Subclasses should override to read usage data
        from the final API response.
        """
        chunks: list[str] = []
        async for chunk in self.stream(prompt, model, temperature, system_prompt):
            chunks.append(chunk)
        return CompletionResult(
            output="".join(chunks),
            model=model,
            provider=self.name,
            temperature=temperature,
        )
