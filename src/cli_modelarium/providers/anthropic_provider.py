"""Anthropic provider implementation.

Two things make Anthropic distinct from OpenAI-style providers:

    1. The `system` prompt is a TOP-LEVEL parameter on `messages.create()`,
       NOT a message with `role: "system"` inside the messages array.
    2. `max_tokens` is REQUIRED on every call. We default to 4096 if the
       caller doesn't specify one.

Streaming uses the `messages.stream()` async context manager. The final
usage payload (input/output/cache-read tokens) is read after iteration
via `stream.get_final_message()`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from cli_modelarium.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderOverloadedError,
    RateLimitError,
)
from cli_modelarium.pricing import calculate_cost
from cli_modelarium.providers._utils import extract_retry_after
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk
from cli_modelarium.security import redact_secrets

DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(BaseProvider):
    """Provider using the official Anthropic Python SDK."""

    name: str = "anthropic"

    def __init__(self, api_key: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)

    def _build_kwargs(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        return kwargs

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(prompt, model, temperature, system_prompt)
        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIError as e:
            self._reraise(e)

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
        *,
        on_chunk: OnChunk | None = None,
    ) -> CompletionResult:
        kwargs = self._build_kwargs(prompt, model, temperature, system_prompt)

        start = time.monotonic()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - start) * 1000
                    if on_chunk is not None:
                        on_chunk(text)
                    chunks.append(text)
                final = await stream.get_final_message()
                usage = final.usage
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        except anthropic.APIError as e:
            self._reraise(e)

        latency_ms = (time.monotonic() - start) * 1000

        try:
            cost = calculate_cost(model, input_tokens, output_tokens, cached_tokens)
        except Exception:
            cost = 0.0

        return CompletionResult(
            output="".join(chunks),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            model=model,
            provider=self.name,
            temperature=temperature,
        )

    def _reraise(self, error: anthropic.APIError) -> None:
        """Translate an Anthropic SDK error into a redacted Cli Modelarium exception."""
        message = redact_secrets(str(error))

        if isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError(message, provider=self.name) from None
        if isinstance(error, anthropic.RateLimitError):
            retry_after = extract_retry_after(error)
            raise RateLimitError(message, provider=self.name, retry_after=retry_after) from None
        # Anthropic returns 529 when their service is overloaded - distinct from rate limits.
        if (
            isinstance(error, anthropic.APIStatusError)
            and getattr(error, "status_code", None) == 529
        ):
            raise ProviderOverloadedError(message, provider=self.name) from None
        raise ProviderError(message, provider=self.name) from None
