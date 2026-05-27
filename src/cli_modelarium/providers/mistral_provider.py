"""Mistral provider using the official mistralai SDK.

Notes on the SDK layout in mistralai 2.4.7:

    The spec's `from mistralai import Mistral` import path does not exist
    in 2.4.7 - the top-level package is now a namespace package with
    subpackages for each cloud host (azure, gcp, generic). The canonical
    client lives at `mistralai.client.Mistral`.

    Stream events wrap the chunk in a `CompletionEvent` whose `.data`
    attribute is the OpenAI-style `CompletionChunk` with the usual
    `choices[0].delta.content` shape.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError

from cli_modelarium.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from cli_modelarium.pricing import calculate_cost
from cli_modelarium.providers._utils import extract_retry_after
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk
from cli_modelarium.security import redact_secrets


class MistralProvider(BaseProvider):
    """Provider using the official Mistral Python SDK."""

    name: str = "mistral"

    def __init__(self, api_key: str) -> None:
        self.client = Mistral(api_key=api_key)

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(prompt, system_prompt)
        try:
            response_stream = await self.client.chat.stream_async(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            async for event in response_stream:
                content = _extract_chunk_text(event)
                if content:
                    yield content
        except SDKError as e:
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
        messages = self._build_messages(prompt, system_prompt)

        start = time.monotonic()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0

        try:
            response_stream = await self.client.chat.stream_async(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            async for event in response_stream:
                content = _extract_chunk_text(event)
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - start) * 1000
                    if on_chunk is not None:
                        on_chunk(content)
                    chunks.append(content)
                usage = _extract_chunk_usage(event)
                if usage is not None:
                    input_tokens, output_tokens = usage
        except SDKError as e:
            self._reraise(e)

        latency_ms = (time.monotonic() - start) * 1000

        try:
            cost = calculate_cost(model, input_tokens, output_tokens)
        except Exception:
            cost = 0.0

        return CompletionResult(
            output="".join(chunks),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            model=model,
            provider=self.name,
            temperature=temperature,
        )

    def _reraise(self, error: SDKError) -> None:
        """Translate a Mistral SDK error into a redacted Cli Modelarium exception."""
        message = redact_secrets(str(error))
        status_code = _extract_status_code(error)

        if status_code in (401, 403):
            raise AuthenticationError(message, provider=self.name) from None
        if status_code == 429:
            retry_after = extract_retry_after(error)
            raise RateLimitError(message, provider=self.name, retry_after=retry_after) from None
        raise ProviderError(message, provider=self.name) from None


def _extract_chunk_text(event: Any) -> str | None:
    """Pull `data.choices[0].delta.content` out of a streamed CompletionEvent."""
    data = getattr(event, "data", None)
    if data is None:
        return None
    choices = getattr(data, "choices", None)
    if not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None
    return getattr(delta, "content", None)


def _extract_chunk_usage(event: Any) -> tuple[int, int] | None:
    """Pull (input_tokens, output_tokens) out of a streamed CompletionEvent's usage payload."""
    data = getattr(event, "data", None)
    if data is None:
        return None
    usage = getattr(data, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return input_tokens, output_tokens


def _extract_status_code(error: SDKError) -> int | None:
    """Pull the HTTP status code from an SDKError's wrapped httpx response."""
    response = getattr(error, "raw_response", None)
    if response is None:
        return None
    return getattr(response, "status_code", None)
