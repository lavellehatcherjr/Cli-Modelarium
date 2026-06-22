"""OpenAI provider implementation.

Designed to be subclassed by the OpenAI-compatible providers (xAI, DeepSeek,
Groq, OpenRouter, Local) by overriding `name`, the constructor (to pass a
different base_url), and optionally `_transform_model()`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

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


class OpenAIProvider(BaseProvider):
    """Provider using the official OpenAI Python SDK."""

    name: str = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if default_headers is not None:
            client_kwargs["default_headers"] = default_headers
        self.client = AsyncOpenAI(**client_kwargs)

    def _transform_model(self, model: str) -> str:
        """Subclass hook for providers that need to rewrite the model ID (e.g. LocalProvider)."""
        return model

    def _extra_create_kwargs(self) -> dict:
        """Subclass hook for extra chat.completions.create() kwargs.

        Defaults to an empty dict (no-op) so every existing OpenAI-compatible
        provider is unaffected. DashScope overrides this to send
        `enable_thinking=False` via `extra_body`.
        """
        return {}

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
        actual_model = self._transform_model(model)
        try:
            response = await self.client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                stream=True,
                **self._extra_create_kwargs(),
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.APIError as e:
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
        """Run a completion via streaming with usage included.

        Streaming with `stream_options={"include_usage": True}` gives us both
        token-by-token output (for TTFT) and final usage numbers in one call.

        If `on_chunk` is provided it is called with each text chunk as it
        arrives, so the streaming orchestrator can surface output live.
        """
        messages = self._build_messages(prompt, system_prompt)
        actual_model = self._transform_model(model)

        start = time.monotonic()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        try:
            response = await self.client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                **self._extra_create_kwargs(),
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - start) * 1000
                    if on_chunk is not None:
                        on_chunk(text)
                    chunks.append(text)
                if getattr(chunk, "usage", None) is not None:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    if details is not None:
                        cached_tokens = getattr(details, "cached_tokens", 0) or 0
        except openai.APIError as e:
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

    def _reraise(self, error: openai.APIError) -> None:
        """Translate an OpenAI SDK error into a redacted Cli Modelarium exception."""
        message = redact_secrets(str(error))

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(message, provider=self.name) from None
        if isinstance(error, openai.RateLimitError):
            retry_after = extract_retry_after(error)
            raise RateLimitError(message, provider=self.name, retry_after=retry_after) from None
        if isinstance(error, openai.APIStatusError) and getattr(error, "status_code", None) == 529:
            raise ProviderOverloadedError(message, provider=self.name) from None
        raise ProviderError(message, provider=self.name) from None
