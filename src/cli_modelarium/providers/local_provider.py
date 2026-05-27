"""Local model provider for Ollama, LM Studio, vLLM, llama.cpp, etc.

All major local LLM tools expose an OpenAI-compatible REST API. This
provider inherits from `OpenAIProvider` and overrides only:

    * the base URL (default Ollama at localhost:11434)
    * `_transform_model()` to strip the `local/` prefix before sending to
      the SDK (`local/llama-3.3-70b` -> `llama-3.3-70b` on the wire)
    * a localhost-only URL guard - this is a SECURITY feature, not a
      convenience. The CLI treats "local" as meaning local; a typo'd remote
      URL would silently send local prompts to a third party. Refused.

Streaming, system prompts, and on_chunk callbacks all work without further
overrides via the inherited OpenAIProvider behaviour.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from cli_modelarium.exceptions import LocalURLError
from cli_modelarium.providers.openai_provider import OpenAIProvider


class LocalProvider(OpenAIProvider):
    """Provider for any OpenAI-compatible LLM server running on localhost."""

    name: str = "local"
    DEFAULT_URL: str = "http://localhost:11434/v1"
    LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
    DUMMY_API_KEY: str = "not-required"
    DISCOVERY_TIMEOUT_SECONDS: float = 5.0

    def __init__(self, base_url: str | None = None) -> None:
        url = base_url or self.DEFAULT_URL
        validated = self._validate_local_url(url)
        super().__init__(api_key=self.DUMMY_API_KEY, base_url=validated)

    def _transform_model(self, model: str) -> str:
        """Strip the `local/` routing prefix before sending to the SDK."""
        return model.removeprefix("local/")

    @classmethod
    def _validate_local_url(cls, url: str) -> str:
        """Return `url` if it parses as an http(s) URL targeting a localhost host.

        Raises LocalURLError otherwise with an actionable message. This is the
        security boundary - a misconfigured URL must never silently leak local
        data to a remote endpoint.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise LocalURLError(
                f"Local URL must use http or https (got scheme {parsed.scheme!r} in {url!r}).\n"
                f"  Example: http://localhost:11434/v1"
            )
        if parsed.hostname not in cls.LOCAL_HOSTS:
            allowed = ", ".join(sorted(cls.LOCAL_HOSTS))
            raise LocalURLError(
                f"Local URL must point to localhost "
                f"(got hostname {parsed.hostname!r} in {url!r}).\n"
                f"  Allowed hostnames: {allowed}\n"
                f"  For remote OpenAI-compatible endpoints, use a real "
                f"provider integration,\n"
                f"  not the local provider - this guard prevents accidental "
                f"data leakage."
            )
        return url

    @classmethod
    async def discover_models(cls, base_url: str | None = None) -> list[dict[str, Any]]:
        """Query the local server's `/models` endpoint and return the entries.

        Returns the `data` array from an OpenAI-compatible `/models` response:
            [{"id": "...", "created": ..., "owned_by": "...", ...}, ...]

        URL validation runs first - a non-localhost URL raises LocalURLError
        before any network I/O happens. Network errors (connection refused,
        timeout, malformed response) propagate so callers can render a friendly
        "no local server" message.
        """
        url = cls._validate_local_url(base_url or cls.DEFAULT_URL)
        models_url = url.rstrip("/") + "/models"
        async with httpx.AsyncClient(timeout=cls.DISCOVERY_TIMEOUT_SECONDS) as client:
            response = await client.get(models_url)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        return data
