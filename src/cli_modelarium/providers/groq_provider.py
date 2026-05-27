"""Groq provider - uses the OpenAI SDK with a different base URL."""

from __future__ import annotations

from cli_modelarium.providers.openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq models via the OpenAI-compatible endpoint at api.groq.com."""

    name: str = "groq"
    BASE_URL: str = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, base_url=self.BASE_URL)
