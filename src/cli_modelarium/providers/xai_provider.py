"""xAI (Grok) provider - uses the OpenAI SDK with a different base URL."""

from __future__ import annotations

from cli_modelarium.providers.openai_provider import OpenAIProvider


class XAIProvider(OpenAIProvider):
    """xAI Grok models via the OpenAI-compatible endpoint at api.x.ai."""

    name: str = "xai"
    BASE_URL: str = "https://api.x.ai/v1"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, base_url=self.BASE_URL)
