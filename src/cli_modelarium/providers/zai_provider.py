"""Z.AI / GLM provider - Zhipu AI via the OpenAI-compatible endpoint.

Uses the OpenAI SDK pointed at Z.AI's international (overseas) endpoint, which
speaks the standard OpenAI `/chat/completions` protocol, so GLM models stream
and report usage like any other OpenAI-compatible provider - no provider-specific
request params needed.
"""

from __future__ import annotations

from cli_modelarium.providers.openai_provider import OpenAIProvider


class ZAIProvider(OpenAIProvider):
    """GLM models via Z.AI's OpenAI-compatible overseas endpoint."""

    name: str = "zai"
    BASE_URL: str = "https://api.z.ai/api/paas/v4/"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, base_url=self.BASE_URL)
