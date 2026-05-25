"""OpenRouter provider - OpenAI-compatible meta-provider with custom headers.

OpenRouter aggregates models from many vendors behind one OpenAI-compatible
API. They expect two extra headers identifying the calling application:

    HTTP-Referer  - URL of the calling app (for analytics/leaderboard)
    X-Title       - human-readable app name

We pass these as `default_headers` on the AsyncOpenAI client so they go on
every request without per-call wiring.
"""
from __future__ import annotations

from cli_modelarium.providers.openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter models via the OpenAI-compatible endpoint at openrouter.ai."""

    name: str = "openrouter"
    BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_HEADERS: dict[str, str] = {
        "HTTP-Referer": "https://github.com/lavellehatcherjr/cli-modelarium",
        "X-Title": "Cli Modelarium",
    }

    def __init__(self, api_key: str) -> None:
        super().__init__(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers=self.DEFAULT_HEADERS,
        )
