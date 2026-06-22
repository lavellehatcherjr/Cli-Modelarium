"""DashScope provider - Alibaba Model Studio via the OpenAI-compatible endpoint.

Uses the OpenAI SDK pointed at the International/Singapore DashScope endpoint.
Qwen3.x models default to thinking ON, which bills extra output tokens against
the non-thinking rates we store; we send `enable_thinking=False` via the
base-class `_extra_create_kwargs()` hook so costs reflect the listed rates.
"""

from __future__ import annotations

from cli_modelarium.providers.openai_provider import OpenAIProvider


class DashScopeProvider(OpenAIProvider):
    """Qwen models via Alibaba's OpenAI-compatible DashScope endpoint."""

    name: str = "dashscope"
    BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, base_url=self.BASE_URL)

    def _extra_create_kwargs(self) -> dict:
        """Disable Qwen's default thinking mode so usage matches non-thinking rates."""
        return {"extra_body": {"enable_thinking": False}}
