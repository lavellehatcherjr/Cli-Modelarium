"""Tests for cli_modelarium.providers.local_provider.

Three things are unique to LocalProvider compared to its OpenAIProvider parent:

    1. localhost-only URL validation (security guard)
    2. `_transform_model()` strips the `local/` prefix
    3. Cost is always 0 because the pricing entry resolves to local/* ($0)

Plus the inheritance contract: stream/complete are NOT overridden, so all
streaming and error-handling behaviour comes from OpenAIProvider for free.
"""
from __future__ import annotations

from typing import Any

import pytest

from cli_modelarium.exceptions import LocalURLError
from cli_modelarium.pricing import calculate_cost
from cli_modelarium.providers.local_provider import LocalProvider
from cli_modelarium.providers.openai_provider import OpenAIProvider


# ===== URL validation =====


class TestUrlValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434/v1",
            "http://localhost:1234/v1",
            "http://127.0.0.1:11434/v1",
            "http://127.0.0.1/v1",
            "http://0.0.0.0:8000/v1",
            "http://[::1]:11434/v1",  # IPv6 must be in brackets
            "https://localhost:11434/v1",  # https also ok for local TLS setups
        ],
    )
    def test_accepts_localhost_variants(self, url: str) -> None:
        assert LocalProvider._validate_local_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "http://evil.com/v1",
            "https://api.example.com/v1",
            "http://1.2.3.4:8000/v1",
            "http://192.168.1.100:11434/v1",  # LAN IP - explicitly rejected
            "http://api.openai.com/v1",
            "https://api.anthropic.com/v1",
        ],
    )
    def test_rejects_non_localhost(self, url: str) -> None:
        with pytest.raises(LocalURLError) as exc_info:
            LocalProvider._validate_local_url(url)
        # Error must mention the actual hostname for actionability.
        assert "localhost" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://localhost/v1",
            "ws://localhost:11434",
            "file:///etc/passwd",
        ],
    )
    def test_rejects_non_http_schemes(self, url: str) -> None:
        with pytest.raises(LocalURLError) as exc_info:
            LocalProvider._validate_local_url(url)
        assert "http" in str(exc_info.value).lower()

    def test_ipv6_uses_bracket_syntax(self) -> None:
        """RFC 3986: IPv6 literals must be in square brackets in URLs."""
        # Bracketed form - accepted.
        assert LocalProvider._validate_local_url("http://[::1]:11434/v1")
        # Unbracketed form - urlparse can't extract `::1` as the hostname, so
        # it should NOT validate as localhost. Confirms the bracket convention.
        with pytest.raises(LocalURLError):
            LocalProvider._validate_local_url("http://::1:11434/v1")


# ===== model name transformation =====


class TestTransformModel:
    def test_strips_local_prefix(self) -> None:
        provider = LocalProvider()
        assert provider._transform_model("local/llama-3.3-70b") == "llama-3.3-70b"

    def test_strips_only_first_occurrence(self) -> None:
        provider = LocalProvider()
        # removeprefix only strips at the start.
        assert provider._transform_model("local/llama/local/v2") == "llama/local/v2"

    def test_pass_through_if_no_prefix(self) -> None:
        provider = LocalProvider()
        # A model without the prefix is left alone (defensive).
        assert provider._transform_model("plain-model") == "plain-model"

    def test_transform_called_before_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When complete() runs, the SDK should receive the stripped model name."""
        captured: dict[str, Any] = {}

        class _FakeCompletions:
            async def create(self, **kwargs: Any) -> Any:
                captured.update(kwargs)
                # Return an empty async iterator to satisfy the streaming loop.
                class _Empty:
                    def __aiter__(self) -> "_Empty":
                        return self

                    async def __anext__(self) -> Any:
                        raise StopAsyncIteration

                return _Empty()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        monkeypatch.setattr(
            "cli_modelarium.providers.openai_provider.AsyncOpenAI",
            lambda **_kwargs: _FakeClient(),
        )

        import asyncio

        provider = LocalProvider()
        asyncio.run(provider.complete("hi", "local/llama-3.3-70b", 0.0))

        # The transformed name (no prefix) must reach the SDK.
        assert captured["model"] == "llama-3.3-70b"


# ===== inheritance / construction =====


class TestConstruction:
    def test_inherits_from_openai_provider(self) -> None:
        assert issubclass(LocalProvider, OpenAIProvider)

    def test_does_not_override_stream_or_complete(self) -> None:
        # The four OpenAI-compat subclasses share this contract: all
        # streaming behavior is inherited unchanged.
        assert "stream" not in LocalProvider.__dict__
        assert "complete" not in LocalProvider.__dict__

    def test_default_url_is_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)

        LocalProvider()

        assert captured["base_url"] == "http://localhost:11434/v1"
        # Confirms the build prompt's intended port.
        assert ":11434/v1" in captured["base_url"]

    def test_custom_base_url_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)

        LocalProvider(base_url="http://localhost:1234/v1")

        assert captured["base_url"] == "http://localhost:1234/v1"

    def test_dummy_api_key_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local servers ignore the api_key; we send `not-required`."""
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)

        LocalProvider()

        assert captured["api_key"] == "not-required"

    def test_invalid_url_at_construction_time(self) -> None:
        """Non-localhost URLs are rejected BEFORE the SDK client is created."""
        with pytest.raises(LocalURLError):
            LocalProvider(base_url="http://api.openai.com/v1")

    def test_name_is_local(self) -> None:
        assert LocalProvider.name == "local"


# ===== cost is always free =====


class TestLocalCostIsFree:
    @pytest.mark.parametrize(
        "model",
        [
            "local/llama-3.3-70b",
            "local/qwen-3-32b",
            "local/anything-with-any-name",
            "local/my-custom-finetune:v2",
        ],
    )
    def test_local_cost_zero_regardless_of_tokens(self, model: str) -> None:
        assert calculate_cost(model, input_tokens=1_000_000, output_tokens=1_000_000) == 0.0
        assert calculate_cost(model, input_tokens=999_999_999, output_tokens=999_999_999) == 0.0
        # Even cached tokens stay free.
        assert calculate_cost(model, input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=999_999) == 0.0


# ===== discovery (URL validation hook) =====


class TestDiscoveryValidatesUrl:
    """`discover_models()` runs URL validation BEFORE any network I/O,
    so a misconfigured remote URL can't even attempt to leak data.
    """

    async def test_validates_url_before_network_call(self) -> None:
        with pytest.raises(LocalURLError):
            await LocalProvider.discover_models("http://evil.example.com/v1")
