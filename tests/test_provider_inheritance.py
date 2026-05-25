"""Tests for the four OpenAI-compatible subclasses.

xAI, DeepSeek, Groq, and OpenRouter all use the OpenAI SDK with a different
base URL (and, for OpenRouter, extra default headers). These tests verify
the subclasses correctly forward those values to the AsyncOpenAI constructor
and inherit all other behavior from OpenAIProvider.
"""
from __future__ import annotations

from typing import Any

import pytest

from cli_modelarium.providers.deepseek_provider import DeepSeekProvider
from cli_modelarium.providers.groq_provider import GroqProvider
from cli_modelarium.providers.openai_provider import OpenAIProvider
from cli_modelarium.providers.openrouter_provider import OpenRouterProvider
from cli_modelarium.providers.xai_provider import XAIProvider


SUBCLASSES = [
    (XAIProvider, "xai", "https://api.x.ai/v1"),
    (DeepSeekProvider, "deepseek", "https://api.deepseek.com/v1"),
    (GroqProvider, "groq", "https://api.groq.com/openai/v1"),
    (OpenRouterProvider, "openrouter", "https://openrouter.ai/api/v1"),
]


@pytest.mark.parametrize("cls,expected_name,expected_url", SUBCLASSES)
def test_inherits_from_openai_provider(
    cls: type, expected_name: str, expected_url: str
) -> None:
    assert issubclass(cls, OpenAIProvider)


@pytest.mark.parametrize("cls,expected_name,expected_url", SUBCLASSES)
def test_provider_name(
    cls: type, expected_name: str, expected_url: str
) -> None:
    assert cls.name == expected_name


@pytest.mark.parametrize("cls,expected_name,expected_url", SUBCLASSES)
def test_base_url_set_on_class(
    cls: type, expected_name: str, expected_url: str
) -> None:
    assert cls.BASE_URL == expected_url


@pytest.mark.parametrize("cls,expected_name,expected_url", SUBCLASSES)
def test_base_url_forwarded_to_async_openai(
    cls: type,
    expected_name: str,
    expected_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)

    cls(api_key="sk-test-1234567890abcdefghi")

    assert captured["api_key"] == "sk-test-1234567890abcdefghi"
    assert captured["base_url"] == expected_url


def test_openrouter_sends_required_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter's convention: every request carries HTTP-Referer + X-Title."""
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)

    OpenRouterProvider(api_key="sk-or-test-1234567890abcdefghi")

    headers = captured["default_headers"]
    assert headers["HTTP-Referer"] == "https://github.com/lavellehatcherjr/cli-modelarium"
    assert headers["X-Title"] == "Cli Modelarium"


@pytest.mark.parametrize("cls", [XAIProvider, DeepSeekProvider, GroqProvider])
def test_non_openrouter_subclasses_do_not_send_default_headers(
    cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only OpenRouter needs special headers - others omit `default_headers`."""
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)

    cls(api_key="sk-test-1234567890abcdefghi")

    assert "default_headers" not in captured


@pytest.mark.parametrize("cls", [XAIProvider, DeepSeekProvider, GroqProvider, OpenRouterProvider])
def test_subclass_inherits_stream_and_complete(cls: type) -> None:
    """The OpenAI-compat subclasses must NOT override stream/complete - they inherit."""
    # The methods should be defined on OpenAIProvider, not on the subclass itself.
    assert "stream" not in cls.__dict__
    assert "complete" not in cls.__dict__


def test_subclass_can_use_transform_model_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a future subclass wants to rewrite the model ID (e.g. LocalProvider),
    the parent's `_transform_model` hook is in place for it.
    """

    class Rewriter(OpenAIProvider):
        name = "rewriter-test"

        def _transform_model(self, model: str) -> str:
            return model.removeprefix("prefix/")

    def capture(**_kwargs: Any) -> Any:
        return object()

    monkeypatch.setattr("cli_modelarium.providers.openai_provider.AsyncOpenAI", capture)
    p = Rewriter(api_key="sk-test-1234567890abcdefghi")
    assert p._transform_model("prefix/foo") == "foo"
    assert p._transform_model("plain") == "plain"
