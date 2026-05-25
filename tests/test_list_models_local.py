"""Tests for `cli-modelarium list-models --local` and `LocalProvider.discover_models()`.

Two layers:

    1. Unit tests of `LocalProvider.discover_models()` - httpx mocked so we
       can assert the URL, response parsing, and the unbubbled exceptions.
    2. CLI integration tests via Click's CliRunner that confirm
       --local-url propagation and the friendly error panels for the various
       failure modes (connect, timeout, HTTP error, empty).
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from cli_modelarium.cli import main as cli_main
from cli_modelarium.providers.local_provider import LocalProvider


# ===== fake httpx client =====


class _FakeResponse:
    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
    ) -> None:
        self._json_data = json_data if json_data is not None else {"data": []}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://localhost:11434/v1/models")
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=request, response=httpx.Response(self.status_code, request=request)
            )

    def json(self) -> Any:
        return self._json_data


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse | None = None,
    exception: Exception | None = None,
) -> dict[str, Any]:
    """Replace httpx.AsyncClient inside local_provider with a controllable fake.

    Returns a dict that the test can inspect for captured state (url, timeout).
    """
    captured: dict[str, Any] = {"url": None, "timeout": None}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def get(self, url: str) -> _FakeResponse:
            captured["url"] = url
            if exception is not None:
                raise exception
            assert response is not None
            return response

    monkeypatch.setattr(
        "cli_modelarium.providers.local_provider.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    return captured


# ===== unit tests of discover_models =====


class TestDiscoverModels:
    async def test_returns_data_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _FakeResponse(
            json_data={
                "object": "list",
                "data": [
                    {"id": "llama3.3:latest", "object": "model", "created": 1700000000, "owned_by": "ollama"},
                    {"id": "qwen2.5:32b", "object": "model", "created": 1700001000, "owned_by": "ollama"},
                ],
            }
        )
        _install_fake_client(monkeypatch, response=response)

        models = await LocalProvider.discover_models()

        assert len(models) == 2
        assert models[0]["id"] == "llama3.3:latest"
        assert models[1]["id"] == "qwen2.5:32b"

    async def test_uses_default_url_when_none_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _FakeResponse(json_data={"data": []})
        captured = _install_fake_client(monkeypatch, response=response)

        await LocalProvider.discover_models()

        assert captured["url"] == "http://localhost:11434/v1/models"

    async def test_uses_custom_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _FakeResponse(json_data={"data": []})
        captured = _install_fake_client(monkeypatch, response=response)

        await LocalProvider.discover_models("http://localhost:1234/v1")

        assert captured["url"] == "http://localhost:1234/v1/models"

    async def test_handles_trailing_slash_in_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _FakeResponse(json_data={"data": []})
        captured = _install_fake_client(monkeypatch, response=response)

        await LocalProvider.discover_models("http://localhost:11434/v1/")

        # Trailing slash shouldn't double-up to /v1//models.
        assert captured["url"] == "http://localhost:11434/v1/models"

    async def test_timeout_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _FakeResponse(json_data={"data": []})
        captured = _install_fake_client(monkeypatch, response=response)

        await LocalProvider.discover_models()

        assert captured["timeout"] == LocalProvider.DISCOVERY_TIMEOUT_SECONDS

    async def test_empty_response_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _FakeResponse(json_data={"data": []})
        _install_fake_client(monkeypatch, response=response)

        models = await LocalProvider.discover_models()

        assert models == []

    async def test_missing_data_key_returns_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive: server returns malformed payload without `data` key."""
        response = _FakeResponse(json_data={"unexpected": "shape"})
        _install_fake_client(monkeypatch, response=response)

        models = await LocalProvider.discover_models()

        assert models == []

    async def test_connect_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Connection errors propagate so the CLI can render a friendly panel."""
        _install_fake_client(monkeypatch, exception=httpx.ConnectError("refused"))

        with pytest.raises(httpx.ConnectError):
            await LocalProvider.discover_models()

    async def test_timeout_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_client(monkeypatch, exception=httpx.ReadTimeout("timed out"))

        with pytest.raises(httpx.TimeoutException):
            await LocalProvider.discover_models()


# ===== CLI integration via CliRunner =====


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences for substring assertions on Rich output."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestListModelsLocalCli:
    def test_renders_models_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _FakeResponse(
            json_data={
                "data": [
                    {"id": "llama3.3:latest", "created": 1700000000, "owned_by": "ollama"},
                    {"id": "qwen2.5:32b", "created": 1700001000, "owned_by": "ollama"},
                ]
            }
        )
        _install_fake_client(monkeypatch, response=response)

        runner = CliRunner()
        result = runner.invoke(cli_main, ["list-models", "--local"])

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "local/llama3.3:latest" in output
        assert "local/qwen2.5:32b" in output
        assert "Free" in output

    def test_connect_error_renders_friendly_panel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Connection refused must NOT fail the command - shows a panel instead."""
        _install_fake_client(monkeypatch, exception=httpx.ConnectError("refused"))

        runner = CliRunner()
        result = runner.invoke(cli_main, ["list-models", "--local"])

        # Soft failure: exit 0 with a friendly message, not exit 1.
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "Could not reach" in output or "no local server" in output.lower()

    def test_timeout_renders_friendly_panel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_client(monkeypatch, exception=httpx.ReadTimeout("slow"))

        runner = CliRunner()
        result = runner.invoke(cli_main, ["list-models", "--local"])

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "timed out" in output.lower() or "Timed out" in output

    def test_empty_server_response_renders_helpful_panel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _FakeResponse(json_data={"data": []})
        _install_fake_client(monkeypatch, response=response)

        runner = CliRunner()
        result = runner.invoke(cli_main, ["list-models", "--local"])

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "no models" in output.lower() or "ollama pull" in output.lower()

    def test_local_url_flag_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--local-url must reach the GET call so users can target LM Studio etc."""
        response = _FakeResponse(json_data={"data": []})
        captured = _install_fake_client(monkeypatch, response=response)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["list-models", "--local", "--local-url", "http://localhost:1234/v1"],
        )

        assert result.exit_code == 0
        assert captured["url"] == "http://localhost:1234/v1/models"

    def test_invalid_local_url_rejected_before_network_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-localhost URL must be rejected before any HTTP I/O."""
        captured = _install_fake_client(monkeypatch, response=_FakeResponse())

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["list-models", "--local", "--local-url", "http://api.openai.com/v1"],
        )

        # Exit non-zero AND no network call should have happened.
        assert result.exit_code != 0
        assert captured["url"] is None
        output = _strip_ansi(result.output)
        assert "localhost" in output.lower()
