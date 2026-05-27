"""CLI-level tests for --max-cost on the `compare` command."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.cli import main as cli_main
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


class _RecordingProvider(BaseProvider):
    def __init__(self) -> None:
        self.name = "fake"
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""
        raise NotImplementedError

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None = None,
        *,
        on_chunk: OnChunk | None = None,
    ) -> CompletionResult:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
            }
        )
        text = f"answer for {prompt[:20]}"
        if on_chunk is not None:
            on_chunk(text)
        return CompletionResult(
            output=text,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.000123,
            latency_ms=42.0,
            ttft_ms=12.0,
            model=model,
            provider="fake",
            temperature=temperature,
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> _RecordingProvider:
    fake = _RecordingProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    return fake


class TestCompareMaxCost:
    def test_estimate_over_max_cost_refuses(
        self, fake_provider: _RecordingProvider
    ) -> None:
        """gpt-5.5 with 500 in / 500 out estimates well above $0.001."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--max-cost", "0.001", "--no-stream"],
        )

        assert result.exit_code != 0
        assert "estimated cost" in result.output.lower()
        # No provider calls happened.
        assert fake_provider.calls == []

    def test_estimate_under_max_cost_proceeds(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--max-cost", "10.0", "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 1

    def test_max_cost_zero_with_local_succeeds(
        self, fake_provider: _RecordingProvider
    ) -> None:
        """Local models contribute $0 to the estimate so --max-cost 0 passes."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "local/foo", "--max-cost", "0", "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 1

    def test_max_cost_zero_with_paid_refuses(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--max-cost", "0", "--no-stream"],
        )

        assert result.exit_code != 0
        assert "estimated cost" in result.output.lower()
        assert fake_provider.calls == []

    def test_max_cost_negative_rejected_by_click(
        self, fake_provider: _RecordingProvider
    ) -> None:
        """FloatRange(min=0.0) makes Click reject negatives before the body runs."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--max-cost", "-1", "--no-stream"],
        )

        assert result.exit_code != 0
        # Click's standard out-of-range message contains the value.
        assert "-1" in result.output
        assert fake_provider.calls == []
