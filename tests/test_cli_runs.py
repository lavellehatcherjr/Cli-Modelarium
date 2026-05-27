"""CLI-level tests for the --runs flag on the compare command.

Mirrors the _RecordingProvider pattern from tests/test_cli_compare_output.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.cli import main as cli_main
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


class _RecordingProvider(BaseProvider):
    """Returns a preset CompletionResult and records every call."""

    def __init__(self, response_text: str = "answer") -> None:
        self.name = "fake"
        self.calls: list[dict[str, Any]] = []
        self._response_text = response_text

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
        if on_chunk is not None:
            on_chunk(self._response_text)
        return CompletionResult(
            output=self._response_text,
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


class TestRunsFlagValidation:
    def test_runs_default_is_one(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 1

    def test_runs_one_explicit(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "1", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 1

    def test_runs_zero_rejected(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "0", "--no-stream"],
        )
        assert result.exit_code != 0
        assert "0" in result.output or "Invalid value" in result.output

    def test_runs_negative_rejected(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "-1", "--no-stream"],
        )
        assert result.exit_code != 0

    def test_runs_over_100_rejected(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "101", "--no-stream"],
        )
        assert result.exit_code != 0


class TestRunsCallCounts:
    def test_runs_five_single_model(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "5", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 5

    def test_runs_five_multiple_models(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "5",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 10

    def test_runs_with_temperatures(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--temperatures",
                "0,0.7",
                "--runs",
                "5",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 10

    def test_runs_with_system_prompts(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--system-prompts",
                "you are math,you are physics",
                "--runs",
                "5",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 10


class TestRunsHelpAndUsage:
    def test_runs_flag_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--runs" in result.output
        assert "--show-all-runs" in result.output

    def test_runs_cost_warning_when_no_max_cost(
        self, fake_provider: _RecordingProvider
    ) -> None:
        """Without --max-cost, a cost warning should print when runs > 1 has paid models."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "3", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        # Cost warning text matches the format in cli.py.
        assert "multiplies cost" in result.output or "Estimated total" in result.output
