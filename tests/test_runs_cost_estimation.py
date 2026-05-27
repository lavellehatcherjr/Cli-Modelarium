"""Tests for cost estimation with --runs N."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.batch import estimate_compare_cost
from cli_modelarium.cli import main as cli_main
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


class _FakeProvider(BaseProvider):
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
            {"prompt": prompt, "model": model, "temperature": temperature}
        )
        if on_chunk is not None:
            on_chunk("ok")
        return CompletionResult(
            output="ok",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0001,
            latency_ms=1.0,
            model=model,
            provider="fake",
            temperature=temperature,
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> _FakeProvider:
    fake = _FakeProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    return fake


def test_estimate_compare_cost_pure_unchanged_by_runs() -> None:
    """estimate_compare_cost remains a pure single-pass function;
    runs multiplication happens in the CLI layer.
    """
    base = estimate_compare_cost(["gpt-5.5"], [0.0], [None])
    assert base > 0
    # The function takes no `runs` parameter; consumers multiply.
    assert estimate_compare_cost(["gpt-5.5"], [0.0], [None]) == base


class TestMaxCostMultipliedByRuns:
    def test_max_cost_exceeded_with_runs_refused(
        self, fake_provider: _FakeProvider
    ) -> None:
        """With --runs 100 and a tiny --max-cost, the run refuses."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--runs",
                "100",
                "--max-cost",
                "0.0001",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0
        assert "exceeds --max-cost" in result.output
        # The error message should reference the multiplication.
        assert "x 100" in result.output or "100" in result.output
        # No actual calls should have happened.
        assert len(fake_provider.calls) == 0

    def test_max_cost_passes_with_runs(self, fake_provider: _FakeProvider) -> None:
        """With a generous --max-cost, --runs succeeds."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--runs",
                "3",
                "--max-cost",
                "1.00",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 3


class TestCostWarningOnRunsGreaterThanOne:
    def test_warning_printed_without_max_cost(
        self, fake_provider: _FakeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "3", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        # The yellow warning appears when --runs > 1 and no --max-cost is set.
        assert "multiplies cost" in result.output or "Estimated total" in result.output

    def test_no_warning_when_runs_is_one(self, fake_provider: _FakeProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "1", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        assert "multiplies cost" not in result.output
