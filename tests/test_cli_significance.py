"""CLI-level tests for the significance flags on the compare command.

Mirrors the _RecordingProvider pattern from tests/test_cli_runs.py.
A _VaryingProvider returns different latencies per call so significance
tests have non-zero variance to chew on.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.cli import main as cli_main
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


class _VaryingProvider(BaseProvider):
    """Returns latency that varies per (model, call_index) so significance
    tests see non-zero variance and produce real p-values.
    """

    def __init__(self) -> None:
        self.name = "fake"
        self.calls: list[dict[str, Any]] = []
        self._call_index_by_model: dict[str, int] = {}

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
        idx = self._call_index_by_model.get(model, 0)
        self._call_index_by_model[model] = idx + 1
        self.calls.append({"model": model, "idx": idx})

        # Assign each unique model a distinct base latency, with small
        # within-model jitter so Welch's t-test has non-zero variance.
        # The first model seen gets ~50ms, the second ~150ms, etc.
        if not hasattr(self, "_base_by_model"):
            self._base_by_model: dict[str, float] = {}
        if model not in self._base_by_model:
            self._base_by_model[model] = 50.0 + 100.0 * len(self._base_by_model)
        latency = self._base_by_model[model] + (idx % 5) * 1.5

        response_text = f"answer-{model}-{idx}"
        if on_chunk is not None:
            on_chunk(response_text)
        return CompletionResult(
            output=response_text,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.000123,
            latency_ms=latency,
            ttft_ms=10.0,
            model=model,
            provider="fake",
            temperature=temperature,
        )


@pytest.fixture
def varying_provider(monkeypatch: pytest.MonkeyPatch) -> _VaryingProvider:
    fake = _VaryingProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    return fake


# ===== Flag parsing & help =====


class TestSignificanceHelp:
    def test_all_flags_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--significance" in result.output
        assert "--no-significance" in result.output
        assert "--significance-threshold" in result.output
        assert "--significance-test" in result.output
        assert "--correction" in result.output
        assert "--significance-metric" in result.output


class TestSignificanceFlagValidation:
    def test_threshold_out_of_range_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--significance-threshold", "1.5",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_threshold_zero_rejected(self, varying_provider: _VaryingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--significance-threshold", "0",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_test_rejected(self, varying_provider: _VaryingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--significance-test", "bogus",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_correction_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--correction", "bogus",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0


# ===== Auto-enable behavior =====


class TestAutoEnable:
    def test_single_run_no_significance_block(
        self, varying_provider: _VaryingProvider
    ) -> None:
        """--runs 1 should never show significance."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5,claude-opus-4-7", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        assert "Statistical Significance Tests" not in result.output

    def test_runs_gt_1_single_model_no_significance(
        self, varying_provider: _VaryingProvider
    ) -> None:
        """Only one model => no pairs => no significance display."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["q", "--models", "gpt-5.5", "--runs", "5", "--no-stream"],
        )
        assert result.exit_code == 0, result.output
        assert "Statistical Significance Tests" not in result.output

    def test_runs_gt_1_two_models_auto_enabled(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Statistical Significance Tests" in result.output

    def test_no_significance_opts_out(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--no-significance",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Statistical Significance Tests" not in result.output


# ===== Test selection =====


class TestSignificanceTestSelection:
    def test_welch_is_default(self, varying_provider: _VaryingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0
        assert "welch" in result.output.lower()

    def test_mann_whitney_explicit(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--significance-test", "mann-whitney",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0
        assert "mann_whitney" in result.output or "mann-whitney" in result.output


# ===== JSON output additive =====


class TestJsonOutput:
    def test_no_significance_key_when_runs_eq_1(
        self,
        varying_provider: _VaryingProvider,
        tmp_path: Any,
    ) -> None:
        """When runs == 1 and no significance computed, JSON must not include
        the significance_tests key (backward compat)."""
        runner = CliRunner()
        out = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--output", str(out),
                "--output-format", "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text())
        assert "significance_tests" not in payload

    def test_no_significance_key_when_opt_out(
        self,
        varying_provider: _VaryingProvider,
        tmp_path: Any,
    ) -> None:
        """With --no-significance, JSON must not include significance_tests."""
        runner = CliRunner()
        out = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--no-significance",
                "--output", str(out),
                "--output-format", "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text())
        assert "significance_tests" not in payload

    def test_significance_key_present_when_computed(
        self,
        varying_provider: _VaryingProvider,
        tmp_path: Any,
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--output", str(out),
                "--output-format", "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text())
        assert "significance_tests" in payload
        tests = payload["significance_tests"]
        assert len(tests) == 1
        entry = tests[0]
        # Fields the prompt requires for the JSON schema
        for required_key in [
            "model_a", "model_b", "metric", "n_a", "n_b",
            "mean_a", "mean_b", "test_used", "p_value",
            "p_value_corrected", "correction_method", "n_comparisons",
            "effect_size", "effect_size_metric",
            "effect_size_interpretation", "threshold",
            "significant_at_threshold",
        ]:
            assert required_key in entry, f"missing key {required_key} in JSON"
        assert entry["correction_method"] == "bonferroni"  # default
        assert entry["effect_size_metric"] == "cohens_d"
        assert entry["test_used"] == "welch_t_test"

    def test_default_metric_latency_when_no_judge(
        self,
        varying_provider: _VaryingProvider,
        tmp_path: Any,
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--runs", "5",
                "--output", str(out),
                "--output-format", "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text())
        assert payload["significance_tests"][0]["metric"] == "latency_ms"


# ===== Backward-compat: runs == 1 JSON unchanged =====


class TestBackwardCompat:
    def test_runs_1_json_has_no_total_runs_or_significance(
        self,
        varying_provider: _VaryingProvider,
        tmp_path: Any,
    ) -> None:
        runner = CliRunner()
        out = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models", "gpt-5.5,claude-opus-4-7",
                "--output", str(out),
                "--output-format", "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text())
        assert "significance_tests" not in payload
        assert "total_runs" not in payload
        assert "stats_by_cell" not in payload
