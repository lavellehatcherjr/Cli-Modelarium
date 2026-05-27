"""CLI-level tests for v0.1.3 flags: --confidence-intervals, --ci-level,
--ci-method, --bootstrap-resamples, --bootstrap-seed, plus the new
paired-t / wilcoxon-signed test choices.

Reuses the _VaryingProvider pattern from tests/test_cli_significance.py.
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
    """Latency varies per (model, call_index) so CIs and paired tests
    see real variance."""

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


# ===== Help text =====


class TestV013FlagsInHelp:
    def test_all_new_flags_present(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--confidence-intervals" in result.output
        assert "--no-confidence-intervals" in result.output
        assert "--ci-level" in result.output
        assert "--ci-method" in result.output
        assert "--bootstrap-resamples" in result.output
        assert "--bootstrap-seed" in result.output

    def test_new_test_choices_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "paired-t" in result.output
        assert "wilcoxon-signed" in result.output

    def test_v012_flags_still_in_help(self) -> None:
        """Backward compat: existing v0.1.2 flags are unchanged."""
        runner = CliRunner()
        result = runner.invoke(cli_main, ["compare", "--help"])
        assert result.exit_code == 0
        for flag in (
            "--significance",
            "--no-significance",
            "--significance-threshold",
            "--significance-test",
            "--correction",
            "--significance-metric",
        ):
            assert flag in result.output


# ===== Flag validation =====


class TestV013FlagValidation:
    def test_ci_level_above_1_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "5",
                "--ci-level",
                "1.5",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_ci_level_zero_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "5",
                "--ci-level",
                "0",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_ci_method_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "5",
                "--ci-method",
                "bogus",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_bootstrap_resamples_too_low_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "5",
                "--bootstrap-resamples",
                "50",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_significance_test_rejected(
        self, varying_provider: _VaryingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "5",
                "--significance-test",
                "bogus",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0


# ===== Auto-enable behavior =====


class TestCIAutoEnable:
    def test_runs_gt_1_auto_enables_cis(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--bootstrap-seed",
                "42",
                "--bootstrap-resamples",
                "1000",
                "--output",
                str(out_file),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out_file.read_text())
        # stats_by_cell should include CI fields when CIs are auto-enabled.
        assert "stats_by_cell" in data
        any_ci_present = any(
            "latency_mean_ms_ci_low" in cell for cell in data["stats_by_cell"]
        )
        assert any_ci_present

    def test_no_confidence_intervals_strips_cis(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        """--no-confidence-intervals returns v0.1.2-shape JSON for cells."""
        runner = CliRunner()
        out_file = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--no-confidence-intervals",
                "--output",
                str(out_file),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out_file.read_text())
        for cell in data["stats_by_cell"]:
            assert "latency_mean_ms_ci_low" not in cell
            assert "latency_mean_ms_ci_high" not in cell

    def test_runs_eq_1_no_cis(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--output",
                str(out_file),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out_file.read_text())
        # No stats_by_cell when runs == 1.
        assert "stats_by_cell" not in data


# ===== Reproducibility =====


class TestReproducibility:
    def test_same_seed_identical_ci_bounds(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()

        def _run(out_file):
            return runner.invoke(
                cli_main,
                [
                    "compare",
                    "Q",
                    "--models",
                    "gpt-5.5,claude-opus-4-7",
                    "--runs",
                    "10",
                    "--bootstrap-seed",
                    "42",
                    "--bootstrap-resamples",
                    "1000",
                    "--output",
                    str(out_file),
                    "--output-format",
                    "json",
                    "--no-stream",
                ],
            )

        out1 = tmp_path / "r1.json"
        out2 = tmp_path / "r2.json"
        # Need separate provider instances for each invocation, but the
        # fixture installs one. Both runs see the same sequence of calls.
        r1 = _run(out1)
        # Reset the provider counters so the second run sees same inputs.
        varying_provider.calls.clear()
        varying_provider._call_index_by_model.clear()
        if hasattr(varying_provider, "_base_by_model"):
            varying_provider._base_by_model.clear()
        r2 = _run(out2)
        assert r1.exit_code == 0
        assert r2.exit_code == 0

        d1 = json.loads(out1.read_text())
        d2 = json.loads(out2.read_text())
        cis1 = [cell.get("latency_mean_ms_ci_low") for cell in d1["stats_by_cell"]]
        cis2 = [cell.get("latency_mean_ms_ci_low") for cell in d2["stats_by_cell"]]
        assert cis1 == cis2

    def test_methodology_block_records_seed(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--bootstrap-seed",
                "7",
                "--bootstrap-resamples",
                "1000",
                "--output",
                str(out_file),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out_file.read_text())
        assert "methodology" in data
        assert data["methodology"]["bootstrap"]["seed"] == 7
        assert data["methodology"]["bootstrap"]["enabled"] is True
        assert data["methodology"]["bootstrap"]["n_resamples"] == 1000


# ===== Paired tests =====


class TestPairedTestRouting:
    def test_paired_t_appears_in_json(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--significance-test",
                "paired-t",
                "--no-confidence-intervals",
                "--output",
                str(out_file),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out_file.read_text())
        assert "significance_tests" in data
        tests = data["significance_tests"]
        assert all(t["test_used"] == "paired_t_test" for t in tests)

    def test_wilcoxon_appears_in_json(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.json"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--significance-test",
                "wilcoxon-signed",
                "--no-confidence-intervals",
                "--output",
                str(out_file),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(out_file.read_text())
        tests = data["significance_tests"]
        assert all(t["test_used"] == "wilcoxon_signed_rank" for t in tests)


# ===== CSV/Markdown plumbing (F3 fix) =====


class TestCSVFormatF3Wiring:
    def test_csv_with_cis_has_new_columns(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.csv"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--bootstrap-seed",
                "42",
                "--bootstrap-resamples",
                "1000",
                "--output",
                str(out_file),
                "--output-format",
                "csv",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        header = out_file.read_text().splitlines()[0]
        assert "latency_ms_ci_low" in header
        assert "latency_ms_ci_high" in header

    def test_csv_no_cis_preserves_v012_columns(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        """Backward compat: --no-confidence-intervals keeps CSV columns
        identical to v0.1.2."""
        runner = CliRunner()
        out_file = tmp_path / "out.csv"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--no-confidence-intervals",
                "--output",
                str(out_file),
                "--output-format",
                "csv",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        header = out_file.read_text().splitlines()[0]
        assert "latency_ms_ci_low" not in header
        assert "latency_ms_ci_high" not in header


class TestMarkdownFormatF3Wiring:
    def test_markdown_with_cis_has_section(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.md"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--bootstrap-seed",
                "42",
                "--bootstrap-resamples",
                "1000",
                "--output",
                str(out_file),
                "--output-format",
                "markdown",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        text = out_file.read_text()
        assert "Bootstrap confidence intervals" in text
        assert "Statistical methodology" in text

    def test_markdown_no_cis_no_section(
        self,
        varying_provider: _VaryingProvider,
        tmp_path,
    ) -> None:
        runner = CliRunner()
        out_file = tmp_path / "out.md"
        result = runner.invoke(
            cli_main,
            [
                "compare",
                "Q",
                "--models",
                "gpt-5.5,claude-opus-4-7",
                "--runs",
                "10",
                "--no-confidence-intervals",
                "--no-significance",
                "--output",
                str(out_file),
                "--output-format",
                "markdown",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        text = out_file.read_text()
        assert "Bootstrap confidence intervals" not in text
