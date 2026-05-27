"""Tests for the significance display logic in cli._display_significance.

We invoke `_display_significance` directly and capture the Console output
to verify rendering for 2, 3-5, and 6+ model scenarios.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from cli_modelarium import cli as cli_module
from cli_modelarium.run_statistics import SignificanceResult


def _make_result(
    model_a: str,
    model_b: str,
    *,
    p_value: float | None = 0.01,
    p_corrected: float | None = 0.04,
    effect: float | None = 1.2,
    significant: bool = True,
    metric: str = "latency_ms",
    test_used: str = "welch_t_test",
) -> SignificanceResult:
    return SignificanceResult(
        model_a=model_a,
        model_b=model_b,
        metric=metric,
        n_a=10,
        n_b=10,
        mean_a=100.0,
        mean_b=150.0,
        stdev_a=5.0,
        stdev_b=5.0,
        test_used=test_used,
        test_statistic=-5.0,
        degrees_of_freedom=18.0,
        p_value=p_value,
        p_value_corrected=p_corrected,
        correction_method="bonferroni",
        n_comparisons=1,
        effect_size=effect,
        effect_size_interpretation="large",
        threshold=0.05,
        significant_at_threshold=significant,
    )


@pytest.fixture
def captured_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Swap cli.console with a Rich Console that writes to a StringIO buffer."""
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, width=200)
    monkeypatch.setattr(cli_module, "console", test_console)
    return buf


class TestTwoModelSingleLine:
    def test_renders_summary_line(self, captured_console: io.StringIO) -> None:
        results = [_make_result("model_a", "model_b")]
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        assert "Statistical Significance Tests" in out
        assert "model_a" in out
        assert "model_b" in out
        # p-value is rendered to 4 decimals
        assert "0.0400" in out
        # significance marker
        assert "*" in out
        # effect size text
        assert "d=" in out
        assert "large" in out

    def test_not_significant_no_star(self, captured_console: io.StringIO) -> None:
        results = [_make_result("a", "b", p_corrected=0.30, significant=False)]
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        # The line for the comparison shouldn't have "0.3000*"
        assert "0.3000*" not in out
        # But it should have 0.3000
        assert "0.3000" in out

    def test_no_p_value_renders_test_name(self, captured_console: io.StringIO) -> None:
        results = [
            _make_result(
                "a", "b",
                p_value=None, p_corrected=None,
                effect=None, significant=False,
                test_used="insufficient_samples",
            )
        ]
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        assert "insufficient_samples" in out


class TestMatrixDisplay:
    def test_three_models_matrix(self, captured_console: io.StringIO) -> None:
        results = [
            _make_result("a", "b", p_corrected=0.01, significant=True),
            _make_result("a", "c", p_corrected=0.03, significant=True),
            _make_result("b", "c", p_corrected=0.20, significant=False),
        ]
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        assert "Pairwise p-values" in out
        # Three model names in column headers
        assert "a" in out and "b" in out and "c" in out
        # p-value entries
        assert "0.0100" in out
        assert "0.0300" in out
        assert "0.2000" in out

    def test_five_models_still_matrix(self, captured_console: io.StringIO) -> None:
        models = ["m1", "m2", "m3", "m4", "m5"]
        results: list[SignificanceResult] = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                results.append(_make_result(a, b, p_corrected=0.04, significant=True))
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        assert "Pairwise p-values" in out
        # Top-K format should NOT trigger
        assert "Top significant pairs" not in out


class TestTopKDisplay:
    def test_six_models_uses_top_k(self, captured_console: io.StringIO) -> None:
        models = [f"m{i}" for i in range(6)]
        results: list[SignificanceResult] = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                results.append(
                    _make_result(a, b, p_corrected=0.001, significant=True)
                )
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        assert "Top significant pairs" in out
        assert "Full matrix available in JSON output." in out

    def test_six_models_no_significant_pairs(
        self, captured_console: io.StringIO
    ) -> None:
        models = [f"m{i}" for i in range(6)]
        results: list[SignificanceResult] = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                results.append(
                    _make_result(a, b, p_corrected=0.40, significant=False)
                )
        cli_module._display_significance(results)
        out = captured_console.getvalue()
        assert "No statistically significant pairs found" in out


class TestEmptyInput:
    def test_none_does_nothing(self, captured_console: io.StringIO) -> None:
        cli_module._display_significance(None)  # type: ignore[arg-type]
        # Empty case: nothing printed
        assert captured_console.getvalue() == ""

    def test_empty_list_does_nothing(self, captured_console: io.StringIO) -> None:
        cli_module._display_significance([])
        assert captured_console.getvalue() == ""
