"""Tests for paired t-test, Wilcoxon signed-rank, and paired sample
alignment (F4)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import scipy.stats as scipy_stats

from cli_modelarium.run_statistics import (
    _align_paired_samples,
    _extract_paired_metric_samples,
    paired_t_test,
    wilcoxon_signed_rank,
)

SAMPLE_A = [8.4, 8.1, 8.6, 8.3, 8.5, 8.2, 8.7, 8.4, 8.3, 8.5]
SAMPLE_B = [7.9, 7.6, 8.1, 7.8, 7.7, 8.0, 7.5, 7.9, 7.8, 8.2]


@dataclass
class MockState:
    """Stand-in for streaming.StreamState in alignment tests."""

    model: str
    run_index: int
    latency_ms: float | None = None
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class TestPairedTTest:
    def test_matches_scipy_ttest_rel(self) -> None:
        t, df, p = paired_t_test(SAMPLE_A, SAMPLE_B)
        oracle = scipy_stats.ttest_rel(SAMPLE_A, SAMPLE_B)
        assert t == pytest.approx(float(oracle.statistic), abs=1e-9)
        assert p == pytest.approx(float(oracle.pvalue), abs=1e-9)
        assert df == 9.0  # n - 1 for paired

    def test_swap_inverts_t(self) -> None:
        t_ab, _, _ = paired_t_test(SAMPLE_A, SAMPLE_B)
        t_ba, _, _ = paired_t_test(SAMPLE_B, SAMPLE_A)
        assert t_ab == pytest.approx(-t_ba, abs=1e-9)

    def test_unequal_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([1.0], [1.0])

    def test_df_is_n_minus_1(self) -> None:
        _, df, _ = paired_t_test([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
        assert df == 3.0


class TestWilcoxonSignedRank:
    def test_matches_scipy_wilcoxon(self) -> None:
        w, p = wilcoxon_signed_rank(SAMPLE_A, SAMPLE_B)
        oracle = scipy_stats.wilcoxon(
            SAMPLE_A,
            SAMPLE_B,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )
        assert w == pytest.approx(float(oracle.statistic), abs=1e-9)
        assert p == pytest.approx(float(oracle.pvalue), abs=1e-9)

    def test_unequal_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            wilcoxon_signed_rank([1.0, 2.0], [1.0])

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            wilcoxon_signed_rank([1.0], [1.0])

    def test_identical_samples_p_is_one(self) -> None:
        """All-zero differences degrade gracefully (suppress scipy warning)."""
        w, p = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert p == pytest.approx(1.0)


class TestExtractPairedSamples:
    def test_pairs_by_run_index_not_position(self) -> None:
        """F4 fix: asymmetric failures must align by run_index, not by
        position in the filtered array."""
        states = {
            "model_a": [
                MockState("model_a", run_index=i, latency_ms=100.0 + i)
                for i in range(5)
            ],
            "model_b": [
                MockState("model_b", run_index=i, latency_ms=200.0 + i)
                for i in [0, 1, 3, 4]
            ],
        }

        paired = _extract_paired_metric_samples(states, None, "latency_ms")

        assert paired["model_a"] == {
            0: 100.0,
            1: 101.0,
            2: 102.0,
            3: 103.0,
            4: 104.0,
        }
        assert paired["model_b"] == {
            0: 200.0,
            1: 201.0,
            3: 203.0,
            4: 204.0,
        }

        # Alignment should drop run_index 2 (only A has it)
        a_aligned, b_aligned = _align_paired_samples(
            paired["model_a"], paired["model_b"]
        )
        assert a_aligned == [100.0, 101.0, 103.0, 104.0]
        assert b_aligned == [200.0, 201.0, 203.0, 204.0]
        assert len(a_aligned) == 4  # NOT 5

    def test_filters_errored_states(self) -> None:
        states = {
            "model_a": [
                MockState("model_a", run_index=0, latency_ms=100.0),
                MockState("model_a", run_index=1, latency_ms=None, error="boom"),
                MockState("model_a", run_index=2, latency_ms=102.0),
            ],
        }
        paired = _extract_paired_metric_samples(states, None, "latency_ms")
        assert paired["model_a"] == {0: 100.0, 2: 102.0}

    def test_unknown_metric_raises(self) -> None:
        # Non-empty states required to reach the metric-routing branch.
        states = {
            "model_a": [MockState("model_a", run_index=0, latency_ms=100.0)],
        }
        with pytest.raises(ValueError, match="Unknown metric"):
            _extract_paired_metric_samples(states, None, "bogus")

    def test_align_empty_intersection_returns_empty(self) -> None:
        a, b = _align_paired_samples({0: 1.0}, {1: 2.0})
        assert a == []
        assert b == []

    def test_align_sorts_by_index(self) -> None:
        a, b = _align_paired_samples(
            {3: 0.3, 1: 0.1, 2: 0.2}, {3: 1.3, 1: 1.1, 2: 1.2}
        )
        assert a == [0.1, 0.2, 0.3]
        assert b == [1.1, 1.2, 1.3]


class TestPairedTestIntegration:
    def test_paired_t_with_aligned_samples_post_failure(self) -> None:
        """End-to-end: extract + align + run paired test - F4 wiring."""
        states = {
            "model_a": [
                MockState("model_a", run_index=i, latency_ms=100.0 + i)
                for i in range(10)
            ],
            "model_b": [
                MockState("model_b", run_index=i, latency_ms=120.0 + i)
                for i in [0, 1, 2, 4, 5, 6, 7, 8, 9]  # run 3 failed on B
            ],
        }
        paired = _extract_paired_metric_samples(states, None, "latency_ms")
        a, b = _align_paired_samples(paired["model_a"], paired["model_b"])
        assert len(a) == 9  # n=9, not 10
        t, df, p = paired_t_test(a, b)
        assert df == 8.0
