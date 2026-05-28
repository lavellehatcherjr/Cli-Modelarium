"""Tests for bootstrap confidence intervals via scipy.stats.bootstrap.

All numerical assertions verify against scipy.stats.bootstrap directly
as the oracle. Reproducibility tests use seeds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats as scipy_stats

from cli_modelarium.run_statistics import ConfidenceInterval, bootstrap_ci

SAMPLE = [8.4, 8.1, 8.6, 8.3, 8.5, 8.2, 8.7, 8.4, 8.3, 8.5]


class TestBootstrapCIBasic:
    def test_basic_bca_ci_on_mean(self) -> None:
        result = bootstrap_ci(SAMPLE, seed=42)
        assert result is not None
        assert isinstance(result, ConfidenceInterval)
        assert result.point_estimate == pytest.approx(float(np.mean(SAMPLE)))
        assert result.ci_low < result.point_estimate < result.ci_high
        assert result.method == "bca"
        assert result.n_resamples == 5000
        assert result.seed == 42
        assert result.n_samples == len(SAMPLE)
        assert result.ci_level == 0.95

    def test_default_method_is_bca(self) -> None:
        result = bootstrap_ci(SAMPLE, seed=42)
        assert result.method == "bca"

    def test_default_n_resamples_is_5000(self) -> None:
        result = bootstrap_ci(SAMPLE, seed=42)
        assert result.n_resamples == 5000

    def test_default_ci_level_is_95(self) -> None:
        result = bootstrap_ci(SAMPLE, seed=42)
        assert result.ci_level == 0.95


class TestBootstrapCIReproducibility:
    def test_same_seed_identical_cis(self) -> None:
        """Same seed must produce byte-identical CIs - critical for
        publication-grade reproducibility."""
        r1 = bootstrap_ci(SAMPLE, seed=42)
        r2 = bootstrap_ci(SAMPLE, seed=42)
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_different_seeds_different_cis(self) -> None:
        r1 = bootstrap_ci(SAMPLE, seed=42)
        r2 = bootstrap_ci(SAMPLE, seed=999)
        assert (r1.ci_low != r2.ci_low) or (r1.ci_high != r2.ci_high)

    def test_no_seed_runs_drift(self) -> None:
        """Without a seed, repeated runs should vary (non-deterministic).

        A single pair can coincidentally match when the sample is small and
        bounds round to a few decimals, so run several and assert that not
        all of them are identical.
        """
        results = [
            bootstrap_ci(SAMPLE, seed=None, n_resamples=1000) for _ in range(10)
        ]
        distinct_bounds = {(r.ci_low, r.ci_high) for r in results}
        assert len(distinct_bounds) > 1, (
            "Expected unseeded bootstrap runs to vary, but all 10 produced "
            "identical CI bounds"
        )


class TestBootstrapCILevel:
    def test_99_wider_than_95(self) -> None:
        r_95 = bootstrap_ci(SAMPLE, ci_level=0.95, seed=42)
        r_99 = bootstrap_ci(SAMPLE, ci_level=0.99, seed=42)
        width_95 = r_95.ci_high - r_95.ci_low
        width_99 = r_99.ci_high - r_99.ci_low
        assert width_99 > width_95

    def test_90_narrower_than_95(self) -> None:
        r_90 = bootstrap_ci(SAMPLE, ci_level=0.90, seed=42)
        r_95 = bootstrap_ci(SAMPLE, ci_level=0.95, seed=42)
        assert (r_90.ci_high - r_90.ci_low) < (r_95.ci_high - r_95.ci_low)


class TestBootstrapCIMethods:
    def test_percentile(self) -> None:
        result = bootstrap_ci(SAMPLE, method="percentile", seed=42)
        assert result is not None
        assert result.method == "percentile"

    def test_basic(self) -> None:
        result = bootstrap_ci(SAMPLE, method="basic", seed=42)
        assert result is not None
        assert result.method == "basic"

    def test_bca(self) -> None:
        result = bootstrap_ci(SAMPLE, method="bca", seed=42)
        assert result is not None
        assert result.method == "bca"


class TestBootstrapCIEdgeCases:
    def test_empty_sample_returns_none(self) -> None:
        assert bootstrap_ci([], seed=42) is None

    def test_n_eq_1_returns_none(self) -> None:
        assert bootstrap_ci([5.0], seed=42) is None

    def test_n_eq_2_either_works_or_none(self) -> None:
        result = bootstrap_ci([1.0, 2.0], seed=42)
        if result is not None:
            assert math.isfinite(result.ci_low)
            assert math.isfinite(result.ci_high)

    def test_zero_variance_returns_none_or_degenerate(self) -> None:
        """Constant samples → BCa returns NaN bounds → we return None."""
        result = bootstrap_ci([5.0] * 10, seed=42)
        # Either None or a degenerate single-point CI - never NaN.
        if result is not None:
            assert math.isfinite(result.ci_low)
            assert math.isfinite(result.ci_high)


class TestBootstrapCIScipyOracle:
    def test_matches_scipy_bca(self) -> None:
        my = bootstrap_ci(SAMPLE, seed=42, n_resamples=1000, method="bca")
        oracle = scipy_stats.bootstrap(
            (SAMPLE,),
            np.mean,
            n_resamples=1000,
            confidence_level=0.95,
            method="bca",
            random_state=42,
        )
        assert my.ci_low == pytest.approx(
            float(oracle.confidence_interval.low), abs=1e-9
        )
        assert my.ci_high == pytest.approx(
            float(oracle.confidence_interval.high), abs=1e-9
        )

    def test_matches_scipy_percentile(self) -> None:
        my = bootstrap_ci(
            SAMPLE, seed=42, n_resamples=1000, method="percentile"
        )
        oracle = scipy_stats.bootstrap(
            (SAMPLE,),
            np.mean,
            n_resamples=1000,
            confidence_level=0.95,
            method="percentile",
            random_state=42,
        )
        assert my.ci_low == pytest.approx(
            float(oracle.confidence_interval.low), abs=1e-9
        )
        assert my.ci_high == pytest.approx(
            float(oracle.confidence_interval.high), abs=1e-9
        )

    def test_matches_scipy_basic(self) -> None:
        my = bootstrap_ci(SAMPLE, seed=42, n_resamples=1000, method="basic")
        oracle = scipy_stats.bootstrap(
            (SAMPLE,),
            np.mean,
            n_resamples=1000,
            confidence_level=0.95,
            method="basic",
            random_state=42,
        )
        assert my.ci_low == pytest.approx(
            float(oracle.confidence_interval.low), abs=1e-9
        )
        assert my.ci_high == pytest.approx(
            float(oracle.confidence_interval.high), abs=1e-9
        )


class TestBootstrapCIStatistic:
    def test_custom_statistic_median(self) -> None:
        result = bootstrap_ci(SAMPLE, statistic=np.median, seed=42)
        assert result is not None
        assert result.point_estimate == pytest.approx(float(np.median(SAMPLE)))

    def test_resamples_parameter_passed(self) -> None:
        r1 = bootstrap_ci(SAMPLE, seed=42, n_resamples=500)
        r2 = bootstrap_ci(SAMPLE, seed=42, n_resamples=5000)
        assert r1.n_resamples == 500
        assert r2.n_resamples == 5000
