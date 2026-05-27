"""Unit tests for statistical significance functions in run_statistics.

Math is delegated to scipy.stats inside the production code; here we
verify the wrappers preserve scipy's results and we cross-check
hand-computed values from the math-verification report.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import scipy.stats as scipy_stats

from cli_modelarium.run_statistics import (
    SignificanceResult,
    bonferroni_correct,
    cohens_d,
    cohens_d_interpretation,
    compute_pairwise_significance,
    holm_correct,
    mann_whitney_u_test,
    welch_t_test,
)

# Hand-computed reference values from the math-verification report (§1.3).
SAMPLE_A = [8.4, 8.1, 8.6, 8.3, 8.5, 8.2, 8.7, 8.4, 8.3, 8.5]
SAMPLE_B = [7.9, 7.6, 8.1, 7.8, 7.7, 8.0, 7.5, 7.9, 7.8, 8.2]


# ===== Welch's t-test =====


class TestWelchTTest:
    def test_known_values_match_hand_computed(self) -> None:
        """Hand-verified: t=6.1279..., df=17.4803..., p=9.87e-6."""
        t, df, p = welch_t_test(SAMPLE_A, SAMPLE_B)
        assert abs(t - 6.1279461598) < 1e-4
        assert abs(df - 17.4803695150) < 1e-4
        assert abs(p - 9.8717e-6) < 1e-7

    def test_matches_scipy_oracle(self) -> None:
        """Our wrapper must produce the same numbers as scipy.stats.ttest_ind."""
        t, df, p = welch_t_test(SAMPLE_A, SAMPLE_B)
        oracle = scipy_stats.ttest_ind(SAMPLE_A, SAMPLE_B, equal_var=False)
        assert abs(t - float(oracle.statistic)) < 1e-12
        assert abs(p - float(oracle.pvalue)) < 1e-12

    def test_requires_two_samples_per_group(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            welch_t_test([1.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 2 samples"):
            welch_t_test([1.0, 2.0], [1.0])

    def test_swapping_arguments_flips_t_preserves_p(self) -> None:
        """t(a, b) = -t(b, a); p is unchanged (two-tailed)."""
        t_ab, _, p_ab = welch_t_test(SAMPLE_A, SAMPLE_B)
        t_ba, _, p_ba = welch_t_test(SAMPLE_B, SAMPLE_A)
        assert abs(t_ab + t_ba) < 1e-9
        assert abs(p_ab - p_ba) < 1e-9

    def test_equal_samples_give_zero_t(self) -> None:
        t, _, _ = welch_t_test([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        assert abs(t) < 1e-9


# ===== Mann-Whitney U =====


class TestMannWhitneyU:
    def test_matches_scipy_oracle(self) -> None:
        u, p = mann_whitney_u_test(SAMPLE_A, SAMPLE_B)
        oracle = scipy_stats.mannwhitneyu(
            SAMPLE_A, SAMPLE_B, alternative="two-sided", use_continuity=True
        )
        assert abs(u - float(oracle.statistic)) < 1e-12
        assert abs(p - float(oracle.pvalue)) < 1e-12

    def test_u_statistic_for_known_data(self) -> None:
        """The hand-counted U_a from the math-verification report is 98."""
        u, _ = mann_whitney_u_test(SAMPLE_A, SAMPLE_B)
        assert u == 98.0

    def test_requires_one_sample_per_group(self) -> None:
        with pytest.raises(ValueError, match="at least 1 sample"):
            mann_whitney_u_test([], [1.0, 2.0])


# ===== Cohen's d =====


class TestCohensD:
    def test_known_value_matches_hand_computed(self) -> None:
        """Hand-verified: d ≈ 2.7405 (large)."""
        d = cohens_d(SAMPLE_A, SAMPLE_B)
        assert d is not None
        assert abs(d - 2.7405) < 1e-3

    def test_symmetric_opposite_sign(self) -> None:
        d_ab = cohens_d(SAMPLE_A, SAMPLE_B)
        d_ba = cohens_d(SAMPLE_B, SAMPLE_A)
        assert d_ab is not None and d_ba is not None
        assert abs(d_ab + d_ba) < 1e-12

    def test_zero_variance_equal_means_returns_zero(self) -> None:
        assert cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 0.0

    def test_zero_variance_different_means_returns_none(self) -> None:
        assert cohens_d([5.0, 5.0, 5.0], [7.0, 7.0, 7.0]) is None

    def test_n_less_than_two_returns_none(self) -> None:
        assert cohens_d([5.0], [1.0, 2.0, 3.0]) is None
        assert cohens_d([1.0, 2.0, 3.0], [5.0]) is None
        assert cohens_d([], [1.0, 2.0]) is None


class TestCohensDInterpretation:
    @pytest.mark.parametrize(
        "d,expected",
        [
            (0.0, "negligible"),
            (0.19, "negligible"),
            (0.20, "small"),
            (0.49, "small"),
            (0.50, "medium"),
            (0.79, "medium"),
            (0.80, "large"),
            (2.50, "large"),
            (None, "undefined"),
        ],
    )
    def test_bands(self, d: float | None, expected: str) -> None:
        assert cohens_d_interpretation(d) == expected

    def test_uses_absolute_value(self) -> None:
        """Negative d should map to the same band as positive."""
        assert cohens_d_interpretation(-0.5) == "medium"
        assert cohens_d_interpretation(-2.0) == "large"
        assert cohens_d_interpretation(-0.1) == "negligible"


# ===== Bonferroni =====


class TestBonferroni:
    def test_basic_multiplies_and_caps(self) -> None:
        assert bonferroni_correct([0.01, 0.04, 0.03, 0.20]) == [
            0.04,
            0.16,
            0.12,
            0.80,
        ]

    def test_cap_at_one(self) -> None:
        assert bonferroni_correct([0.4, 0.5]) == [0.8, 1.0]

    def test_empty_returns_empty(self) -> None:
        assert bonferroni_correct([]) == []

    def test_explicit_n_comparisons_overrides_len(self) -> None:
        # Adjusting one p-value against 5 comparisons
        assert bonferroni_correct([0.01], n_comparisons=5) == [0.05]
        assert bonferroni_correct([0.05], n_comparisons=10) == [0.5]


# ===== Holm =====


class TestHolm:
    def test_basic_known_values(self) -> None:
        """From math-verification report:
        input [0.01, 0.04, 0.03, 0.20] -> [0.04, 0.09, 0.09, 0.20]
        """
        assert holm_correct([0.01, 0.04, 0.03, 0.20]) == [
            0.04,
            0.09,
            0.09,
            0.20,
        ]

    def test_empty_returns_empty(self) -> None:
        assert holm_correct([]) == []

    def test_monotone_enforcement(self) -> None:
        """In sorted order, adjusted p-values must be non-decreasing."""
        raw = [0.005, 0.04, 0.01, 0.03, 0.20]
        adjusted = holm_correct(raw)
        pairs = sorted(zip(raw, adjusted), key=lambda x: x[0])
        adj_in_sorted_order = [a for _, a in pairs]
        for i in range(1, len(adj_in_sorted_order)):
            assert adj_in_sorted_order[i] >= adj_in_sorted_order[i - 1]

    def test_single_value(self) -> None:
        # n=1, multiplier=1, no correction needed
        assert holm_correct([0.03]) == [0.03]

    def test_all_caps_at_one(self) -> None:
        assert holm_correct([0.5, 0.6, 0.7]) == [1.0, 1.0, 1.0]


# ===== compute_pairwise_significance =====


@dataclass
class _MockState:
    """Minimal StreamState-compatible object for testing."""

    model: str
    latency_ms: float | None = None
    output_tokens: int = 10
    cost_usd: float = 0.001
    error: str | None = None


class TestComputePairwiseSignificance:
    def test_returns_empty_for_fewer_than_two_models(self) -> None:
        states = {"only_model": [_MockState(model="only_model", latency_ms=100.0)]}
        assert compute_pairwise_significance(states, None) == []

    def test_two_models_one_pair(self) -> None:
        states = {
            "a": [
                _MockState(model="a", latency_ms=v)
                for v in [100, 110, 105, 108, 102]
            ],
            "b": [
                _MockState(model="b", latency_ms=v)
                for v in [120, 130, 125, 128, 122]
            ],
        }
        results = compute_pairwise_significance(
            states,
            None,
            metric="latency_ms",
            test="welch",
            correction="bonferroni",
            threshold=0.05,
        )
        assert len(results) == 1
        r = results[0]
        assert r.model_a == "a"
        assert r.model_b == "b"
        assert r.n_a == 5
        assert r.n_b == 5
        assert r.test_used == "welch_t_test"
        assert r.p_value is not None and r.p_value < 0.01  # well-separated samples
        assert r.effect_size is not None
        assert abs(r.effect_size) > 0.8  # large effect
        assert r.significant_at_threshold

    def test_three_models_three_pairs(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=v) for v in [100, 105, 110, 102, 108]],
            "b": [_MockState(model="b", latency_ms=v) for v in [200, 210, 205, 208, 202]],
            "c": [_MockState(model="c", latency_ms=v) for v in [300, 305, 310, 308, 302]],
        }
        results = compute_pairwise_significance(states, None)
        assert len(results) == 3
        pairs = {(r.model_a, r.model_b) for r in results}
        assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}
        for r in results:
            assert r.n_comparisons == 3

    def test_correction_bonferroni_vs_holm_vs_none(self) -> None:
        """Bonferroni is most conservative; Holm in between; none least."""
        states = {
            "a": [_MockState(model="a", latency_ms=v) for v in [100, 102, 105, 103, 101]],
            "b": [_MockState(model="b", latency_ms=v) for v in [110, 112, 115, 113, 111]],
            "c": [_MockState(model="c", latency_ms=v) for v in [120, 122, 125, 123, 121]],
        }
        none_r = compute_pairwise_significance(states, None, correction="none")
        bonf_r = compute_pairwise_significance(states, None, correction="bonferroni")
        holm_r = compute_pairwise_significance(states, None, correction="holm")
        # Each pair: bonf >= holm >= raw
        for n, b, h in zip(none_r, bonf_r, holm_r):
            assert n.p_value is not None
            assert b.p_value_corrected is not None
            assert h.p_value_corrected is not None
            assert b.p_value_corrected >= h.p_value_corrected - 1e-12
            assert h.p_value_corrected >= n.p_value - 1e-12

    def test_insufficient_samples_skips_test(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=100.0)],  # only 1
            "b": [_MockState(model="b", latency_ms=v) for v in [100, 110, 120]],
        }
        results = compute_pairwise_significance(states, None)
        assert len(results) == 1
        assert results[0].test_used == "insufficient_samples"
        assert results[0].p_value is None
        assert not results[0].significant_at_threshold

    def test_zero_variance_equal_means_trivial(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=100.0) for _ in range(5)],
            "b": [_MockState(model="b", latency_ms=100.0) for _ in range(5)],
        }
        results = compute_pairwise_significance(states, None)
        assert len(results) == 1
        assert results[0].test_used == "trivial"
        assert results[0].p_value == 1.0
        assert results[0].effect_size == 0.0
        assert not results[0].significant_at_threshold

    def test_zero_variance_different_means_marked_zero_variance(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=100.0) for _ in range(5)],
            "b": [_MockState(model="b", latency_ms=200.0) for _ in range(5)],
        }
        results = compute_pairwise_significance(states, None)
        assert len(results) == 1
        assert results[0].test_used == "zero_variance"
        assert results[0].p_value is None
        assert results[0].effect_size is None

    def test_mann_whitney_branch(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=v) for v in [100, 110, 105, 108, 102]],
            "b": [_MockState(model="b", latency_ms=v) for v in [120, 130, 125, 128, 122]],
        }
        results = compute_pairwise_significance(states, None, test="mann-whitney")
        assert results[0].test_used == "mann_whitney_u"
        assert results[0].degrees_of_freedom is None
        assert results[0].p_value is not None

    def test_failed_states_excluded(self) -> None:
        """States with error != None should be filtered out."""
        states = {
            "a": [_MockState(model="a", latency_ms=v) for v in [100, 105, 110, 102, 108]]
            + [_MockState(model="a", error="api error")],
            "b": [_MockState(model="b", latency_ms=v) for v in [200, 205, 210, 202, 208]]
            + [_MockState(model="b", error="api error")],
        }
        results = compute_pairwise_significance(states, None)
        # 5 successful in each group, error rows excluded
        assert results[0].n_a == 5
        assert results[0].n_b == 5

    def test_metric_output_tokens(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=100.0, output_tokens=10) for _ in range(5)],
            "b": [_MockState(model="b", latency_ms=100.0, output_tokens=20) for _ in range(5)],
        }
        results = compute_pairwise_significance(states, None, metric="output_tokens")
        assert results[0].metric == "output_tokens"
        assert results[0].mean_a == 10.0
        assert results[0].mean_b == 20.0

    def test_metric_cost_usd(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=100.0, cost_usd=0.001 + i * 0.0001)
                  for i in range(5)],
            "b": [_MockState(model="b", latency_ms=100.0, cost_usd=0.002 + i * 0.0001)
                  for i in range(5)],
        }
        results = compute_pairwise_significance(states, None, metric="cost_usd")
        assert results[0].metric == "cost_usd"
        assert results[0].mean_a < results[0].mean_b

    def test_score_metric_with_judge_results(self) -> None:
        states = {
            "a": [_MockState(model="a", latency_ms=100.0) for _ in range(5)],
            "b": [_MockState(model="b", latency_ms=100.0) for _ in range(5)],
        }

        class _MockJR:
            def __init__(self, score: float, state_id: int) -> None:
                self.average_score = score
                self._state_id = state_id

        judge_results = [
            _MockJR(7.0 + i * 0.1, id(s)) for i, s in enumerate(states["a"])
        ] + [_MockJR(5.0 + i * 0.1, id(s)) for i, s in enumerate(states["b"])]

        results = compute_pairwise_significance(
            states, judge_results, metric="score"
        )
        assert len(results) == 1
        assert results[0].metric == "score"
        assert results[0].mean_a > results[0].mean_b  # group a scored higher

    def test_six_models_full_pairwise(self) -> None:
        """6 models => 15 pairs."""
        models = ["m" + str(i) for i in range(6)]
        states = {
            m: [_MockState(model=m, latency_ms=100.0 + (i + 1) * 10 + j) for j in range(5)]
            for i, m in enumerate(models)
        }
        results = compute_pairwise_significance(states, None)
        assert len(results) == 15  # C(6,2) = 15
        assert all(r.n_comparisons == 15 for r in results)


# ===== SignificanceResult dataclass shape =====


class TestSignificanceResultShape:
    def test_all_fields_present(self) -> None:
        """Smoke test: the dataclass exposes the expected attributes."""
        states = {
            "a": [_MockState(model="a", latency_ms=v) for v in [100, 110, 105, 108, 102]],
            "b": [_MockState(model="b", latency_ms=v) for v in [200, 210, 205, 208, 202]],
        }
        results = compute_pairwise_significance(states, None)
        r = results[0]
        expected_fields = {
            "model_a", "model_b", "metric", "n_a", "n_b",
            "mean_a", "mean_b", "stdev_a", "stdev_b",
            "test_used", "test_statistic", "degrees_of_freedom",
            "p_value", "p_value_corrected", "correction_method",
            "n_comparisons", "effect_size", "effect_size_interpretation",
            "threshold", "significant_at_threshold",
        }
        for field in expected_fields:
            assert hasattr(r, field), f"Missing field: {field}"
        assert isinstance(r, SignificanceResult)
