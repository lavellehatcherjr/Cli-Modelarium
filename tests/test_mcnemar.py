"""Tests for McNemar's test (F1 — manual implementation, NOT
scipy.stats.chi2_contingency).

These tests lock in the correct McNemar math by:
1. Verifying exact binomial path matches scipy.stats.binomtest.
2. Verifying Edwards-corrected path matches scipy.stats.chi2.sf.
3. Asserting that McNemar p-values differ DRAMATICALLY from
   chi2_contingency on the full 2x2 table (different hypotheses).
"""

from __future__ import annotations

import pytest
import scipy.stats as scipy_stats

from cli_modelarium.run_statistics import mcnemar_test


class TestMcNemarExact:
    def test_known_b15_c5_exact(self) -> None:
        """From verification report: b=15, c=5 → exact p ~ 0.041."""
        chi2, p = mcnemar_test(15, 5, exact=True)
        assert chi2 is None
        assert p == pytest.approx(0.041, abs=0.005)

    def test_matches_scipy_binomtest(self) -> None:
        chi2, p = mcnemar_test(15, 5, exact=True)
        oracle = scipy_stats.binomtest(15, 20, p=0.5)
        assert p == pytest.approx(float(oracle.pvalue), abs=1e-12)

    def test_auto_exact_for_small_n(self) -> None:
        """n_discordant < 25 triggers exact path even without exact=True."""
        chi2, p_auto = mcnemar_test(15, 5)
        _, p_explicit = mcnemar_test(15, 5, exact=True)
        assert chi2 is None
        assert p_auto == pytest.approx(p_explicit, abs=1e-12)


class TestMcNemarAsymptotic:
    def test_large_discordant_uses_edwards(self) -> None:
        chi2, _ = mcnemar_test(60, 40)
        # (|60-40| - 1)^2 / 100 = 19^2/100 = 3.61
        assert chi2 == pytest.approx(3.61, abs=0.01)

    def test_edwards_matches_chi2_sf(self) -> None:
        chi2, p = mcnemar_test(60, 40)
        oracle_p = float(scipy_stats.chi2.sf(chi2, df=1))
        assert p == pytest.approx(oracle_p, abs=1e-12)

    def test_n_discordant_25_threshold(self) -> None:
        # n_discordant = 25 -> Edwards (>= 25, not exact)
        chi2_25, _ = mcnemar_test(13, 12)
        # n_discordant = 24 -> exact
        chi2_24, _ = mcnemar_test(13, 11)
        assert chi2_25 is not None  # Edwards path
        assert chi2_24 is None  # exact path


class TestMcNemarNotIndependence:
    def test_correct_test_vs_chi2_contingency(self) -> None:
        """LOCKED-IN CORRECTNESS TEST.

        Verification report example: 2x2 = [[40, 15], [5, 40]].
        - chi2_contingency on full table: chi2=35.5, p=2.5e-09 (independence)
        - True McNemar(b=15, c=5): p ~= 0.041 (paired-marginal change)

        Without F1 fix, this test fails dramatically. With F1 fix, the two
        p-values are orders of magnitude apart - confirming they test
        different hypotheses.
        """
        wrong_chi2, wrong_p, _, _ = scipy_stats.chi2_contingency(
            [[40, 15], [5, 40]], correction=True
        )
        _, mcnemar_p = mcnemar_test(15, 5, exact=True)

        assert wrong_p < 1e-6  # chi2_contingency declares it ultra-significant
        assert mcnemar_p > 0.01  # McNemar does not
        assert abs(wrong_p - mcnemar_p) > 0.01  # Off by orders of magnitude


class TestMcNemarEdgeCases:
    def test_zero_discordant_returns_no_test(self) -> None:
        chi2, p = mcnemar_test(0, 0)
        assert chi2 is None
        assert p == 1.0

    def test_only_b_discordant(self) -> None:
        # b=5, c=0 - all change in one direction
        chi2, p = mcnemar_test(5, 0)
        assert chi2 is None  # exact (n_disc=5 < 25)
        assert p < 0.1  # significant when all discordants go one way

    def test_only_c_discordant_same_as_b(self) -> None:
        # Test symmetry: mcnemar(5, 0) == mcnemar(0, 5)
        _, p_b = mcnemar_test(5, 0)
        _, p_c = mcnemar_test(0, 5)
        assert p_b == pytest.approx(p_c, abs=1e-12)

    def test_balanced_discordant_p_is_one(self) -> None:
        # b = c → no evidence of asymmetric change → p=1.0
        _, p = mcnemar_test(10, 10)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_exact_flag_overrides_auto_threshold(self) -> None:
        # n_discordant=100 would auto-use Edwards, but exact=True forces
        # the binomial test.
        chi2, p = mcnemar_test(60, 40, exact=True)
        assert chi2 is None
        # Should match binomtest(60, 100)
        oracle = scipy_stats.binomtest(60, 100, p=0.5)
        assert p == pytest.approx(float(oracle.pvalue), abs=1e-12)
