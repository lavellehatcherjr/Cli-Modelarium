"""Tests for run_statistics module.

Covers the RunStats dataclass and the compute_run_stats function.
"""

from __future__ import annotations

from cli_modelarium.run_statistics import (
    compute_run_stats,
    group_states_by_cell,
)
from cli_modelarium.streaming import StreamState


def _make_state(
    *,
    model: str = "gpt-5.5",
    output: str = "Paris",
    latency_ms: float | None = 100.0,
    ttft_ms: float | None = 50.0,
    output_tokens: int = 10,
    cost_usd: float = 0.001,
    error: str | None = None,
    run_index: int = 0,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> StreamState:
    """Build a StreamState for testing."""
    state = StreamState(
        model=model,
        provider_name="openai",
        temperature=temperature,
        system_prompt=system_prompt,
        run_index=run_index,
    )
    state.text = output
    state.latency_ms = latency_ms
    state.ttft_ms = ttft_ms
    state.output_tokens = output_tokens
    state.cost_usd = cost_usd
    state.error = error
    return state


class TestRunStatsEmpty:
    def test_empty_list(self) -> None:
        stats = compute_run_stats([])
        assert stats.n_runs == 0
        assert stats.n_succeeded == 0
        assert stats.n_failed == 0
        assert stats.latency_mean_ms is None
        assert stats.latency_stdev_ms is None
        assert stats.latency_cv is None
        assert stats.cost_total_usd == 0.0
        assert stats.cost_mean_usd is None
        assert stats.unique_outputs == 0
        assert stats.mode_output is None
        assert stats.mode_count == 0
        assert stats.output_diversity == 0.0


class TestRunStatsSingleRun:
    def test_single_run_stdev_is_none(self) -> None:
        states = [_make_state(latency_ms=100.0)]
        stats = compute_run_stats(states)
        assert stats.n_runs == 1
        assert stats.n_succeeded == 1
        assert stats.latency_mean_ms == 100.0
        assert stats.latency_median_ms == 100.0
        assert stats.latency_stdev_ms is None
        assert stats.latency_cv is None
        assert stats.output_tokens_stdev is None


class TestRunStatsMultipleRuns:
    def test_basic_stats(self) -> None:
        states = [
            _make_state(output="Paris", latency_ms=100.0, cost_usd=0.001),
            _make_state(output="Paris", latency_ms=110.0, cost_usd=0.0012),
            _make_state(output="Paris", latency_ms=105.0, cost_usd=0.0011),
        ]
        stats = compute_run_stats(states)
        assert stats.n_runs == 3
        assert stats.n_succeeded == 3
        assert stats.n_failed == 0
        assert stats.latency_mean_ms == 105.0
        assert stats.latency_median_ms == 105.0
        assert stats.latency_stdev_ms is not None
        assert 4 < stats.latency_stdev_ms < 6
        assert stats.latency_cv is not None
        assert 0.04 < stats.latency_cv < 0.06
        assert abs(stats.cost_total_usd - 0.0033) < 1e-9
        assert stats.unique_outputs == 1
        assert stats.mode_output == "Paris"
        assert stats.mode_count == 3
        assert stats.output_diversity == 1 / 3


class TestRunStatsAllFailed:
    def test_all_failed(self) -> None:
        states = [
            _make_state(error="provider error 1"),
            _make_state(error="provider error 2"),
        ]
        stats = compute_run_stats(states)
        assert stats.n_runs == 2
        assert stats.n_succeeded == 0
        assert stats.n_failed == 2
        assert stats.latency_mean_ms is None
        assert stats.latency_stdev_ms is None
        assert stats.latency_cv is None
        assert stats.cost_total_usd == 0.0
        assert stats.cost_mean_usd is None
        assert stats.unique_outputs == 0
        assert stats.mode_output is None
        assert stats.output_diversity == 0.0


class TestRunStatsPartialFailures:
    def test_partial_failures(self) -> None:
        states = [
            _make_state(output="Paris", latency_ms=100.0),
            _make_state(output="Paris", latency_ms=110.0),
            _make_state(error="provider error"),
        ]
        stats = compute_run_stats(states)
        assert stats.n_runs == 3
        assert stats.n_succeeded == 2
        assert stats.n_failed == 1
        assert stats.latency_mean_ms == 105.0
        assert stats.latency_stdev_ms is not None
        assert stats.mode_output == "Paris"
        assert stats.mode_count == 2


class TestRunStatsAllIdentical:
    def test_all_identical_outputs(self) -> None:
        states = [_make_state(output="Paris") for _ in range(5)]
        stats = compute_run_stats(states)
        assert stats.unique_outputs == 1
        assert stats.mode_output == "Paris"
        assert stats.mode_count == 5
        assert stats.output_diversity == 0.2


class TestRunStatsAllUnique:
    def test_all_unique_outputs_no_mode(self) -> None:
        states = [_make_state(output=f"Output {i}") for i in range(5)]
        stats = compute_run_stats(states)
        assert stats.unique_outputs == 5
        assert stats.mode_output is None
        assert stats.mode_count == 0
        assert stats.output_diversity == 1.0


class TestRunStatsEmptyOutputs:
    def test_with_empty_outputs(self) -> None:
        states = [
            _make_state(output="Paris"),
            _make_state(output=""),
            _make_state(output="Paris"),
            _make_state(output=""),
            _make_state(output="Paris"),
        ]
        stats = compute_run_stats(states)
        assert stats.unique_outputs == 2
        assert stats.mode_output == "Paris"
        assert stats.mode_count == 3
        assert stats.output_diversity == 0.4


class TestRunStatsCV:
    def test_cv_calculation(self) -> None:
        """Coefficient of variation = stdev / mean."""
        states = [
            _make_state(latency_ms=100.0),
            _make_state(latency_ms=120.0),
            _make_state(latency_ms=80.0),
            _make_state(latency_ms=110.0),
        ]
        stats = compute_run_stats(states)
        assert stats.latency_mean_ms == 102.5
        assert stats.latency_cv is not None
        assert 0.15 < stats.latency_cv < 0.18


class TestRunStatsModeTie:
    def test_mode_with_tie(self) -> None:
        """When 2 outputs tie for mode, Counter returns the first inserted."""
        states = [
            _make_state(output="A"),
            _make_state(output="B"),
            _make_state(output="A"),
            _make_state(output="B"),
        ]
        stats = compute_run_stats(states)
        # Both A and B appear twice. Either is a valid mode.
        assert stats.mode_output in ("A", "B")
        assert stats.mode_count == 2
        assert stats.unique_outputs == 2
        assert stats.output_diversity == 0.5


class TestRunStatsCostMean:
    def test_cost_mean_with_successes(self) -> None:
        states = [
            _make_state(cost_usd=0.001),
            _make_state(cost_usd=0.002),
            _make_state(cost_usd=0.003),
        ]
        stats = compute_run_stats(states)
        assert abs(stats.cost_total_usd - 0.006) < 1e-9
        assert stats.cost_mean_usd is not None
        assert abs(stats.cost_mean_usd - 0.002) < 1e-9


class TestGroupStatesByCell:
    def test_groups_by_cell_correctly(self) -> None:
        states = [
            _make_state(model="gpt-5.5", temperature=0.0, run_index=0),
            _make_state(model="gpt-5.5", temperature=0.0, run_index=1),
            _make_state(model="gpt-5.5", temperature=0.7, run_index=0),
            _make_state(model="claude-opus-4-7", temperature=0.0, run_index=0),
        ]
        groups = group_states_by_cell(states)
        assert len(groups) == 3
        assert len(groups[("gpt-5.5", 0.0, None)]) == 2
        assert len(groups[("gpt-5.5", 0.7, None)]) == 1
        assert len(groups[("claude-opus-4-7", 0.0, None)]) == 1

    def test_groups_sorted_by_run_index(self) -> None:
        states = [
            _make_state(run_index=2),
            _make_state(run_index=0),
            _make_state(run_index=1),
        ]
        groups = group_states_by_cell(states)
        cell_states = list(groups.values())[0]
        assert [s.run_index for s in cell_states] == [0, 1, 2]

    def test_groups_distinguish_system_prompt(self) -> None:
        states = [
            _make_state(system_prompt="A"),
            _make_state(system_prompt="B"),
            _make_state(system_prompt=None),
        ]
        groups = group_states_by_cell(states)
        assert len(groups) == 3
