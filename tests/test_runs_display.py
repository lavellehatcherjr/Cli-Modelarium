"""Tests for the display and output-formatter paths with --runs > 1.

KEY CONSTRAINT: When runs == 1, output MUST be byte-identical to v0.1.0.
We verify this with a head-to-head diff between "no --runs" and "--runs 1".
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.cli import main as cli_main
from cli_modelarium.output_formatters import (
    BatchResult,
    _format_csv,
    _format_json,
    _format_markdown,
)
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


class _RecordingProvider(BaseProvider):
    """Returns a preset CompletionResult."""

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
            on_chunk("Paris")
        return CompletionResult(
            output="Paris",
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


def _make_result(
    *,
    prompt_id: str = "p1",
    model: str = "gpt-5.5",
    temperature: float = 0.0,
    output: str = "Paris",
    cost_usd: float = 0.001,
    latency_ms: float = 100.0,
    run_index: int = 0,
    error: str | None = None,
) -> BatchResult:
    return BatchResult(
        prompt_id=prompt_id,
        prompt="capital?",
        system=None,
        model=model,
        temperature=temperature,
        latency_ms=latency_ms,
        ttft_ms=50.0,
        input_tokens=10,
        output_tokens=5,
        cached_tokens=0,
        cost_usd=cost_usd,
        output=output,
        error=error,
        retries=0,
        run_index=run_index,
    )


class TestRunsEqualsOneOutputUnchanged:
    """Backward-compat regression: runs=1 output must equal v0.1.0 output."""

    def test_csv_format_byte_identical_when_runs_one(self) -> None:
        results = [_make_result()]
        baseline = _format_csv(results)
        with_runs_one = _format_csv(results, runs=1)
        assert baseline == with_runs_one

    def test_json_format_byte_identical_when_runs_one(self) -> None:
        results = [_make_result()]
        baseline = _format_json(results)
        with_runs_one = _format_json(results, runs=1)
        assert baseline == with_runs_one
        # Confirm no run_index/total_runs/stats_by_cell leaked in.
        parsed = json.loads(baseline)
        assert "run_index" not in parsed["results"][0]
        assert "total_runs" not in parsed
        assert "stats_by_cell" not in parsed

    def test_markdown_format_byte_identical_when_runs_one(self) -> None:
        results = [_make_result()]
        baseline = _format_markdown(results)
        with_runs_one = _format_markdown(results, runs=1)
        assert baseline == with_runs_one
        # No "Per-cell statistical summary" header.
        assert "Per-cell statistical summary" not in baseline


class TestRunsAboveOneOutputExtended:
    def test_csv_includes_run_index_when_runs_above_one(self) -> None:
        results = [
            _make_result(prompt_id="p1", run_index=0),
            _make_result(prompt_id="p1", run_index=1),
        ]
        out = _format_csv(results, runs=2)
        reader = csv.DictReader(io.StringIO(out))
        assert "run_index" in reader.fieldnames
        rows = list(reader)
        assert rows[0]["run_index"] == "0"
        assert rows[1]["run_index"] == "1"

    def test_json_includes_total_runs_and_stats_when_runs_above_one(self) -> None:
        results = [
            _make_result(run_index=0, latency_ms=100.0),
            _make_result(run_index=1, latency_ms=110.0),
            _make_result(run_index=2, latency_ms=105.0),
        ]
        out = _format_json(results, runs=3)
        parsed = json.loads(out)
        assert parsed["total_runs"] == 3
        assert "stats_by_cell" in parsed
        assert len(parsed["stats_by_cell"]) == 1
        cell = parsed["stats_by_cell"][0]
        assert cell["n_runs"] == 3
        assert cell["n_succeeded"] == 3
        assert cell["mode_output"] == "Paris"
        # Per-result run_index appears too.
        assert parsed["results"][0]["run_index"] == 0
        assert parsed["results"][1]["run_index"] == 1

    def test_markdown_includes_stats_section_when_runs_above_one(self) -> None:
        results = [
            _make_result(run_index=0),
            _make_result(run_index=1),
            _make_result(run_index=2),
        ]
        out = _format_markdown(results, runs=3)
        assert "Per-cell statistical summary" in out
        assert "OK/Fail" in out
        assert "CV" in out


class TestRunsAndOutputFileIntegration:
    def test_runs_writes_json_with_stats(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        out = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--runs",
                "3",
                "--output",
                str(out),
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(out.read_text())
        assert parsed["total_runs"] == 3
        assert "stats_by_cell" in parsed
        assert parsed["total_results"] == 3

    def test_runs_one_writes_json_with_v0_1_0_schema(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        """When runs == 1, the JSON schema must not gain runs-related fields.

        Two CLI invocations can't be byte-identical because orchestrator-level
        TTFT is measured wall-clock; the schema-level invariant (no new keys
        at runs == 1) is tested via `_format_json` byte-equality in
        TestRunsEqualsOneOutputUnchanged. Here we verify the full end-to-end
        CLI path with --runs 1 produces v0.1.0-shaped output.
        """
        out = tmp_path / "runs_one.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--runs",
                "1",
                "--output",
                str(out),
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(out.read_text())
        # No runs-related top-level fields.
        assert "total_runs" not in parsed
        assert "stats_by_cell" not in parsed
        # No run_index in per-result dicts.
        assert "run_index" not in parsed["results"][0]

    def test_runs_writes_csv_with_run_index(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        out = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "q",
                "--models",
                "gpt-5.5",
                "--runs",
                "3",
                "--output",
                str(out),
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.output
        text = out.read_text()
        reader = csv.DictReader(io.StringIO(text))
        assert "run_index" in reader.fieldnames
        rows = list(reader)
        assert len(rows) == 3
        assert sorted(r["run_index"] for r in rows) == ["0", "1", "2"]
