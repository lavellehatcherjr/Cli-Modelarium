"""CLI-level tests for `compare --judge ... --output ...`.

Verifies that the judge results flow through the StreamState -> BatchResult
adapter into the file output formats (JSON and CSV).
"""

from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.cli import main as cli_main
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


class _DualPurposeProvider(BaseProvider):
    """Fake provider that handles both main-call and judge prompts.

    Mirrors `_DualPurposeProvider` from tests/test_cli_judging.py.
    """

    JUDGE_MARKER = "Respond with ONLY a JSON object"

    def __init__(self, judge_response: str | None = None) -> None:
        self.name = "fake"
        self.calls: list[dict[str, Any]] = []
        self._judge_response = judge_response or '{"score": 8, "reasoning": "decent"}'

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
        is_judge_call = self.JUDGE_MARKER in prompt
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "is_judge_call": is_judge_call,
            }
        )
        output = self._judge_response if is_judge_call else f"answer for {model}"
        if on_chunk is not None:
            on_chunk(output)
        return CompletionResult(
            output=output,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001 if is_judge_call else 0.001,
            latency_ms=42.0,
            ttft_ms=12.0,
            model=model,
            provider="fake",
            temperature=temperature,
        )


@pytest.fixture
def dual_provider(monkeypatch: pytest.MonkeyPatch) -> _DualPurposeProvider:
    fake = _DualPurposeProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    for env, value in (
        ("OPENAI_API_KEY", "sk-proj-test1234567890abcdefghi"),
        ("ANTHROPIC_API_KEY", "sk-ant-test1234567890abcdefghi"),
    ):
        monkeypatch.setenv(env, value)
    return fake


class TestCompareJudgeOutput:
    def test_json_output_contains_judge_scores(
        self, dual_provider: _DualPurposeProvider, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "what is X?",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--output",
                str(output),
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(output.read_text(encoding="utf-8"))
        # judge_cost_usd appears at the top level when judging was active.
        assert "judge_cost_usd" in payload
        row = payload["results"][0]
        assert "judges" in row
        assert len(row["judges"]) == 1
        assert row["judges"][0]["score"] == 8
        assert row["judge_score_avg"] == 8

    def test_csv_output_contains_judge_columns(
        self, dual_provider: _DualPurposeProvider, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "what is X?",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--output",
                str(output),
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        # judge_score_avg / judge_count are canonical batch CSV columns and
        # should now be populated for compare too.
        assert rows[0]["judge_score_avg"] == "8"
        assert rows[0]["judge_count"] == "1"
