"""CLI-level tests for `compare --check-hallucination --output ...`.

Verifies hallucination risk levels flow through the file output formats.
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


class _HallucinationFake(BaseProvider):
    """Returns the right JSON shape for hallucination judge prompts."""

    JUDGE_MARKER = "risk_level"

    def __init__(self, judge_response: str | None = None) -> None:
        self.name = "fake"
        self.calls: list[dict[str, Any]] = []
        self._judge_response = judge_response or (
            '{"score": 4, "risk_level": "Medium", "reasoning": "some doubt"}'
        )

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
        is_judge = self.JUDGE_MARKER in prompt
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "is_judge_call": is_judge,
            }
        )
        output = self._judge_response if is_judge else f"answer from {model}"
        if on_chunk is not None:
            on_chunk(output)
        return CompletionResult(
            output=output,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001 if is_judge else 0.001,
            latency_ms=50.0,
            ttft_ms=10.0,
            model=model,
            provider="fake",
            temperature=temperature,
        )


@pytest.fixture
def hallu_fake(monkeypatch: pytest.MonkeyPatch) -> _HallucinationFake:
    fake = _HallucinationFake()
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


class TestCompareHallucinationOutput:
    def test_csv_output_contains_hallucination_risk(
        self, hallu_fake: _HallucinationFake, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "what year did X happen?",
                "--models",
                "gpt-5.5",
                "--check-hallucination",
                "--expected-facts",
                "X happened in 1969",
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
        # hallucination_risk is column 17 in CSV_COLUMNS; should be Medium
        # because the fake judge always returns risk_level=Medium.
        assert rows[0]["hallucination_risk"] == "Medium"

    def test_json_output_contains_aggregated_risk_level(
        self, hallu_fake: _HallucinationFake, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "what year did X happen?",
                "--models",
                "gpt-5.5",
                "--check-hallucination",
                "--expected-facts",
                "X happened in 1969",
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
        row = payload["results"][0]
        assert row["hallucination_risk"] == "Medium"
        assert row["judges"][0]["risk_level"] == "Medium"
