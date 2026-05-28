"""CLI-level tests for --check-hallucination on compare and batch.

Uses a smart fake provider that detects hallucination-judge prompts (by
the "risk_level" marker in JUDGE_PROMPT_TEMPLATE / HALLUCINATION_CRITERIA_BASE)
and returns a properly-shaped hallucination JSON response.
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

# ===== fake provider that responds with hallucination JSON =====


class _HallucinationFake(BaseProvider):
    """Returns the right JSON shape for hallucination judge prompts.

    Detects judge prompts by looking for the "risk_level" marker which only
    appears in the hallucination template. Otherwise returns a normal answer.
    """

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
                "temperature": temperature,
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
def hallu_provider(monkeypatch: pytest.MonkeyPatch) -> _HallucinationFake:
    fake = _HallucinationFake()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    # Judge validation reads is_key_configured directly; set fake env vars.
    for env, value in (
        ("OPENAI_API_KEY", "sk-proj-test1234567890abcdefghi"),
        ("ANTHROPIC_API_KEY", "sk-ant-test1234567890abcdefghi"),
        ("GOOGLE_API_KEY", "AIzaTestabcdef1234567890abcdef1234567"),
    ):
        monkeypatch.setenv(env, value)
    return fake


# ===== flag combination validation =====


class TestFlagCombinations:
    def test_check_hallucination_without_judge_rejected(
        self, hallu_provider: _HallucinationFake
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--check-hallucination",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code != 0
        # Error message references the missing flag.
        assert "--judge" in result.output

    def test_expected_facts_without_check_hallucination_rejected(
        self, hallu_provider: _HallucinationFake
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--expected-facts",
                "fact one,fact two",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code != 0
        # Either explicitly mentions --check-hallucination OR refuses
        # with a clear message.
        assert "--check-hallucination" in result.output or "require" in result.output.lower()

    def test_facts_and_facts_file_together_rejected(
        self, hallu_provider: _HallucinationFake, tmp_path: Path
    ) -> None:
        facts_file = tmp_path / "f.txt"
        facts_file.write_text("a fact", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--expected-facts",
                "a",
                "--expected-facts-file",
                str(facts_file),
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_template_and_facts_together_rejected(
        self, hallu_provider: _HallucinationFake, tmp_path: Path
    ) -> None:
        template = tmp_path / "tpl.txt"
        template.write_text("custom rubric", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--expected-facts",
                "fact one",
                "--hallucination-template",
                str(template),
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# ===== --check-hallucination happy paths =====


class TestCheckHallucination:
    def test_with_judge_works_end_to_end(self, hallu_provider: _HallucinationFake) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "Tell me about the Eiffel Tower",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code == 0, result.output
        # The Hallucination Risk header replaces Score. Rich may truncate the
        # long header in a narrow CliRunner terminal; check that "Risk" is
        # in the header and "Score" is NOT (the latter would mean we didn't
        # swap the column at all).
        flattened = "".join(result.output.split())
        assert "Risk" in flattened
        assert "Score" not in flattened
        # The fake returned risk_level "Medium".
        assert "Medium" in result.output

    def test_expected_facts_threaded_into_judge_prompt(
        self, hallu_provider: _HallucinationFake
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "Tell me about the Eiffel Tower",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--expected-facts",
                "Built 1887-1889,Located in Paris",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code == 0, result.output
        judge_call = next(c for c in hallu_provider.calls if c["is_judge_call"])
        assert "Built 1887-1889" in judge_call["prompt"]
        assert "Located in Paris" in judge_call["prompt"]

    def test_expected_facts_file(self, hallu_provider: _HallucinationFake, tmp_path: Path) -> None:
        facts_file = tmp_path / "facts.txt"
        facts_file.write_text(
            "# header comment\nDesigned by Gustave Eiffel\nOriginal height 300m\n", encoding="utf-8"
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "Tell me about the Eiffel Tower",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--expected-facts-file",
                str(facts_file),
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code == 0, result.output
        judge_call = next(c for c in hallu_provider.calls if c["is_judge_call"])
        assert "Designed by Gustave Eiffel" in judge_call["prompt"]
        assert "Original height 300m" in judge_call["prompt"]
        # Comment NOT propagated.
        assert "header comment" not in judge_call["prompt"]

    def test_footer_shows_facts_count(self, hallu_provider: _HallucinationFake) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--expected-facts",
                "a,b,c",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code == 0
        assert "3 reference fact" in result.output


# ===== panel mode aggregation =====


class TestPanelAggregation:
    def test_panel_aggregates_worst_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three judges: Low, Medium, High - panel must show High."""
        # Custom fake that returns different risk levels per judge model.
        responses = {
            "claude-opus-4-7": '{"score": 9, "risk_level": "Low", "reasoning": "x"}',
            "gemini-3.1-pro": '{"score": 5, "risk_level": "Medium", "reasoning": "x"}',
            "grok-4.3": '{"score": 2, "risk_level": "High", "reasoning": "x"}',
        }

        class _PanelFake(BaseProvider):
            def __init__(self) -> None:
                self.name = "fake"
                self.calls: list[dict[str, Any]] = []

            async def stream(self, *a: Any, **k: Any) -> AsyncIterator[str]:
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
                self.calls.append({"prompt": prompt, "model": model})
                is_judge = "risk_level" in prompt
                output = responses.get(model, f"answer from {model}") if is_judge else "answer"
                if on_chunk is not None:
                    on_chunk(output)
                return CompletionResult(
                    output=output,
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.0001,
                    latency_ms=50.0,
                    ttft_ms=10.0,
                    model=model,
                    provider="fake",
                    temperature=temperature,
                )

        fake = _PanelFake()
        monkeypatch.setattr(
            "cli_modelarium.cli._get_provider_instance",
            lambda name, **_kwargs: fake,
        )
        for env, value in (
            ("OPENAI_API_KEY", "sk-proj-test1234567890abcdefghi"),
            ("ANTHROPIC_API_KEY", "sk-ant-test1234567890abcdefghi"),
            ("GOOGLE_API_KEY", "AIzaTestabcdef1234567890abcdef1234567"),
            ("XAI_API_KEY", "xai-test1234567890abcdefghi"),
        ):
            monkeypatch.setenv(env, value)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judges",
                "claude-opus-4-7,gemini-3.1-pro,grok-4.3",
                "--check-hallucination",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        # The aggregated risk_level is "High" (worst-wins).
        assert "High" in result.output


# ===== ToS disclosure =====


class TestToSDisclosure:
    def test_hallucination_extension_shown_with_check(
        self, hallu_provider: _HallucinationFake
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0
        # Standard judge ToS is still shown.
        assert "Judge scores are for evaluation only" in result.output
        # Extended hallucination notice is appended.
        assert "guidance, not ground truth" in result.output

    def test_no_judge_tos_suppresses_both(self, hallu_provider: _HallucinationFake) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code == 0
        assert "Judge scores are for evaluation only" not in result.output
        assert "guidance, not ground truth" not in result.output


# ===== batch + hallucination output =====


class TestBatchHallucinationOutput:
    def test_csv_has_hallucination_risk_column(
        self, hallu_provider: _HallucinationFake, tmp_path: Path
    ) -> None:
        prompts_file = tmp_path / "p.json"
        prompts_file.write_text(json.dumps([{"prompt": "tell me about X"}]), encoding="utf-8")
        out = tmp_path / "out.csv"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "batch",
                str(prompts_file),
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--output",
                str(out),
                "--no-judge-tos",
            ],
        )
        assert result.exit_code == 0, result.output
        with out.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert "hallucination_risk" in rows[0]
        # The fake returns "Medium".
        assert rows[0]["hallucination_risk"] == "Medium"

    def test_json_has_risk_level_per_judge(
        self, hallu_provider: _HallucinationFake, tmp_path: Path
    ) -> None:
        prompts_file = tmp_path / "p.json"
        prompts_file.write_text(json.dumps([{"prompt": "x"}]), encoding="utf-8")
        out = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "batch",
                str(prompts_file),
                "--models",
                "gpt-5.5",
                "--judge",
                "claude-opus-4-7",
                "--check-hallucination",
                "--output",
                str(out),
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text(encoding="utf-8"))
        first = payload["results"][0]
        # Each judge entry has a risk_level.
        assert first["judges"][0]["risk_level"] == "Medium"
        # And the aggregated risk for the result is included.
        assert first["hallucination_risk"] == "Medium"
