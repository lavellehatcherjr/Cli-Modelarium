"""CLI-level tests for the --judge / --judges flags on compare and batch.

Strategy: monkeypatch `_get_provider_instance` to return a smart fake that
detects whether the prompt is a judge prompt (looks for the "JSON object"
marker we put in JUDGE_PROMPT_TEMPLATE) and responds accordingly.
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
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


# ===== smart fake that detects judge prompts =====


class _DualPurposeProvider(BaseProvider):
    """Fake provider that responds to main-call prompts AND judge-call prompts.

    Detects judge prompts by looking for the JUDGE_PROMPT_TEMPLATE's
    signature substrings. Records every call so tests can inspect.
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
                "temperature": temperature,
                "system_prompt": system_prompt,
                "is_judge_call": is_judge_call,
            }
        )

        if is_judge_call:
            output = self._judge_response
        else:
            output = f"answer for {model}"

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
    # _validate_judge_models reads `is_key_configured` directly, not through
    # _get_provider_instance. Set fake env vars for every provider that
    # might be used as a judge in these tests so validation passes.
    for env, value in (
        ("OPENAI_API_KEY", "sk-proj-test1234567890abcdefghi"),
        ("ANTHROPIC_API_KEY", "sk-ant-test1234567890abcdefghi"),
        ("GOOGLE_API_KEY", "AIzaTestabcdef1234567890abcdef1234567"),
        ("XAI_API_KEY", "xai-test1234567890abcdefghi"),
        ("GROQ_API_KEY", "gsk_test1234567890abcdefghi"),
    ):
        monkeypatch.setenv(env, value)
    return fake


# ===== --judge flag =====


class TestJudgeFlag:
    def test_single_judge_runs_end_to_end(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
                "--no-judge-tos",  # suppress ToS panel for cleaner output
            ],
        )

        assert result.exit_code == 0, result.output
        # 1 main call (gpt-5.5) + 1 judge call (claude-opus-4-7).
        assert len(dual_provider.calls) == 2
        # The judge call must use temperature 0.0.
        judge_calls = [c for c in dual_provider.calls if c["is_judge_call"]]
        assert len(judge_calls) == 1
        assert judge_calls[0]["temperature"] == 0.0

    def test_score_appears_in_output(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        # The fake judge returns score 8. The Score column should show "8".
        assert "Score" in result.output
        assert "8" in result.output

    def test_judge_cost_displayed(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        assert "Judge cost" in result.output


# ===== --judges panel =====


class TestJudgesPanel:
    def test_three_judges_each_called_once(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judges", "claude-opus-4-7,gemini-3.1-pro,grok-4.3",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        # 1 main + 3 judge calls.
        assert len(dual_provider.calls) == 4
        judge_calls = [c for c in dual_provider.calls if c["is_judge_call"]]
        assert len(judge_calls) == 3
        judge_models = {c["model"] for c in judge_calls}
        assert judge_models == {"claude-opus-4-7", "gemini-3.1-pro", "grok-4.3"}


# ===== --judge-criteria =====


class TestJudgeCriteria:
    def test_custom_criteria_reach_judge_prompt(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--judge-criteria", "Brevity,Clever wordplay",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        judge_call = [c for c in dual_provider.calls if c["is_judge_call"]][0]
        assert "Brevity" in judge_call["prompt"]
        assert "Clever wordplay" in judge_call["prompt"]

    def test_escaped_comma_in_criteria(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--judge-criteria", r"crit one,crit two\, with comma",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        judge_call = [c for c in dual_provider.calls if c["is_judge_call"]][0]
        assert "crit two, with comma" in judge_call["prompt"]


# ===== --judge-template =====


class TestJudgeTemplate:
    def test_custom_template_replaces_default(
        self, dual_provider: _DualPurposeProvider, tmp_path: Path
    ) -> None:
        # The custom template must still contain the JUDGE_MARKER so the fake
        # recognizes it as a judge call.
        template_path = tmp_path / "judge.txt"
        template_path.write_text(
            "Custom marker: {criteria}\n"
            "Q: {prompt}\n"
            "A: {response}\n"
            "Respond with ONLY a JSON object",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--judge-template", str(template_path),
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        judge_call = [c for c in dual_provider.calls if c["is_judge_call"]][0]
        assert "Custom marker:" in judge_call["prompt"]


# ===== --include-reasoning =====


class TestIncludeReasoning:
    def test_reasoning_appears_when_flag_set(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--include-reasoning",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        # Fake judge's reasoning text is "decent".
        assert "decent" in result.output

    def test_reasoning_hidden_by_default(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        assert "decent" not in result.output


# ===== ToS panel =====


class TestToSPanel:
    def test_tos_shown_by_default(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0
        # The build prompt's verbatim wording.
        assert "Judge scores are for evaluation only" in result.output
        assert "competing AI models" in result.output
        assert "ToS" in result.output

    def test_tos_suppressed_with_flag(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test",
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0
        assert "Judge scores are for evaluation only" not in result.output

    def test_no_tos_when_no_judging(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["test", "--models", "gpt-5.5", "--no-stream"],
        )

        assert result.exit_code == 0
        assert "Judge scores are for evaluation only" not in result.output


# ===== mutual exclusion =====


class TestMutualExclusion:
    def test_judge_and_judges_together_rejected(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p", "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--judges", "claude-opus-4-7,gemini-3.1-pro",
                "--no-stream",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()
        assert dual_provider.calls == []

    def test_judge_criteria_and_template_together_rejected(
        self, dual_provider: _DualPurposeProvider, tmp_path: Path
    ) -> None:
        template = tmp_path / "t.txt"
        template.write_text("Template", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p", "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--judge-criteria", "x,y",
                "--judge-template", str(template),
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# ===== judge model validation =====


class TestJudgeModelValidation:
    def test_unknown_judge_model_rejected_before_main_run(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test", "--models", "gpt-5.5",
                "--judge", "totally-fake-judge-model",
                "--no-stream",
                "--no-judge-tos",
            ],
        )
        assert result.exit_code != 0
        # No main calls happened either.
        assert dual_provider.calls == []
        # Error message names the bad model.
        assert "totally-fake-judge-model" in result.output

    def test_judge_key_missing_rejected_before_main_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the judge provider has no API key, refuse before the main run.

        We don't install the dual_provider fixture here so the real
        `_get_provider_instance` runs and KeyNotConfiguredError fires.
        """
        # Ensure no env var grants implicit access for any provider.
        for env in (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "XAI_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY",
            "GROQ_API_KEY", "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test", "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code != 0
        # The actionable KeyNotConfiguredError message includes both
        # the keys-set command and the env-var fallback.
        assert "keys set anthropic" in result.output.lower()


# ===== self-evaluation skip =====


class TestSelfEvaluationSkip:
    def test_same_model_as_main_skipped(
        self, dual_provider: _DualPurposeProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test", "--models", "gpt-5.5",
                "--judge", "gpt-5.5",
                "--no-stream",
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        # Only 1 call total: the main call. The judge was the same model,
        # so it was skipped via self-eval guard.
        judge_calls = [c for c in dual_provider.calls if c["is_judge_call"]]
        assert judge_calls == []


# ===== batch + judge integration =====


class TestBatchWithJudges:
    def test_batch_csv_includes_judge_columns(
        self, dual_provider: _DualPurposeProvider, tmp_path: Path
    ) -> None:
        prompts_file = tmp_path / "p.txt"
        prompts_file.write_text("what is 2+2?\n", encoding="utf-8")
        output = tmp_path / "out.csv"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "batch", str(prompts_file),
                "--models", "gpt-5.5",
                "--judge", "claude-opus-4-7",
                "--output", str(output),
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["judge_score_avg"] == "8"
        assert rows[0]["judge_count"] == "1"

    def test_batch_json_includes_judges_array(
        self, dual_provider: _DualPurposeProvider, tmp_path: Path
    ) -> None:
        prompts_file = tmp_path / "p.json"
        prompts_file.write_text(json.dumps([{"prompt": "hi"}]), encoding="utf-8")
        output = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "batch", str(prompts_file),
                "--models", "gpt-5.5",
                "--judges", "claude-opus-4-7,gemini-3.1-pro",
                "--output", str(output),
                "--no-judge-tos",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(output.read_text(encoding="utf-8"))
        # Metadata has judge_cost_usd.
        assert "judge_cost_usd" in payload
        assert "total_cost_usd_with_judges" in payload
        # Per-result judges array.
        first = payload["results"][0]
        assert "judges" in first
        assert len(first["judges"]) == 2
        assert {j["model"] for j in first["judges"]} == {
            "claude-opus-4-7", "gemini-3.1-pro",
        }
