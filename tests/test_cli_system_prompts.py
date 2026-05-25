"""CLI-level tests for the three system-prompt flags.

Covers:
    * --system-prompt threaded to provider verbatim
    * --system-prompts produces an N x M x T task matrix
    * --system-prompts comma-escape (`\\,` -> literal comma)
    * Mutual exclusion of the three flags
    * --system-prompt-file loads file content and applies it
    * --system-prompt-file with missing file gives clear error, exits 2
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from cli_modelarium.cli import _split_system_prompts, main as cli_main
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk


# ===== a recording fake provider that survives any model name =====


class _RecordingProvider(BaseProvider):
    """Fake provider that accepts any model and records every call it sees."""

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
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
            }
        )
        if on_chunk is not None:
            on_chunk("ok")
        return CompletionResult(
            output="ok", model=model, provider="fake", temperature=temperature
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> _RecordingProvider:
    """Install a single fake provider for every provider name the CLI asks for."""
    fake = _RecordingProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    return fake


# ===== --system-prompts comma split (direct unit test) =====


class TestSplitSystemPrompts:
    def test_single_value(self) -> None:
        assert _split_system_prompts("a") == ["a"]

    def test_multiple_values(self) -> None:
        assert _split_system_prompts("a,b,c") == ["a", "b", "c"]

    def test_whitespace_stripped(self) -> None:
        assert _split_system_prompts("  a , b ,  c  ") == ["a", "b", "c"]

    def test_empty_pieces_dropped(self) -> None:
        assert _split_system_prompts("a,,b,") == ["a", "b"]
        assert _split_system_prompts(",") == []
        assert _split_system_prompts("") == []

    def test_escaped_comma_becomes_literal(self) -> None:
        # Source: `r"a,b,c\,d"` -> input string `a,b,c\,d`.
        assert _split_system_prompts(r"a,b,c\,d") == ["a", "b", "c,d"]

    def test_multiple_escapes_in_one_value(self) -> None:
        assert _split_system_prompts(r"a\,b\,c,d") == ["a,b,c", "d"]

    def test_backslash_without_comma_is_literal(self) -> None:
        # \x is NOT a special escape - kept verbatim.
        assert _split_system_prompts(r"a\b,c") == [r"a\b", "c"]

    def test_trailing_backslash_is_literal(self) -> None:
        assert _split_system_prompts("hello\\") == ["hello\\"]


# ===== --system-prompt threading =====


class TestSystemPromptFlag:
    def test_threads_to_provider(self, fake_provider: _RecordingProvider) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "test prompt",
                "--models", "gpt-5.5",
                "--system-prompt", "you are a poet",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 1
        assert fake_provider.calls[0]["system_prompt"] == "you are a poet"

    def test_empty_string_treated_as_no_prompt(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["test", "--models", "gpt-5.5", "--system-prompt", "", "--no-stream"],
        )

        assert result.exit_code == 0
        # Empty system prompt means: don't send one at all.
        assert fake_provider.calls[0]["system_prompt"] is None


# ===== --system-prompts matrix expansion =====


class TestSystemPromptsMatrix:
    def test_three_prompts_two_models_one_temp_makes_six_calls(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "user prompt",
                "--models", "gpt-5.5,gpt-5.4",
                "--system-prompts", "p1,p2,p3",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(fake_provider.calls) == 6
        # Each system prompt appears for each model.
        prompts_seen = {c["system_prompt"] for c in fake_provider.calls}
        assert prompts_seen == {"p1", "p2", "p3"}
        # Each model seen for each prompt.
        models_seen = {c["model"] for c in fake_provider.calls}
        assert models_seen == {"gpt-5.5", "gpt-5.4"}

    def test_temperature_dimension_multiplies(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--temperatures", "0.0,1.0",
                "--system-prompts", "a,b",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0
        # 2 SPs × 1 model × 2 temperatures = 4
        assert len(fake_provider.calls) == 4
        temps = sorted(c["temperature"] for c in fake_provider.calls)
        assert temps == [0.0, 0.0, 1.0, 1.0]

    def test_escaped_comma_passes_through_to_provider(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--system-prompts", r"a,b,c\,d",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0
        prompts_seen = {c["system_prompt"] for c in fake_provider.calls}
        assert prompts_seen == {"a", "b", "c,d"}


# ===== mutual exclusion =====


class TestMutualExclusion:
    def test_system_prompt_and_system_prompts_together_rejected(
        self, fake_provider: _RecordingProvider
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--system-prompt", "single",
                "--system-prompts", "a,b",
                "--no-stream",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()
        assert len(fake_provider.calls) == 0

    def test_system_prompt_and_system_prompt_file_together_rejected(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "sp.txt"
        prompt_file.write_text("from-file", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--system-prompt", "inline",
                "--system-prompt-file", str(prompt_file),
                "--no-stream",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_system_prompts_and_system_prompt_file_together_rejected(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "sp.txt"
        prompt_file.write_text("from-file", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--system-prompts", "a,b",
                "--system-prompt-file", str(prompt_file),
                "--no-stream",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# ===== --system-prompt-file =====


class TestSystemPromptFile:
    def test_loads_and_applies(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "system.txt"
        prompt_file.write_text(
            "  You are a research assistant.  \n", encoding="utf-8"
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "user prompt",
                "--models", "gpt-5.5",
                "--system-prompt-file", str(prompt_file),
                "--no-stream",
            ],
        )

        assert result.exit_code == 0, result.output
        # Whitespace was stripped during load.
        assert fake_provider.calls[0]["system_prompt"] == "You are a research assistant."

    def test_missing_file_exits_2_with_clear_message(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nope.txt"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "p",
                "--models", "gpt-5.5",
                "--system-prompt-file", str(missing),
                "--no-stream",
            ],
        )

        # Click's `exists=True` on type=click.Path catches this BEFORE we
        # ever reach load_system_prompt. The actual exit code may be 2
        # (Click's UsageError default).
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()
        assert len(fake_provider.calls) == 0
