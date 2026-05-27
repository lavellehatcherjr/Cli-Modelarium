"""CLI-level tests for --output, --output-format, and --force on `compare`.

Strategy: monkeypatch `_get_provider_instance` to return a fake that records
every call and returns a deterministic CompletionResult. Then drive the
compare command via Click's CliRunner with various flag combinations.

Mirrors the test approach used by tests/test_cli_batch.py.
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


class _RecordingProvider(BaseProvider):
    """Returns a preset CompletionResult and records every call."""

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
        text = f"answer for {prompt[:20]}"
        if on_chunk is not None:
            on_chunk(text)
        return CompletionResult(
            output=text,
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
    """Install the fake provider for every provider name the CLI requests."""
    fake = _RecordingProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    return fake


# ===== --output happy paths =====


class TestCompareOutputHappyPaths:
    def test_csv_output_writes_file(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["what is 2+2?", "--models", "gpt-5.5", "--output", str(output), "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        assert output.exists()
        with output.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["prompt_id"] == "p1"
        assert rows[0]["model"] == "gpt-5.5"
        assert rows[0]["output"].startswith("answer for")

    def test_json_output_writes_file(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi there", "--models", "gpt-5.5", "--output", str(output), "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["total_results"] == 1
        assert payload["results"][0]["prompt_id"] == "p1"
        assert payload["results"][0]["model"] == "gpt-5.5"

    def test_markdown_output_writes_file(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.md"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--output", str(output), "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        text = output.read_text(encoding="utf-8")
        assert "Cli Modelarium" in text
        assert "gpt-5.5" in text

    def test_output_format_override_extension(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        """--output-format wins over the inferred extension."""
        # .data extension can't be auto-detected; --output-format json wins.
        output = tmp_path / "results.data"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "hi",
                "--models",
                "gpt-5.5",
                "--output",
                str(output),
                "--output-format",
                "json",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0, result.output
        # Content parses as JSON regardless of the .data extension.
        json.loads(output.read_text(encoding="utf-8"))


# ===== --output validation failures =====


class TestCompareOutputValidation:
    def test_unknown_extension_without_format_fails(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        output = tmp_path / "results.unknown"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--output", str(output), "--no-stream"],
        )

        assert result.exit_code != 0
        assert "cannot infer output format" in result.output.lower()
        # No provider calls should have happened.
        assert fake_provider.calls == []

    def test_existing_file_without_force_refused(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        existing = tmp_path / "existing.csv"
        existing.write_text("pre-existing", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--output", str(existing), "--no-stream"],
        )

        assert result.exit_code != 0
        assert "already exists" in result.output.lower()
        assert "--force" in result.output
        assert existing.read_text(encoding="utf-8") == "pre-existing"
        # No provider calls should have happened.
        assert fake_provider.calls == []

    def test_existing_file_with_force_overwrites(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        existing = tmp_path / "existing.csv"
        existing.write_text("pre-existing", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "hi",
                "--models",
                "gpt-5.5",
                "--output",
                str(existing),
                "--force",
                "--no-stream",
            ],
        )

        assert result.exit_code == 0, result.output
        # The CSV header replaced the placeholder content.
        assert "prompt_id" in existing.read_text(encoding="utf-8")


# ===== backward compat =====


class TestCompareDisplayBackwardCompat:
    def test_output_suppresses_rich_table(
        self, fake_provider: _RecordingProvider, tmp_path: Path
    ) -> None:
        """When --output is set the Rich summary table is not printed."""
        output = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--output", str(output), "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        # The Rich display path prints "Comparing N completion" as a table
        # title; when --output is set we route to the batch writer instead.
        assert "Comparing 1 completion" not in result.output

    def test_no_output_flags_preserves_default_display(
        self, fake_provider: _RecordingProvider
    ) -> None:
        """No --output / --output-format / --max-cost: existing Rich display."""
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["hi", "--models", "gpt-5.5", "--no-stream"],
        )

        assert result.exit_code == 0, result.output
        # The existing _display_results path emits a summary table titled
        # "Comparing N completion(s)".
        assert "Comparing 1 completion" in result.output
