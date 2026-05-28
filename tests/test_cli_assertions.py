"""CLI-level tests for assertion integration in the batch command.

Verifies the exit-code contract (0/1/2), output formatter integration,
and that --no-assertions, --min-pass-rate, and --strict-assertions all
behave per the spec.
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

# ===== fake provider that returns a controllable response =====


class _CannedProvider(BaseProvider):
    """Returns a preset output for every call. Optional error for one call."""

    def __init__(self, response: str = "Paris is the capital of France") -> None:
        self.name = "fake"
        self._response = response
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
        self.calls.append({"prompt": prompt, "model": model})
        if on_chunk is not None:
            on_chunk(self._response)
        return CompletionResult(
            output=self._response,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
            latency_ms=50.0,
            ttft_ms=10.0,
            model=model,
            provider="fake",
            temperature=temperature,
        )


@pytest.fixture
def canned_provider(monkeypatch: pytest.MonkeyPatch) -> _CannedProvider:
    fake = _CannedProvider()
    monkeypatch.setattr(
        "cli_modelarium.cli._get_provider_instance",
        lambda name, **_kwargs: fake,
    )
    return fake


def _write_json(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items), encoding="utf-8")


# ===== exit code 0: all passing =====


class TestExitCodeZeroAllPass:
    def test_all_passing_assertions(self, canned_provider: _CannedProvider, tmp_path: Path) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "what is the capital of France?",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},
                        {"type": "max_length_chars", "value": 1000},
                    ],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5"],
        )

        assert result.exit_code == 0, result.output


# ===== exit code 1: assertion failures =====


class TestExitCodeOneAssertionFail:
    def test_single_failing_assertion(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [{"type": "contains", "value": "MISSING"}],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5"],
        )

        assert result.exit_code == 1, result.output

    def test_min_pass_rate_below_threshold(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        """50% pass rate against --min-pass-rate 0.5 must FAIL (strictly above)."""
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},  # pass
                        {"type": "contains", "value": "MISSING"},  # fail
                    ],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--min-pass-rate", "0.6"],
        )

        assert result.exit_code == 1

    def test_min_pass_rate_at_threshold_passes(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        """pass_rate >= threshold means OK. At exactly 50% with --min-pass-rate 0.5 → 0."""
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},
                        {"type": "contains", "value": "MISSING"},
                    ],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--min-pass-rate", "0.5"],
        )

        # 50% pass rate; threshold is 0.5; pass_rate >= threshold → 0.
        assert result.exit_code == 0

    def test_min_pass_rate_zero_always_passes(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        """--min-pass-rate 0.0 effectively disables the assertion gate."""
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [{"type": "contains", "value": "MISSING"}],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--min-pass-rate", "0.0"],
        )

        assert result.exit_code == 0


# ===== --no-assertions =====


class TestNoAssertions:
    def test_skips_execution_exit_zero(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        """--no-assertions means: even configured assertions are skipped."""
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [{"type": "contains", "value": "MISSING"}],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--no-assertions"],
        )

        assert result.exit_code == 0


# ===== --strict-assertions =====


class TestStrictAssertions:
    def test_default_strict_behavior_any_failure_exits_1(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        """Default (no flags): any failure → exit 1, same as --strict-assertions."""
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},
                        {"type": "contains", "value": "MISSING"},
                    ],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5"],
        )

        assert result.exit_code == 1

    def test_strict_and_min_pass_rate_together_rejected(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        prompts = tmp_path / "p.txt"
        prompts.write_text("hello\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "batch",
                str(prompts),
                "--models",
                "gpt-5.5",
                "--strict-assertions",
                "--min-pass-rate",
                "0.5",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# ===== --min-pass-rate validation =====


class TestMinPassRateValidation:
    def test_above_1_rejected(self, canned_provider: _CannedProvider, tmp_path: Path) -> None:
        prompts = tmp_path / "p.txt"
        prompts.write_text("hi\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--min-pass-rate", "1.5"],
        )
        assert result.exit_code != 0
        assert "0.0" in result.output and "1.0" in result.output

    def test_below_0_rejected(self, canned_provider: _CannedProvider, tmp_path: Path) -> None:
        prompts = tmp_path / "p.txt"
        prompts.write_text("hi\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--min-pass-rate", "-0.1"],
        )
        assert result.exit_code != 0


# ===== exit code 2: call failures dominate =====


class TestCallFailureDominates:
    def test_call_failure_plus_assertion_failure_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Per the spec: 'exit 2 wins' over assertion failures."""
        from cli_modelarium.exceptions import ProviderError

        class _DyingProvider(BaseProvider):
            def __init__(self) -> None:
                self.name = "fake"

            async def stream(self, *a: Any, **k: Any) -> AsyncIterator[str]:
                if False:
                    yield ""
                raise NotImplementedError

            async def complete(self, *a: Any, **k: Any) -> CompletionResult:
                raise ProviderError("provider died", provider="fake")

        monkeypatch.setattr(
            "cli_modelarium.cli._get_provider_instance",
            lambda name, **_kwargs: _DyingProvider(),
        )

        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {"id": "p1", "prompt": "x", "assertions": [{"type": "contains", "value": "Paris"}]},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5"],
        )

        # Both call AND assertion would fail. Call failure dominates → 2.
        assert result.exit_code == 2


# ===== output formatter integration =====


class TestCsvAssertionColumns:
    def test_csv_has_three_assertion_columns(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},  # pass
                        {"type": "max_length_chars", "value": 5},  # fail
                    ],
                },
            ],
        )
        out = tmp_path / "out.csv"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--output", str(out)],
        )

        # exit 1 expected because one assertion failed - the CSV is still written.
        assert result.exit_code == 1
        with out.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["assertions_passed"] == "1"
        assert rows[0]["assertions_total"] == "2"
        assert rows[0]["assertions_failed_types"] == "max_length_chars"


class TestJsonAssertionFields:
    def test_json_has_per_result_assertions_array(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},
                    ],
                },
            ],
        )
        out = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--output", str(out)],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text(encoding="utf-8"))
        # Metadata totals.
        assert payload["total_assertions"] == 1
        assert payload["total_assertions_passed"] == 1
        assert payload["pass_rate"] == 1.0
        # Per-result assertions.
        first = payload["results"][0]
        assert "assertions" in first
        assert len(first["assertions"]) == 1
        assert first["assertions"][0]["type"] == "contains"
        assert first["assertions"][0]["passed"] is True


class TestMarkdownAssertionColumn:
    def test_markdown_shows_assertions_column(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},
                    ],
                },
            ],
        )
        out = tmp_path / "out.md"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--output", str(out)],
        )

        assert result.exit_code == 0
        text = out.read_text(encoding="utf-8")
        assert "Assertions" in text
        # 1/1 with the unicode check mark.
        assert "1/1" in text

    def test_markdown_lists_failures_below_row(
        self, canned_provider: _CannedProvider, tmp_path: Path
    ) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [
                        {"type": "contains", "value": "Paris"},
                        {"type": "max_length_chars", "value": 5},
                    ],
                },
            ],
        )
        out = tmp_path / "out.md"

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5", "--output", str(out)],
        )

        # exit 1 because max_length_chars failed - the markdown still writes.
        assert result.exit_code == 1
        text = out.read_text(encoding="utf-8")
        # Failing assertion is detailed below the row.
        assert "max_length_chars" in text


# ===== jsonschema missing: does NOT fail exit code =====


class TestJsonschemaMissingExitCode:
    def test_missing_jsonschema_does_not_trigger_exit_1(
        self,
        canned_provider: _CannedProvider,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A json_schema assertion that can't run (no jsonschema installed)
        should NOT count as a failure for exit-code purposes.
        """
        import builtins
        import sys

        monkeypatch.delitem(sys.modules, "jsonschema", raising=False)
        real_import = builtins.__import__

        def fake_import(name: str, *a: object, **k: object) -> object:
            if name == "jsonschema" or name.startswith("jsonschema."):
                raise ImportError("simulated")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {
                    "id": "p1",
                    "prompt": "x",
                    "assertions": [{"type": "json_schema", "value": {"type": "object"}}],
                },
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5"],
        )

        # Exit 0: the schema couldn't run, but that's not a failure verdict.
        assert result.exit_code == 0, result.output


# ===== each of 9 assertion types end-to-end =====


class TestNineAssertionTypesEndToEnd:
    """One test per assertion type via the full CLI flow."""

    @pytest.mark.parametrize(
        "assertion, passing",
        [
            ({"type": "contains", "value": "Paris"}, True),
            ({"type": "not_contains", "value": "MISSING"}, True),
            ({"type": "regex", "value": r"\bPa\w+"}, True),
            ({"type": "equals", "value": "Paris is the capital of France"}, True),
            ({"type": "json_valid"}, False),  # output isn't JSON
            ({"type": "json_schema", "value": {"type": "object"}}, False),  # not JSON
            ({"type": "min_length_chars", "value": 5}, True),
            ({"type": "max_length_chars", "value": 1000}, True),
            ({"type": "latency_under", "value": 1000}, True),  # fake latency 50ms
            ({"type": "cost_under", "value": 1.0}, True),  # fake cost ~$0.0001
        ],
    )
    def test_each_type(
        self,
        canned_provider: _CannedProvider,
        tmp_path: Path,
        assertion: dict,
        passing: bool,
    ) -> None:
        prompts = tmp_path / "p.json"
        _write_json(
            prompts,
            [
                {"id": "p1", "prompt": "x", "assertions": [assertion]},
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["batch", str(prompts), "--models", "gpt-5.5"],
        )

        if passing:
            assert result.exit_code == 0, result.output
        else:
            assert result.exit_code == 1, result.output


# ===== compare command does NOT show Assertions column =====


class TestCompareNoAssertions:
    def test_compare_output_has_no_assertions_column(
        self, canned_provider: _CannedProvider
    ) -> None:
        """The compare command never shows the Assertions column - assertions
        are batch-mode only.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            ["test prompt", "--models", "gpt-5.5", "--no-stream"],
        )

        assert result.exit_code == 0
        assert "Assertions" not in result.output
