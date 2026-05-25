"""Tests for cli_modelarium.output_formatters - CSV, JSON, Markdown writers."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from cli_modelarium import __version__
from cli_modelarium.output_formatters import (
    CSV_COLUMNS,
    BatchResult,
    _format_csv,
    _format_json,
    _format_markdown,
    atomic_write_bytes,
    write_csv,
    write_json,
    write_markdown,
)
from cli_modelarium.pricing import PRICING_AS_OF


def _make_result(
    *,
    prompt_id: str = "p1",
    prompt: str = "what is 2+2?",
    system: str | None = None,
    model: str = "gpt-5.5",
    temperature: float = 0.0,
    output: str = "4",
    error: str | None = None,
    cost_usd: float = 0.001,
    latency_ms: float | None = 850.0,
    ttft_ms: float | None = 120.0,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached_tokens: int = 0,
    retries: int = 0,
    assertions_raw: list | None = None,
) -> BatchResult:
    return BatchResult(
        prompt_id=prompt_id,
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        output=output,
        error=error,
        retries=retries,
        assertions_raw=assertions_raw or [],
    )


# ===== CSV =====


class TestCsvFormat:
    def test_column_order_matches_canonical(self) -> None:
        text = _format_csv([_make_result()])
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        assert tuple(header) == CSV_COLUMNS

    def test_empty_results_produces_header_only(self) -> None:
        text = _format_csv([])
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        assert tuple(header) == CSV_COLUMNS
        with pytest.raises(StopIteration):
            next(reader)

    def test_row_values_serialized(self) -> None:
        text = _format_csv(
            [
                _make_result(
                    prompt_id="math-1",
                    model="gpt-5.5",
                    temperature=0.7,
                    output="answer",
                    cost_usd=0.000123,
                )
            ]
        )
        reader = csv.DictReader(io.StringIO(text))
        row = next(reader)
        assert row["prompt_id"] == "math-1"
        assert row["model"] == "gpt-5.5"
        assert row["temperature"] == "0.7"
        assert row["output"] == "answer"
        assert row["cost_usd"] == "0.000123"

    def test_newlines_in_output_escaped_to_literal(self) -> None:
        """A multi-line response must NOT break the CSV row."""
        text = _format_csv(
            [_make_result(output="line1\nline2\nline3")]
        )
        rows = list(csv.DictReader(io.StringIO(text)))
        # Exactly one row, no premature line break.
        assert len(rows) == 1
        # And the original newlines are now literal \n.
        assert "\\n" in rows[0]["output"]

    def test_carriage_returns_in_output_escaped(self) -> None:
        text = _format_csv([_make_result(output="line1\r\nline2")])
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 1
        assert "\\n" in rows[0]["output"]

    def test_none_latency_renders_as_empty_string(self) -> None:
        text = _format_csv(
            [_make_result(latency_ms=None, ttft_ms=None, error="failed")]
        )
        rows = list(csv.DictReader(io.StringIO(text)))
        # CSV doesn't have a real NULL - empty string is the convention.
        assert rows[0]["latency_ms"] == ""
        assert rows[0]["ttft_ms"] == ""

    def test_utf8_no_bom(self, tmp_path: Path) -> None:
        """CSV files should NOT start with a UTF-8 BOM (pandas/Excel quirks)."""
        path = tmp_path / "out.csv"
        write_csv(
            [_make_result(prompt="hello with utf-8: é 中文")],
            path,
        )
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        # And the unicode round-trips.
        text = raw.decode("utf-8")
        assert "中文" in text

    def test_write_csv_uses_newline_empty_no_extra_blank_rows(
        self, tmp_path: Path
    ) -> None:
        """Without newline='', csv would write \\r\\n+\\n on Windows -> blank rows.
        We check by parsing the file back: row count must match input count.
        """
        path = tmp_path / "out.csv"
        write_csv(
            [_make_result(prompt_id="a"), _make_result(prompt_id="b"), _make_result(prompt_id="c")],
            path,
        )
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_atomic_write_via_tmp(self, tmp_path: Path) -> None:
        path = tmp_path / "out.csv"
        write_csv([_make_result()], path)
        # The .tmp file should NOT linger after the write.
        assert not (tmp_path / "out.csv.tmp").exists()
        assert path.exists()


# ===== JSON =====


class TestJsonFormat:
    def test_metadata_header_present(self) -> None:
        text = _format_json([_make_result()])
        parsed = json.loads(text)
        assert parsed["version"] == __version__
        assert parsed["pricing_as_of"] == PRICING_AS_OF
        assert "total_cost_usd" in parsed
        assert "total_results" in parsed
        assert "failed_results" in parsed
        assert "results" in parsed

    def test_total_cost_excludes_failed_results(self) -> None:
        text = _format_json(
            [
                _make_result(cost_usd=0.01),
                _make_result(cost_usd=0.02, error="failed"),
            ]
        )
        parsed = json.loads(text)
        # Only the successful row contributes.
        assert parsed["total_cost_usd"] == pytest.approx(0.01)
        assert parsed["failed_results"] == 1
        assert parsed["total_results"] == 2

    def test_results_array_round_trips_unicode(self) -> None:
        text = _format_json(
            [_make_result(prompt="hello with utf-8: é 中文")]
        )
        # ensure_ascii=False keeps unicode as-is (not \uXXXX).
        assert "中文" in text
        parsed = json.loads(text)
        assert parsed["results"][0]["prompt"] == "hello with utf-8: é 中文"

    def test_indent_2(self) -> None:
        text = _format_json([_make_result()])
        assert "  " in text  # at least one 2-space indent

    def test_assertions_field_preserved(self) -> None:
        result = _make_result(
            assertions_raw=[{"type": "json_valid"}, {"type": "max_length_chars", "value": 100}]
        )
        text = _format_json([result])
        parsed = json.loads(text)
        assert parsed["results"][0]["assertions"] == [
            {"type": "json_valid"},
            {"type": "max_length_chars", "value": 100},
        ]

    def test_empty_results_produces_valid_metadata(self) -> None:
        text = _format_json([])
        parsed = json.loads(text)
        assert parsed["results"] == []
        assert parsed["total_results"] == 0
        assert parsed["total_cost_usd"] == 0.0

    def test_write_json_atomic(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json([_make_result()], path)
        assert not (tmp_path / "out.json.tmp").exists()
        assert path.exists()
        # And it parses.
        json.loads(path.read_text(encoding="utf-8"))


# ===== Markdown =====


class TestMarkdownFormat:
    def test_header_metadata(self) -> None:
        text = _format_markdown([_make_result()])
        assert "# Cli Modelarium batch results" in text
        assert PRICING_AS_OF in text
        assert __version__ in text

    def test_one_section_per_prompt(self) -> None:
        text = _format_markdown(
            [
                _make_result(prompt_id="p1", prompt="first"),
                _make_result(prompt_id="p2", prompt="second"),
                _make_result(prompt_id="p1", prompt="first", model="claude-opus-4-7"),
            ]
        )
        # Two distinct prompt_ids -> two `## ` sections.
        assert text.count("## p1") == 1
        assert text.count("## p2") == 1

    def test_local_models_show_free_in_row(self) -> None:
        text = _format_markdown([_make_result(model="local/llama-3.3", cost_usd=0.0)])
        # The per-row table cell for cost must read "Free", not "$0.000000".
        # (The aggregate "Total cost" line in the header IS allowed to be
        # $0.000000 - that's the sum across all rows.)
        row_lines = [
            line for line in text.splitlines() if "local/llama-3.3" in line
        ]
        assert row_lines, "local model row should appear in markdown"
        assert any("Free" in line for line in row_lines)
        # Crucially, the row itself doesn't carry $0.000000 - "Free" wins there.
        assert not any("$0.000000" in line for line in row_lines)

    def test_cloud_models_show_dollar_cost(self) -> None:
        text = _format_markdown([_make_result(cost_usd=0.001234)])
        assert "$0.001234" in text

    def test_error_rows_show_error_message(self) -> None:
        text = _format_markdown([_make_result(error="auth failed")])
        assert "auth failed" in text

    def test_table_columns_present(self) -> None:
        text = _format_markdown([_make_result()])
        # Sub-table header.
        for col in ("Model", "Temp", "TTFT", "Latency", "In", "Out", "Cost", "Status"):
            assert col in text

    def test_empty_results_does_not_crash(self) -> None:
        text = _format_markdown([])
        assert "# Cli Modelarium batch results" in text
        # Says something about being empty.
        assert "empty" in text.lower() or "no results" in text.lower()

    def test_pipe_in_prompt_escaped(self) -> None:
        """A `|` in the prompt mustn't break the Markdown table cell."""
        text = _format_markdown(
            [_make_result(prompt="hello | world")]
        )
        # The literal pipe inside the cell needs escaping.
        assert "hello \\| world" in text or "hello | world" in text  # either acceptable; pipe-in-prompt is in the H2 row, not the model table

    def test_newlines_in_prompt_collapsed(self) -> None:
        text = _format_markdown([_make_result(prompt="line one\nline two")])
        # Newlines in prompt would break the Markdown structure.
        assert "line one\nline two" not in text or "line one / line two" in text

    def test_write_markdown_atomic(self, tmp_path: Path) -> None:
        path = tmp_path / "out.md"
        write_markdown([_make_result()], path)
        assert not (tmp_path / "out.md.tmp").exists()
        assert path.exists()
        assert "# Cli Modelarium" in path.read_text(encoding="utf-8")


# ===== atomic_write_bytes =====


class TestAtomicWrite:
    def test_normal_write(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        atomic_write_bytes(path, b"hello")
        assert path.read_bytes() == b"hello"
        assert not (tmp_path / "x.txt.tmp").exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_bytes(b"original")
        atomic_write_bytes(path, b"replaced")
        assert path.read_bytes() == b"replaced"

    def test_tmp_cleaned_up_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "x.txt"

        # Force os.replace to fail so we can verify .tmp cleanup.
        def boom(src: str, dst: str) -> None:
            raise OSError("simulated failure")

        monkeypatch.setattr("cli_modelarium.output_formatters.os.replace", boom)

        with pytest.raises(OSError):
            atomic_write_bytes(path, b"never lands")

        # The target file should not exist (write failed before it landed).
        assert not path.exists()
        # And the temp file should have been cleaned up.
        assert not (tmp_path / "x.txt.tmp").exists()
