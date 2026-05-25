"""Tests for the optional `jsonschema` dependency handling.

The `json_schema` assertion type degrades gracefully when jsonschema isn't
installed: the AssertionResult has `error` set with an install hint, and
the result does NOT count as a failure for exit-code purposes.

The trick to testing this is patching the import inside the check function
so the `try: import jsonschema` line raises ImportError.
"""
from __future__ import annotations

import builtins

import pytest

from cli_modelarium.assertions import (
    AssertionResult,
    count_failed,
    count_passed,
    run_assertions,
)


def _patch_missing(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """Force `import module_name` to raise ImportError, simulating uninstalled."""
    import sys

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == module_name or name.startswith(module_name + "."):
            raise ImportError(f"simulated: {module_name} not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestJsonschemaOptional:
    def test_jsonschema_present_assertions_run_normally(self) -> None:
        """Sanity baseline: with jsonschema installed (real env), schema validation works."""
        results = run_assertions(
            '{"x": 1}', None, 0.0,
            [{"type": "json_schema",
              "value": {"type": "object", "properties": {"x": {"type": "integer"}}}}],
        )
        assert results[0].passed
        assert results[0].error is None

    def test_jsonschema_missing_sets_error_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_missing(monkeypatch, "jsonschema")

        results = run_assertions(
            '{"x": 1}', None, 0.0,
            [{"type": "json_schema",
              "value": {"type": "object"}}],
        )

        # Single assertion result.
        assert len(results) == 1
        r: AssertionResult = results[0]
        # error is set with the install hint.
        assert r.error is not None
        assert "jsonschema" in r.error
        assert "cli-modelarium[schema]" in r.error
        # passed=False but the row is treated as "couldn't run".
        assert r.passed is False

    def test_jsonschema_missing_does_not_crash_other_assertions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing optional dep must not poison the other 8 assertion types."""
        _patch_missing(monkeypatch, "jsonschema")

        results = run_assertions(
            "Paris is the capital", None, 0.0,
            [
                {"type": "contains", "value": "Paris"},
                {"type": "json_schema", "value": {"type": "object"}},
                {"type": "max_length_chars", "value": 100},
            ],
        )

        assert results[0].passed
        assert results[0].error is None
        assert results[1].error is not None  # the json_schema row
        assert results[2].passed
        assert results[2].error is None

    def test_jsonschema_missing_excluded_from_pass_fail_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error rows are NOT counted as failures - this is the contract that
        keeps `--strict-assertions` from exiting 1 when the user simply
        forgot to pip install cli-modelarium[schema].
        """
        _patch_missing(monkeypatch, "jsonschema")

        results = run_assertions(
            '{"x": 1}', None, 0.0,
            [
                {"type": "contains", "value": "x"},
                {"type": "json_schema", "value": {"type": "object"}},
            ],
        )

        passed, definitive = count_passed(results)
        assert passed == 1
        # Only the contains assertion is a "definitive" verdict; the
        # json_schema row was unable to run and is excluded.
        assert definitive == 1
        # Zero definitive failures.
        assert count_failed(results) == 0


class TestSchemaExtraPyproject:
    """Cross-check that pyproject.toml's [schema] extra is the documented path."""

    def test_pyproject_schema_extra_includes_jsonschema(self) -> None:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        extras = cfg["project"]["optional-dependencies"]
        assert "schema" in extras
        # The dep is named jsonschema; the version pin is not strictly part of
        # this contract, but check that the dep is at least listed.
        schema_deps = " ".join(extras["schema"])
        assert "jsonschema" in schema_deps
