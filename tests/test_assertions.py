"""Tests for cli_modelarium.assertions - the 9 assertion types and the runner.

The runner contract is "never raises" - every error becomes an
AssertionResult with `error` set. The exit-code logic in cli.py treats
those as "couldn't run" (excluded from pass/fail tallies).
"""
from __future__ import annotations

import json

import pytest

from cli_modelarium.assertions import (
    AssertionType,
    PASS_MARK,
    FAIL_MARK,
    ERROR_MARK,
    _CHECKERS,
    all_passed,
    count_failed,
    count_passed,
    failed_types,
    format_assertion_message,
    parse_assertion_config,
    run_assertions,
)
from cli_modelarium.exceptions import AssertionConfigError


# ===== AssertionType enum =====


class TestAssertionTypeEnum:
    def test_str_subclass_serializes_to_value(self) -> None:
        """str subclass means we can drop the .value access in most places."""
        assert AssertionType.CONTAINS == "contains"
        assert json.dumps(AssertionType.CONTAINS.value) == '"contains"'

    def test_all_nine_types_present(self) -> None:
        values = {t.value for t in AssertionType}
        assert values == {
            "contains", "not_contains", "regex", "equals",
            "json_valid", "json_schema",
            "min_length_chars", "max_length_chars",
            "latency_under", "cost_under",
        }

    def test_every_type_has_a_checker(self) -> None:
        for t in AssertionType:
            assert t.value in _CHECKERS


# ===== parse_assertion_config =====


class TestParseAssertionConfig:
    def test_minimal_contains(self) -> None:
        cfg = parse_assertion_config({"type": "contains", "value": "Paris"})
        assert cfg.type == "contains"
        assert cfg.value == "Paris"
        assert cfg.case_sensitive is True

    def test_json_valid_does_not_need_value(self) -> None:
        cfg = parse_assertion_config({"type": "json_valid"})
        assert cfg.value is None

    def test_missing_type_rejected(self) -> None:
        with pytest.raises(AssertionConfigError) as exc_info:
            parse_assertion_config({"value": "x"})
        assert "type" in str(exc_info.value).lower()

    def test_missing_value_rejected_except_json_valid(self) -> None:
        with pytest.raises(AssertionConfigError):
            parse_assertion_config({"type": "contains"})
        with pytest.raises(AssertionConfigError):
            parse_assertion_config({"type": "max_length_chars"})

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(AssertionConfigError) as exc_info:
            parse_assertion_config({"type": "totally-fake", "value": "x"})
        # Error message lists supported types.
        assert "contains" in str(exc_info.value)

    def test_non_string_type_rejected(self) -> None:
        with pytest.raises(AssertionConfigError):
            parse_assertion_config({"type": 42, "value": "x"})

    def test_non_dict_rejected(self) -> None:
        with pytest.raises(AssertionConfigError):
            parse_assertion_config("not a dict")  # type: ignore[arg-type]

    def test_case_sensitive_flag_accepted(self) -> None:
        cfg = parse_assertion_config(
            {"type": "contains", "value": "X", "case_sensitive": False}
        )
        assert cfg.case_sensitive is False

    def test_extra_fields_ignored(self) -> None:
        cfg = parse_assertion_config(
            {"type": "contains", "value": "X", "unknown_field": "ignored"}
        )
        assert cfg.value == "X"


# ===== CONTAINS =====


class TestContains:
    def test_pass(self) -> None:
        results = run_assertions("Paris is the capital", None, 0.0,
                                 [{"type": "contains", "value": "Paris"}])
        assert results[0].passed
        assert results[0].error is None

    def test_fail(self) -> None:
        results = run_assertions("London is foggy", None, 0.0,
                                 [{"type": "contains", "value": "Paris"}])
        assert not results[0].passed
        assert results[0].error is None

    def test_case_sensitive_by_default(self) -> None:
        results = run_assertions("paris", None, 0.0,
                                 [{"type": "contains", "value": "Paris"}])
        assert not results[0].passed

    def test_case_insensitive_flag(self) -> None:
        results = run_assertions("paris", None, 0.0,
                                 [{"type": "contains", "value": "Paris",
                                   "case_sensitive": False}])
        assert results[0].passed

    def test_empty_value(self) -> None:
        # Empty string is "in" any string.
        results = run_assertions("anything", None, 0.0,
                                 [{"type": "contains", "value": ""}])
        assert results[0].passed

    def test_multi_byte_chars(self) -> None:
        results = run_assertions("中文 with utf8", None, 0.0,
                                 [{"type": "contains", "value": "中文"}])
        assert results[0].passed


# ===== NOT_CONTAINS =====


class TestNotContains:
    def test_pass_when_absent(self) -> None:
        results = run_assertions("benign output", None, 0.0,
                                 [{"type": "not_contains", "value": "FORBIDDEN"}])
        assert results[0].passed

    def test_fail_when_present(self) -> None:
        results = run_assertions("contains FORBIDDEN word", None, 0.0,
                                 [{"type": "not_contains", "value": "FORBIDDEN"}])
        assert not results[0].passed

    def test_case_insensitive(self) -> None:
        results = run_assertions("OK", None, 0.0,
                                 [{"type": "not_contains", "value": "ok",
                                   "case_sensitive": False}])
        assert not results[0].passed


# ===== REGEX =====


class TestRegex:
    def test_simple_match(self) -> None:
        results = run_assertions("Year: 2026", None, 0.0,
                                 [{"type": "regex", "value": r"\d{4}"}])
        assert results[0].passed

    def test_no_match(self) -> None:
        results = run_assertions("no digits here", None, 0.0,
                                 [{"type": "regex", "value": r"\d{4}"}])
        assert not results[0].passed
        assert results[0].error is None

    def test_malformed_pattern_sets_error(self) -> None:
        results = run_assertions("anything", None, 0.0,
                                 [{"type": "regex", "value": "[unclosed"}])
        assert not results[0].passed
        assert results[0].error is not None
        assert "re.error" in results[0].error.lower() or "regex" in results[0].error.lower()

    def test_dotall_matches_across_newlines(self) -> None:
        results = run_assertions("foo\nbar", None, 0.0,
                                 [{"type": "regex", "value": r"foo.bar"}])
        # With DOTALL, `.` matches the newline.
        assert results[0].passed

    def test_anchors_work(self) -> None:
        results = run_assertions("hello world", None, 0.0,
                                 [{"type": "regex", "value": r"^hello"}])
        assert results[0].passed

    def test_case_insensitive_regex(self) -> None:
        results = run_assertions("HELLO", None, 0.0,
                                 [{"type": "regex", "value": "hello",
                                   "case_sensitive": False}])
        assert results[0].passed


# ===== EQUALS =====


class TestEquals:
    def test_exact_match(self) -> None:
        results = run_assertions("yes", None, 0.0,
                                 [{"type": "equals", "value": "yes"}])
        assert results[0].passed

    def test_whitespace_normalized(self) -> None:
        """Leading/trailing whitespace is stripped before comparison."""
        results = run_assertions("  yes  \n", None, 0.0,
                                 [{"type": "equals", "value": "yes"}])
        assert results[0].passed

    def test_crlf_normalized(self) -> None:
        results = run_assertions("hello\r\nworld", None, 0.0,
                                 [{"type": "equals", "value": "hello\nworld"}])
        assert results[0].passed

    def test_case_sensitive_default(self) -> None:
        results = run_assertions("Yes", None, 0.0,
                                 [{"type": "equals", "value": "yes"}])
        assert not results[0].passed

    def test_case_insensitive_flag(self) -> None:
        results = run_assertions("Yes", None, 0.0,
                                 [{"type": "equals", "value": "yes",
                                   "case_sensitive": False}])
        assert results[0].passed


# ===== JSON_VALID =====


class TestJsonValid:
    def test_bare_json_object(self) -> None:
        results = run_assertions('{"key": "value"}', None, 0.0,
                                 [{"type": "json_valid"}])
        assert results[0].passed

    def test_bare_json_array(self) -> None:
        results = run_assertions("[1, 2, 3]", None, 0.0,
                                 [{"type": "json_valid"}])
        assert results[0].passed

    def test_json_fenced(self) -> None:
        text = '```json\n{"key": 1}\n```'
        results = run_assertions(text, None, 0.0, [{"type": "json_valid"}])
        assert results[0].passed

    def test_bare_fence(self) -> None:
        text = '```\n{"key": 1}\n```'
        results = run_assertions(text, None, 0.0, [{"type": "json_valid"}])
        assert results[0].passed

    def test_malformed_fails(self) -> None:
        results = run_assertions("{ not valid", None, 0.0,
                                 [{"type": "json_valid"}])
        assert not results[0].passed
        assert results[0].error is None

    def test_empty_fails(self) -> None:
        results = run_assertions("", None, 0.0, [{"type": "json_valid"}])
        assert not results[0].passed

    def test_leading_whitespace_tolerated(self) -> None:
        results = run_assertions("   \n  {}", None, 0.0,
                                 [{"type": "json_valid"}])
        assert results[0].passed

    def test_json_embedded_in_prose_fails(self) -> None:
        """json_valid expects the OUTPUT to be JSON, not contain JSON."""
        results = run_assertions("Here is JSON: {\"x\": 1}", None, 0.0,
                                 [{"type": "json_valid"}])
        assert not results[0].passed


# ===== JSON_SCHEMA =====


class TestJsonSchema:
    SIMPLE_SCHEMA = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
    }

    def test_pass(self) -> None:
        results = run_assertions(
            '{"name": "Alice", "age": 30}', None, 0.0,
            [{"type": "json_schema", "value": self.SIMPLE_SCHEMA}],
        )
        assert results[0].passed, results[0].message

    def test_fail_with_error_path(self) -> None:
        results = run_assertions(
            '{"name": "Alice", "age": -5}', None, 0.0,
            [{"type": "json_schema", "value": self.SIMPLE_SCHEMA}],
        )
        assert not results[0].passed
        # The message should mention the path or the field that failed.
        assert "age" in results[0].message.lower() or "minimum" in results[0].message.lower()

    def test_invalid_json_fails_at_parse_step(self) -> None:
        results = run_assertions(
            "not json", None, 0.0,
            [{"type": "json_schema", "value": self.SIMPLE_SCHEMA}],
        )
        assert not results[0].passed
        assert "json" in results[0].message.lower()

    def test_malformed_schema_sets_error(self) -> None:
        """A bad schema is a CONFIG error, not a verdict on the output."""
        bad_schema = {"type": "not-a-type"}
        results = run_assertions(
            '{"x": 1}', None, 0.0,
            [{"type": "json_schema", "value": bad_schema}],
        )
        assert results[0].error is not None

    def test_fenced_json_handled(self) -> None:
        results = run_assertions(
            '```json\n{"name": "Bob", "age": 25}\n```', None, 0.0,
            [{"type": "json_schema", "value": self.SIMPLE_SCHEMA}],
        )
        assert results[0].passed

    def test_non_dict_schema_value_sets_error(self) -> None:
        results = run_assertions(
            '{"x": 1}', None, 0.0,
            [{"type": "json_schema", "value": "not a schema dict"}],
        )
        assert results[0].error is not None

    def test_empty_output(self) -> None:
        results = run_assertions(
            "", None, 0.0,
            [{"type": "json_schema", "value": self.SIMPLE_SCHEMA}],
        )
        assert not results[0].passed
        assert "empty" in results[0].message.lower()


# ===== MIN/MAX_LENGTH_CHARS =====


class TestLengthChars:
    def test_min_length_pass(self) -> None:
        results = run_assertions("hello world", None, 0.0,
                                 [{"type": "min_length_chars", "value": 5}])
        assert results[0].passed

    def test_min_length_fail(self) -> None:
        results = run_assertions("hi", None, 0.0,
                                 [{"type": "min_length_chars", "value": 5}])
        assert not results[0].passed

    def test_min_length_boundary(self) -> None:
        """len == limit means PASS (>=)."""
        results = run_assertions("12345", None, 0.0,
                                 [{"type": "min_length_chars", "value": 5}])
        assert results[0].passed

    def test_max_length_pass(self) -> None:
        results = run_assertions("hi", None, 0.0,
                                 [{"type": "max_length_chars", "value": 5}])
        assert results[0].passed

    def test_max_length_fail(self) -> None:
        results = run_assertions("hello world", None, 0.0,
                                 [{"type": "max_length_chars", "value": 5}])
        assert not results[0].passed

    def test_max_length_boundary(self) -> None:
        """len == limit means PASS (<=)."""
        results = run_assertions("12345", None, 0.0,
                                 [{"type": "max_length_chars", "value": 5}])
        assert results[0].passed

    def test_multi_byte_chars_counted_correctly(self) -> None:
        """Python len() counts codepoints, not bytes - so 中文 is 2 chars."""
        results = run_assertions("中文", None, 0.0,
                                 [{"type": "min_length_chars", "value": 2}])
        assert results[0].passed
        results = run_assertions("中文", None, 0.0,
                                 [{"type": "max_length_chars", "value": 2}])
        assert results[0].passed

    def test_non_numeric_value_sets_error(self) -> None:
        results = run_assertions("hi", None, 0.0,
                                 [{"type": "min_length_chars", "value": "five"}])
        assert results[0].error is not None


# ===== LATENCY_UNDER =====


class TestLatencyUnder:
    def test_pass_when_under(self) -> None:
        results = run_assertions("x", 1000.0, 0.0,
                                 [{"type": "latency_under", "value": 3000}])
        assert results[0].passed

    def test_fail_when_equal_strictly_less_than(self) -> None:
        """Strictly less than - equality is a FAIL."""
        results = run_assertions("x", 3000.0, 0.0,
                                 [{"type": "latency_under", "value": 3000}])
        assert not results[0].passed

    def test_fail_when_over(self) -> None:
        results = run_assertions("x", 4000.0, 0.0,
                                 [{"type": "latency_under", "value": 3000}])
        assert not results[0].passed

    def test_latency_none_sets_error(self) -> None:
        """Missing latency couldn't be checked - 'error' not 'fail'."""
        results = run_assertions("x", None, 0.0,
                                 [{"type": "latency_under", "value": 3000}])
        assert results[0].error is not None
        # It's NOT counted as a failure for exit code purposes.
        _, total = count_passed(results)
        assert total == 0

    def test_non_numeric_value_sets_error(self) -> None:
        results = run_assertions("x", 1000.0, 0.0,
                                 [{"type": "latency_under", "value": "fast"}])
        assert results[0].error is not None


# ===== COST_UNDER =====


class TestCostUnder:
    def test_pass_when_under(self) -> None:
        results = run_assertions("x", None, 0.0005,
                                 [{"type": "cost_under", "value": 0.001}])
        assert results[0].passed

    def test_fail_when_over(self) -> None:
        results = run_assertions("x", None, 0.002,
                                 [{"type": "cost_under", "value": 0.001}])
        assert not results[0].passed

    def test_local_models_free_always_pass(self) -> None:
        """cost_usd=0.0 < any positive limit - local models always pass."""
        results = run_assertions("x", None, 0.0,
                                 [{"type": "cost_under", "value": 0.001}])
        assert results[0].passed

    def test_strictly_less_than(self) -> None:
        results = run_assertions("x", None, 0.001,
                                 [{"type": "cost_under", "value": 0.001}])
        assert not results[0].passed

    def test_non_numeric_value_sets_error(self) -> None:
        results = run_assertions("x", None, 0.001,
                                 [{"type": "cost_under", "value": "cheap"}])
        assert results[0].error is not None


# ===== run_assertions runner =====


class TestRunAssertions:
    def test_empty_list_returns_empty(self) -> None:
        assert run_assertions("x", None, 0.0, []) == []

    def test_unknown_type_handled_gracefully(self) -> None:
        results = run_assertions("x", None, 0.0,
                                 [{"type": "totally-fake", "value": "x"}])
        assert not results[0].passed
        # error is set because the config validation rejected it.
        assert results[0].error is not None
        # And the type-name is preserved for the row's display.
        assert results[0].type == "totally-fake"

    def test_mixed_pass_fail(self) -> None:
        results = run_assertions(
            "Paris is great", None, 0.0,
            [
                {"type": "contains", "value": "Paris"},
                {"type": "not_contains", "value": "Paris"},
                {"type": "max_length_chars", "value": 100},
            ],
        )
        assert [r.passed for r in results] == [True, False, True]

    def test_order_preserved(self) -> None:
        results = run_assertions(
            "x", None, 0.0,
            [
                {"type": "contains", "value": "x"},
                {"type": "max_length_chars", "value": 5},
                {"type": "min_length_chars", "value": 1},
            ],
        )
        assert [r.type for r in results] == [
            "contains", "max_length_chars", "min_length_chars"
        ]

    def test_does_not_raise_on_anything(self) -> None:
        """Defensive: even pathological inputs must not raise."""
        # None as the assertion list entry would be weird but shouldn't crash.
        results = run_assertions("x", None, 0.0, [None])  # type: ignore[list-item]
        assert results[0].error is not None


# ===== aggregation helpers =====


class TestAggregationHelpers:
    def _make(self, passed: bool, error: str | None = None) -> object:
        from cli_modelarium.assertions import AssertionResult

        return AssertionResult(
            type="contains", passed=passed, expected="x", actual="x",
            message="", error=error,
        )

    def test_all_passed_empty_list(self) -> None:
        assert all_passed([]) is True

    def test_all_passed_true_case(self) -> None:
        results = [self._make(True), self._make(True)]
        assert all_passed(results) is True

    def test_all_passed_with_one_failure(self) -> None:
        results = [self._make(True), self._make(False)]
        assert all_passed(results) is False

    def test_all_passed_error_rows_dont_count_as_fail(self) -> None:
        """An 'error' row (couldn't run) isn't a failure."""
        results = [self._make(True), self._make(False, error="couldn't run")]
        assert all_passed(results) is True

    def test_count_passed_excludes_error_rows(self) -> None:
        results = [
            self._make(True),
            self._make(False),
            self._make(False, error="couldn't run"),
        ]
        passed, definitive = count_passed(results)
        assert passed == 1
        assert definitive == 2  # The error row is excluded.

    def test_count_failed_excludes_errors(self) -> None:
        results = [
            self._make(True),
            self._make(False),
            self._make(False, error="couldn't run"),
        ]
        assert count_failed(results) == 1

    def test_failed_types_preserves_order_and_dedupes(self) -> None:
        results = [
            self._make(False),  # contains
            self._make(True),
            self._make(False),  # contains again
        ]
        # All three were created with type="contains".
        assert failed_types(results) == ["contains"]


# ===== format_assertion_message =====


class TestFormatAssertionMessage:
    def test_pass_uses_check_mark(self) -> None:
        results = run_assertions("hi", None, 0.0,
                                 [{"type": "contains", "value": "hi"}])
        msg = format_assertion_message(results[0])
        assert PASS_MARK in msg

    def test_fail_uses_cross_mark(self) -> None:
        results = run_assertions("hi", None, 0.0,
                                 [{"type": "contains", "value": "bye"}])
        msg = format_assertion_message(results[0])
        assert FAIL_MARK in msg

    def test_error_uses_warn_mark(self) -> None:
        results = run_assertions("hi", None, 0.0,
                                 [{"type": "regex", "value": "[unclosed"}])
        msg = format_assertion_message(results[0])
        assert ERROR_MARK in msg
