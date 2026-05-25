"""Deterministic assertions for batch evaluation.

Nine assertion types implement mechanical pass/fail checks over LLM outputs.
The whole point is that "deterministic" means: given the same output, the
result is always the same - no LLM judge in the loop, no fuzziness. Useful
for prompt regression tests in CI/CD.

The dispatcher is data-driven: a `dict[str, callable]` maps each type to
its check function. Adding a tenth type means adding one entry plus one
function.

`jsonschema` is an OPTIONAL dependency. Eight of nine assertion types work
without it; only `json_schema` needs it. The import happens inside the
check function so:
    a) tests can simulate a missing install with monkeypatch
    b) users who don't need schema validation don't pay the import cost
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cli_modelarium.exceptions import AssertionConfigError


class AssertionType(str, Enum):
    """Supported assertion types. `str` subclass so values JSON-serialize natively."""

    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    EQUALS = "equals"
    JSON_VALID = "json_valid"
    JSON_SCHEMA = "json_schema"
    MIN_LENGTH_CHARS = "min_length_chars"
    MAX_LENGTH_CHARS = "max_length_chars"
    LATENCY_UNDER = "latency_under"
    COST_UNDER = "cost_under"


_ASSERTION_VALUES: set[str] = {t.value for t in AssertionType}

# json_valid is the only type that doesn't need a `value` field.
_NO_VALUE_TYPES: set[str] = {AssertionType.JSON_VALID.value}

# Display markers. Prefer Unicode by default; ASCII fallbacks available
# for environments with terminal encoding issues or downstream tools that
# choke on Unicode.
PASS_MARK = "✓"  # ✓
FAIL_MARK = "✗"  # ✗
ERROR_MARK = "⚠"  # ⚠
PASS_MARK_ASCII = "PASS"
FAIL_MARK_ASCII = "FAIL"
ERROR_MARK_ASCII = "ERR"


@dataclass
class AssertionConfig:
    """A validated assertion configuration ready to run."""

    type: str
    value: Any = None
    case_sensitive: bool = True


@dataclass
class AssertionResult:
    """The outcome of running one assertion against one output.

    Semantics:
        passed=True, error=None        - assertion ran and succeeded
        passed=False, error=None       - assertion ran and the output failed it
        passed=False, error="..."      - assertion COULDN'T run (config error,
                                         missing dependency, regex compile fail)

    The exit-code logic in cli.py treats `error`-set results as neither pass
    nor fail - they're surfaced but don't fail the build.
    """

    type: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    error: str | None = None


# ===== config parsing =====


def parse_assertion_config(raw: Any) -> AssertionConfig:
    """Validate a raw dict from a JSON batch file's `assertions` array.

    Raises AssertionConfigError with the offending config in the message so
    callers can surface it to the user.
    """
    if not isinstance(raw, dict):
        raise AssertionConfigError(
            f"Assertion config must be an object, got "
            f"{type(raw).__name__}: {raw!r}"
        )

    raw_type = raw.get("type")
    if raw_type is None:
        raise AssertionConfigError(
            f"Assertion config missing required 'type' field: {raw!r}"
        )
    if not isinstance(raw_type, str):
        raise AssertionConfigError(
            f"Assertion 'type' must be a string, got "
            f"{type(raw_type).__name__}: {raw!r}"
        )
    if raw_type not in _ASSERTION_VALUES:
        supported = ", ".join(sorted(_ASSERTION_VALUES))
        raise AssertionConfigError(
            f"Unknown assertion type {raw_type!r}. Supported: {supported}"
        )

    if raw_type not in _NO_VALUE_TYPES and "value" not in raw:
        raise AssertionConfigError(
            f"Assertion type {raw_type!r} requires a 'value' field: {raw!r}"
        )

    case_sensitive_raw = raw.get("case_sensitive", True)
    case_sensitive = bool(case_sensitive_raw)

    return AssertionConfig(
        type=raw_type,
        value=raw.get("value"),
        case_sensitive=case_sensitive,
    )


# ===== runner =====


def run_assertions(
    output: str,
    latency_ms: float | None,
    cost_usd: float,
    assertions: list[dict[str, Any]],
) -> list[AssertionResult]:
    """Run each assertion in order. Never raises.

    Config errors become AssertionResult with `error` set, preserving the
    type-name for the row's display while signalling that this one should
    NOT count toward pass/fail tallies.
    """
    results: list[AssertionResult] = []
    for raw in assertions:
        try:
            config = parse_assertion_config(raw)
        except AssertionConfigError as e:
            raw_type = (
                raw.get("type", "<unknown>") if isinstance(raw, dict) else "<unknown>"
            )
            results.append(
                AssertionResult(
                    type=str(raw_type),
                    passed=False,
                    expected=None,
                    actual=None,
                    message="invalid assertion config",
                    error=str(e),
                )
            )
            continue

        try:
            result = _dispatch(config, output, latency_ms, cost_usd)
        except Exception as e:  # noqa: BLE001 - defensive; impls shouldn't raise
            result = AssertionResult(
                type=config.type,
                passed=False,
                expected=config.value,
                actual=None,
                message="assertion check raised unexpectedly",
                error=f"{type(e).__name__}: {e}",
            )
        results.append(result)
    return results


def _dispatch(
    config: AssertionConfig,
    output: str,
    latency_ms: float | None,
    cost_usd: float,
) -> AssertionResult:
    """Route a parsed config to its check function."""
    checker = _CHECKERS.get(config.type)
    if checker is None:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=None,
            message="unknown assertion type",
            error=f"no checker registered for type: {config.type}",
        )
    return checker(config, output, latency_ms, cost_usd)


# ===== aggregation helpers =====


def all_passed(results: list[AssertionResult]) -> bool:
    """True if every result either passed or errored (error rows don't count as fail).

    Empty list passes vacuously.
    """
    return all(r.passed or r.error is not None for r in results)


def count_passed(results: list[AssertionResult]) -> tuple[int, int]:
    """Return (passed_count, total_definitive_count).

    Definitive = passed=True OR (passed=False AND error is None). Results
    with `error` set are EXCLUDED from both numerator and denominator -
    they couldn't run, so they aren't a verdict either way.
    """
    passed = sum(1 for r in results if r.passed and r.error is None)
    definitive = sum(1 for r in results if r.error is None)
    return passed, definitive


def count_failed(results: list[AssertionResult]) -> int:
    """Number of definitive failures (excludes error rows)."""
    return sum(1 for r in results if not r.passed and r.error is None)


def failed_types(results: list[AssertionResult]) -> list[str]:
    """Distinct types of definitively-failed assertions, preserving first-occurrence order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in results:
        if not r.passed and r.error is None and r.type not in seen_set:
            seen.append(r.type)
            seen_set.add(r.type)
    return seen


def format_assertion_message(result: AssertionResult) -> str:
    """Render an AssertionResult as a one-line human-readable string."""
    if result.error is not None:
        return f"{ERROR_MARK} {result.type}: {result.error}"
    mark = PASS_MARK if result.passed else FAIL_MARK
    return f"{mark} {result.type}: {result.message}"


# ===== assertion implementations =====


def _strip_code_fence(text: str) -> str:
    """Strip an outer ```json (or bare ```) fence if present. Otherwise unchanged."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    # Drop the opening fence line (handles both ``` and ```json).
    lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def _check_contains(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    value = str(config.value)
    if config.case_sensitive:
        found = value in output
    else:
        found = value.lower() in output.lower()
    return AssertionResult(
        type=config.type,
        passed=found,
        expected=value,
        actual="found" if found else "not found",
        message=(
            f"contains {value!r}"
            if found
            else f"output does not contain {value!r}"
        ),
    )


def _check_not_contains(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    value = str(config.value)
    if config.case_sensitive:
        found = value in output
    else:
        found = value.lower() in output.lower()
    return AssertionResult(
        type=config.type,
        passed=not found,
        expected=value,
        actual="found" if found else "not found",
        message=(
            f"output does not contain {value!r}"
            if not found
            else f"forbidden substring {value!r} appears in output"
        ),
    )


def _check_regex(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    pattern = str(config.value)
    flags = re.DOTALL | (0 if config.case_sensitive else re.IGNORECASE)
    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=pattern,
            actual=None,
            message="invalid regex pattern",
            error=f"re.error: {e}",
        )
    match = compiled.search(output)
    return AssertionResult(
        type=config.type,
        passed=match is not None,
        expected=pattern,
        actual=(match.group(0)[:80] if match else "no match"),
        message=(
            f"regex {pattern!r} matched"
            if match
            else f"regex {pattern!r} did not match"
        ),
    )


def _check_equals(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    value = str(config.value)
    # Normalize line endings before stripping so cross-platform comparisons work.
    actual_norm = output.replace("\r\n", "\n").strip()
    expected_norm = value.replace("\r\n", "\n").strip()
    actual_cmp = actual_norm if config.case_sensitive else actual_norm.lower()
    expected_cmp = expected_norm if config.case_sensitive else expected_norm.lower()
    passed = actual_cmp == expected_cmp
    return AssertionResult(
        type=config.type,
        passed=passed,
        expected=value,
        actual=actual_norm[:200],
        message=(
            f"output equals expected"
            if passed
            else f"output does not equal expected"
        ),
    )


def _check_json_valid(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    text = _strip_code_fence(output) if output else ""
    if not text:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=None,
            actual="(empty)",
            message="output is empty",
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=None,
            actual=text[:100],
            message=f"output is not valid JSON: {e.msg}",
        )
    return AssertionResult(
        type=config.type,
        passed=True,
        expected=None,
        actual=type(data).__name__,
        message="output parsed as valid JSON",
    )


def _check_json_schema(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    # Fresh import inside the function so tests can simulate a missing
    # install by patching sys.modules / __import__.
    try:
        import jsonschema
    except ImportError:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=None,
            message="jsonschema not installed",
            error="jsonschema not installed. Run: pip install cli-modelarium[schema]",
        )

    text = _strip_code_fence(output) if output else ""
    if not text:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual="(empty)",
            message="output is empty",
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=text[:100],
            message=f"output is not valid JSON: {e.msg}",
        )

    schema = config.value
    if not isinstance(schema, dict):
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=schema,
            actual=data,
            message="schema must be an object",
            error=f"schema value must be a JSON object, got {type(schema).__name__}",
        )

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=schema,
            actual=data,
            message=f"schema violation at {path}: {e.message}",
        )
    except jsonschema.SchemaError as e:
        # The schema ITSELF is malformed - this is a config error, not a fail.
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=schema,
            actual=data,
            message="schema is malformed",
            error=f"jsonschema schema error: {e.message}",
        )
    except Exception as e:  # noqa: BLE001 - any other jsonschema quirk
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=schema,
            actual=data,
            message="jsonschema raised unexpectedly",
            error=f"{type(e).__name__}: {e}",
        )

    return AssertionResult(
        type=config.type,
        passed=True,
        expected=schema,
        actual=data,
        message="output matches schema",
    )


def _check_min_length_chars(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    try:
        limit = int(config.value)
    except (TypeError, ValueError):
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=None,
            message="min_length_chars value must be an integer",
            error=f"value must be a number, got {config.value!r}",
        )
    actual = len(output)
    passed = actual >= limit
    return AssertionResult(
        type=config.type,
        passed=passed,
        expected=limit,
        actual=actual,
        message=(
            f"{actual} chars (min {limit})"
            if passed
            else f"only {actual} chars (need at least {limit})"
        ),
    )


def _check_max_length_chars(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    try:
        limit = int(config.value)
    except (TypeError, ValueError):
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=None,
            message="max_length_chars value must be an integer",
            error=f"value must be a number, got {config.value!r}",
        )
    actual = len(output)
    passed = actual <= limit
    return AssertionResult(
        type=config.type,
        passed=passed,
        expected=limit,
        actual=actual,
        message=(
            f"{actual} chars (max {limit})"
            if passed
            else f"{actual} chars exceeds max {limit}"
        ),
    )


def _check_latency_under(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    try:
        limit = float(config.value)
    except (TypeError, ValueError):
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=None,
            message="latency_under value must be a number",
            error=f"value must be a number, got {config.value!r}",
        )
    if latency_ms is None:
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=limit,
            actual=None,
            message="latency unavailable",
            error="latency was not measured for this call",
        )
    passed = latency_ms < limit
    return AssertionResult(
        type=config.type,
        passed=passed,
        expected=limit,
        actual=round(latency_ms, 1),
        message=(
            f"{latency_ms:.1f}ms < {limit}ms"
            if passed
            else f"{latency_ms:.1f}ms exceeds {limit}ms limit"
        ),
    )


def _check_cost_under(
    config: AssertionConfig, output: str, latency_ms: float | None, cost_usd: float
) -> AssertionResult:
    try:
        limit = float(config.value)
    except (TypeError, ValueError):
        return AssertionResult(
            type=config.type,
            passed=False,
            expected=config.value,
            actual=None,
            message="cost_under value must be a number",
            error=f"value must be a number, got {config.value!r}",
        )
    passed = cost_usd < limit
    return AssertionResult(
        type=config.type,
        passed=passed,
        expected=limit,
        actual=cost_usd,
        message=(
            f"${cost_usd:.6f} < ${limit:.6f}"
            if passed
            else f"${cost_usd:.6f} exceeds ${limit:.6f} limit"
        ),
    )


# Dispatcher: assertion type value -> check function.
_CHECKERS: dict[str, Any] = {
    AssertionType.CONTAINS.value: _check_contains,
    AssertionType.NOT_CONTAINS.value: _check_not_contains,
    AssertionType.REGEX.value: _check_regex,
    AssertionType.EQUALS.value: _check_equals,
    AssertionType.JSON_VALID.value: _check_json_valid,
    AssertionType.JSON_SCHEMA.value: _check_json_schema,
    AssertionType.MIN_LENGTH_CHARS.value: _check_min_length_chars,
    AssertionType.MAX_LENGTH_CHARS.value: _check_max_length_chars,
    AssertionType.LATENCY_UNDER.value: _check_latency_under,
    AssertionType.COST_UNDER.value: _check_cost_under,
}
