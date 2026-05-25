"""Hallucination detection preset.

A specialized judge configuration that:

    * Replaces the default `JUDGE_PROMPT_TEMPLATE` with a hallucination-
      focused prompt that asks for a Low/Medium/High risk classification
      alongside the 1-10 score.
    * Optionally substitutes a `{reference_check}` block with either
      reference facts (when --expected-facts is provided) or a "use your
      own knowledge" instruction.
    * Uses `parse_hallucination_response` instead of `parse_judge_response`
      so the JudgeScore picks up the `risk_level` field.
    * Worst-case aggregates risk_level across panel judges (any High wins).

This module deliberately reuses the Phase 8 judging machinery rather than
forking it - the only judging-side additions are the optional `risk_level`
field on `JudgeScore` and `aggregated_risk_level` on `JudgeResult`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from cli_modelarium.exceptions import BatchValidationError, ModelariumError
from cli_modelarium.io_safety import safe_input_path, split_escaped_csv
from cli_modelarium.judging import (
    JudgeResult,
    _extract_first_json_object,
    _strip_code_fence,
    parse_judge_response,
)

# ===== prompt templates =====

# The full evaluation rubric. `{reference_check}` is substituted with either
# HALLUCINATION_WITH_FACTS (when facts are provided) or
# HALLUCINATION_WITHOUT_FACTS. The literal `{"score": ...}` braces in the
# response-format line do NOT need escaping because we use `.replace()`,
# not `.format()`.
HALLUCINATION_CRITERIA_BASE = """\
Evaluate the response for hallucination risk. Check:
1. Factual accuracy of claimed facts (dates, names, statistics)
2. Whether the response invents specific details (fake citations, quotes, sources)
3. Whether the response contradicts commonly accepted facts
4. Whether the response hedges appropriately on uncertain claims
5. {reference_check}

Rate hallucination risk on 1-10:
- 10 = All claims accurate and well-hedged
- 7-9 = Mostly accurate with minor unhedged claims
- 4-6 = Some incorrect or unverifiable claims
- 1-3 = Multiple incorrect claims or fabricated details

Respond ONLY with JSON: {"score": <1-10>, "risk_level": "Low" | "Medium" | "High", "reasoning": "<one sentence>"}"""

HALLUCINATION_WITH_FACTS = """\
Verify whether the response is consistent with these known facts:
{facts}
Flag any contradictions or omissions of these facts."""

HALLUCINATION_WITHOUT_FACTS = """\
Evaluate against the judge model's own factual knowledge.
Lower scores for any claims that cannot be verified."""

# Wrapper template that build_judge_prompt fills with {criteria}/{prompt}/{response}.
# Mirrors JUDGE_PROMPT_TEMPLATE's shape but drops the trailing "Respond with..."
# instructions because HALLUCINATION_CRITERIA_BASE already includes them.
HALLUCINATION_TEMPLATE = """\
{criteria}

Original prompt: {prompt}
Response to evaluate: {response}"""

# Size limit for --expected-facts-file. Matches SYSTEM_PROMPT_MAX_BYTES.
EXPECTED_FACTS_MAX_BYTES = 1_000_000

# The three permitted risk_level classifications.
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"
_RISK_LEVELS = {RISK_LOW, RISK_MEDIUM, RISK_HIGH}

# Extension to the ToS panel, appended when --check-hallucination is active.
HALLUCINATION_TOS_EXTENSION = (
    "Hallucination detection uses LLM-as-judge methodology with reference "
    "facts when provided. Accuracy varies by model and topic. Treat results "
    "as guidance, not ground truth."
)


# ===== facts handling =====


def parse_facts_csv(value: str) -> list[str]:
    """Split a comma-separated facts string. Same `\\,` escape as elsewhere.

    Empty string returns []. Whitespace stripped, empty pieces dropped.
    """
    return split_escaped_csv(value)


def load_expected_facts(file_path: str) -> list[str]:
    """Load expected facts from a .txt (one per line) or .json (array) file.

    Returns [] for an empty file. Rejects duplicate facts (case-insensitive)
    with BatchValidationError. Unknown extension also raises.
    """
    path = safe_input_path(file_path, max_size_bytes=EXPECTED_FACTS_MAX_BYTES)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        facts = _load_facts_txt(path)
    elif suffix == ".json":
        facts = _load_facts_json(path)
    else:
        raise BatchValidationError(
            f"Cannot detect expected-facts file format from extension {suffix!r}.\n"
            f"  Supported: .txt (one fact per line) or .json (array of strings).\n"
            f"  At: {path}"
        )
    _reject_duplicates(facts, path=str(path))
    return facts


def _load_facts_txt(path) -> list[str]:
    """Parse a .txt facts file. Comments (`#`) and blank lines ignored."""
    text = path.read_text(encoding="utf-8-sig")
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return out


def _load_facts_json(path) -> list[str]:
    """Parse a .json facts file. Must be a top-level array of strings."""
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise BatchValidationError(
            f"Expected-facts JSON file must be an array at top level, "
            f"got {type(data).__name__!r}. At: {path}"
        )
    out: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise BatchValidationError(
                f"Expected-facts JSON element #{i} must be a string "
                f"(got {type(item).__name__!r}). At: {path}"
            )
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return out


def _reject_duplicates(facts: list[str], *, path: str) -> None:
    """Raise if `facts` contains any case-insensitive duplicate."""
    seen: dict[str, str] = {}
    for fact in facts:
        key = fact.lower()
        if key in seen:
            raise BatchValidationError(
                f"Duplicate expected fact (case-insensitive): "
                f"{seen[key]!r} and {fact!r}. At: {path}"
            )
        seen[key] = fact


# ===== criteria builder =====


def build_hallucination_criteria(facts: list[str] | None) -> list[str]:
    """Build the single-item criteria list passed to run_judging.

    The returned list has exactly one element: the filled
    HALLUCINATION_CRITERIA_BASE with `{reference_check}` substituted. When
    `facts` is provided, the WITH_FACTS template is used with facts
    rendered as a bullet list. When `facts` is None or empty, the
    WITHOUT_FACTS template is used.
    """
    if facts:
        bullets = "\n".join(f"  - {f}" for f in facts)
        reference_check = HALLUCINATION_WITH_FACTS.replace("{facts}", bullets)
    else:
        reference_check = HALLUCINATION_WITHOUT_FACTS
    filled = HALLUCINATION_CRITERIA_BASE.replace(
        "{reference_check}", reference_check
    )
    return [filled]


# ===== response parser =====


def parse_hallucination_response(text: str) -> dict:
    """Parse a hallucination-judge response into score/risk_level/reasoning.

    Wraps `parse_judge_response` for score+reasoning+parse_error, then
    extracts `risk_level` from the same JSON. The risk_level field:

        * MUST be one of "Low" / "Medium" / "High" (case-insensitive,
          normalized to title case).
        * Invalid values (e.g. "Critical") become parse_error.
        * Missing risk_level falls back to derivation from score
          (1-3 -> High, 4-6 -> Medium, 7-10 -> Low). If score is also
          missing, risk_level stays None.
    """
    base = parse_judge_response(text)

    risk_level: str | None = None
    risk_error: str | None = None

    parsed = _try_parse_object(text)
    if parsed is not None and "risk_level" in parsed:
        raw_rl = parsed["risk_level"]
        if isinstance(raw_rl, str):
            normalized = raw_rl.strip().title()
            if normalized in _RISK_LEVELS:
                risk_level = normalized
            else:
                risk_error = (
                    f"invalid risk_level {raw_rl!r}; "
                    f"must be one of {sorted(_RISK_LEVELS)}"
                )
        else:
            risk_error = (
                f"risk_level must be a string, got {type(raw_rl).__name__}"
            )
    else:
        # Not in the JSON - derive from score per spec.
        risk_level = risk_level_from_score(base["score"])

    parse_error = base["parse_error"] or risk_error
    return {
        "score": base["score"],
        "reasoning": base["reasoning"],
        "risk_level": risk_level,
        "parse_error": parse_error,
    }


def _try_parse_object(text: str) -> dict | None:
    """Run the same code-fence + extract-first-json strategies as parse_judge_response."""
    text = text.strip()
    if not text:
        return None
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(text)
    if isinstance(parsed, dict):
        return parsed
    return None


def risk_level_from_score(score: int | None) -> str | None:
    """Map a 1-10 score to a risk_level. Deterministic per spec.

    1-3 = High, 4-6 = Medium, 7-10 = Low. Out-of-range scores or None
    return None - we don't guess.
    """
    if score is None:
        return None
    if not isinstance(score, int):
        return None
    if score < 1 or score > 10:
        return None
    if score <= 3:
        return RISK_HIGH
    if score <= 6:
        return RISK_MEDIUM
    return RISK_LOW


# ===== aggregation =====


def aggregate_risk_levels(judges: list) -> str | None:
    """Worst-wins aggregation across a panel's per-judge risk_levels.

    Any High -> "High". No High but any Medium -> "Medium". All Low -> "Low".
    Empty (no judges with a risk_level) -> None.
    """
    levels = [j.risk_level for j in judges if getattr(j, "risk_level", None)]
    if not levels:
        return None
    if RISK_HIGH in levels:
        return RISK_HIGH
    if RISK_MEDIUM in levels:
        return RISK_MEDIUM
    return RISK_LOW


def annotate_risk_levels(judge_results: list[JudgeResult]) -> None:
    """Populate `aggregated_risk_level` on each result in place.

    Called by cli.py after run_judging() returns; mutates the
    JudgeResults so downstream formatters see the aggregated value.
    """
    for jr in judge_results:
        jr.aggregated_risk_level = aggregate_risk_levels(jr.judges)


# ===== config dataclass =====


@dataclass
class HallucinationConfig:
    """Resolved configuration for a hallucination run."""

    criteria: list[str]
    template: str
    # facts is None when no facts were supplied (uses WITHOUT_FACTS template).
    # Empty list is normalized to None at resolve time.
    facts: list[str] | None = None


def resolve_hallucination_config(
    *,
    check_hallucination: bool,
    expected_facts: str | None,
    expected_facts_file: str | None,
    hallucination_template: str | None,
    judge_models_present: bool,
) -> HallucinationConfig | None:
    """Validate the hallucination flag combination and return a config or None.

    Returns None when --check-hallucination is NOT set (the standard
    judging path applies). Raises ModelariumError for invalid combinations
    so cli.py can surface them as click.UsageError / exit 2.
    """
    from cli_modelarium.io_safety import load_system_prompt

    if not check_hallucination:
        if expected_facts or expected_facts_file or hallucination_template:
            raise BatchValidationError(
                "--expected-facts, --expected-facts-file, and --hallucination-template "
                "require --check-hallucination."
            )
        return None

    if not judge_models_present:
        raise BatchValidationError(
            "--check-hallucination requires --judge or --judges - "
            "hallucination scoring is implemented as a judge configuration."
        )

    if expected_facts and expected_facts_file:
        raise BatchValidationError(
            "--expected-facts and --expected-facts-file are mutually exclusive."
        )

    if hallucination_template and (expected_facts or expected_facts_file):
        raise BatchValidationError(
            "--hallucination-template is mutually exclusive with "
            "--expected-facts / --expected-facts-file - custom templates "
            "are responsible for their own fact-checking structure."
        )

    facts: list[str] | None = None
    if expected_facts_file:
        facts = load_expected_facts(expected_facts_file)
    elif expected_facts:
        facts = parse_facts_csv(expected_facts)
    # Empty CSV -> [] -> normalize to None so we use WITHOUT_FACTS.
    if facts is not None and len(facts) == 0:
        facts = None

    if hallucination_template:
        # User-supplied evaluation rubric. Replaces the BASE constant entirely.
        # The custom text is treated as a single criterion; the standard
        # HALLUCINATION_TEMPLATE wrapper still provides {prompt}/{response} slots.
        custom_text = load_system_prompt(hallucination_template)
        criteria = [custom_text]
    else:
        criteria = build_hallucination_criteria(facts)

    return HallucinationConfig(
        criteria=criteria,
        template=HALLUCINATION_TEMPLATE,
        facts=facts,
    )
