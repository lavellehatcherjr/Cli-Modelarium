"""LLM-as-a-judge scoring.

Three concerns:

    1. Prompt building - `build_judge_prompt` formats a judge template with
       criteria, the original user prompt, and the response to evaluate.
    2. Response parsing - `parse_judge_response` is forgiving about wrapping
       (markdown fences, leading/trailing text) and returns a structured
       result with parse_error set when JSON couldn't be extracted.
    3. Orchestration - `run_judging` runs every (state, judge) pairing in
       parallel under per-judge-provider semaphores, with self-evaluation
       auto-skip and aggregation across the panel.

The judge call itself uses `temperature=0.0` always - deterministic scoring
makes the panel comparable across runs.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel

from cli_modelarium.exceptions import ModelariumError
from cli_modelarium.models_registry import get_provider_for_model
from cli_modelarium.providers.base import BaseProvider
from cli_modelarium.security import redact_secrets
from cli_modelarium.streaming import StreamState

# Defaults the user picked in the build prompt.
DEFAULT_CRITERIA: tuple[str, ...] = (
    "Accuracy: Is the response factually correct?",
    "Helpfulness: Does it actually address what was asked?",
    "Clarity: Is the explanation clear and well-structured?",
)

# The exact wording from the build prompt. `{criteria}`, `{prompt}`, and
# `{response}` are the three substitution points. Double `{{` / `}}` escape
# the JSON braces so str.format doesn't try to interpret them.
JUDGE_PROMPT_TEMPLATE = """You are evaluating an AI assistant's response. \
Rate the response on a scale of 1-10 for overall quality based on these criteria:

{criteria}

Original prompt: {prompt}
Response to evaluate: {response}

Respond with ONLY a JSON object in this exact format:
{{"score": <1-10 integer>, "reasoning": "<one sentence explanation>"}}

Do not include any other text."""

# Printed before every judging run unless --no-judge-tos is passed. See
# print_tos_disclosure() for the yellow Rich panel.
TOS_DISCLOSURE = (
    "Reminder: Judge scores are for evaluation only.\n"
    "Do not use to train, fine-tune, or develop competing AI models\n"
    "(violates provider ToS)."
)

# Anthropic / OpenAI both clamp 1-10 in their guidance; outside that range
# we treat as a parse failure rather than silently clamping.
SCORE_MIN = 1
SCORE_MAX = 10


@dataclass
class JudgeScore:
    """One judge model's verdict on one response."""

    model: str
    score: int | None
    reasoning: str
    cost_usd: float
    latency_ms: float
    parse_error: str | None = None
    # Phase 10 hallucination preset: when the custom parser extracts a
    # Low/Medium/High classification, it lands here. None for normal judging.
    risk_level: str | None = None


@dataclass
class JudgeResult:
    """Aggregated panel verdict on one response."""

    judges: list[JudgeScore] = field(default_factory=list)
    average_score: float | None = None
    std_dev: float | None = None
    # Models that were skipped due to self-evaluation - tracked so callers
    # can surface that information in the display.
    skipped_models: list[str] = field(default_factory=list)
    # Phase 10 hallucination preset: worst-case aggregation across the panel
    # (any High -> High; else any Medium -> Medium; else Low; else None).
    # Populated by hallucination.annotate_risk_levels(), not _aggregate.
    aggregated_risk_level: str | None = None


# ===== Response parsing =====


def parse_judge_response(text: str) -> dict:
    """Parse a judge model's response into {score, reasoning, parse_error}.

    Tries several strategies in order:
        1. Strip an outer markdown code fence (```json ... ``` or ``` ... ```)
        2. Try `json.loads` on the result.
        3. If that fails, search for the first balanced `{...}` block.

    Returns a dict with the keys:
        score (int | None): 1..10 or None on any parse failure.
        reasoning (str): the judge's reasoning, or "".
        parse_error (str | None): set when we couldn't extract a valid score.
    """
    text = text.strip()
    if not text:
        return {"score": None, "reasoning": "", "parse_error": "empty response"}

    candidate = _strip_code_fence(text)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to extracting the first {...} object from anywhere in the
        # text. This covers responses with leading/trailing commentary.
        data = _extract_first_json_object(text)
        if data is None:
            return {
                "score": None,
                "reasoning": "",
                "parse_error": "could not extract JSON object",
            }

    if not isinstance(data, dict):
        return {
            "score": None,
            "reasoning": "",
            "parse_error": f"expected JSON object, got {type(data).__name__}",
        }

    reasoning_raw = data.get("reasoning", "")
    reasoning = str(reasoning_raw) if reasoning_raw is not None else ""

    if "score" not in data:
        return {
            "score": None,
            "reasoning": reasoning,
            "parse_error": "missing 'score' field",
        }

    score, err = _coerce_score(data["score"])
    return {"score": score, "reasoning": reasoning, "parse_error": err}


def _strip_code_fence(text: str) -> str:
    """Strip a leading ```json (or bare ```) fence and matching trailing ```."""
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    # Drop the opening fence line entirely (handles ``` and ```json).
    lines = lines[1:]
    # Drop trailing ``` lines.
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> dict | None:
    """Find the first balanced {...} block in `text` and parse it as JSON.

    Returns None if no balanced block parses successfully.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    string_char = ""
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if c == "\\":
                escape_next = True
                continue
            if c == string_char:
                in_string = False
            continue
        if c in ('"', "'"):
            in_string = True
            string_char = c
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                    return None
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_score(raw: object) -> tuple[int | None, str | None]:
    """Coerce a raw score value to an int in [SCORE_MIN, SCORE_MAX].

    Accepts int, float (rounded to nearest), or numeric string. Booleans are
    rejected (they are int subclasses but make no sense as scores). Returns
    (score, error_message_or_None).
    """
    if isinstance(raw, bool):
        return None, "score is boolean, expected number"
    if isinstance(raw, int):
        if SCORE_MIN <= raw <= SCORE_MAX:
            return raw, None
        return None, f"score {raw} out of range {SCORE_MIN}-{SCORE_MAX}"
    if isinstance(raw, float):
        coerced = int(round(raw))
        if SCORE_MIN <= coerced <= SCORE_MAX:
            return coerced, None
        return None, f"score {raw} out of range {SCORE_MIN}-{SCORE_MAX}"
    if isinstance(raw, str):
        try:
            as_float = float(raw.strip())
        except (ValueError, AttributeError):
            return None, f"score is non-numeric string: {raw!r}"
        coerced = int(round(as_float))
        if SCORE_MIN <= coerced <= SCORE_MAX:
            return coerced, None
        return None, f"score {as_float} out of range {SCORE_MIN}-{SCORE_MAX}"
    return None, f"score has unsupported type {type(raw).__name__}"


# ===== Prompt building =====


def build_judge_prompt(
    original_prompt: str,
    response_to_eval: str,
    criteria: list[str],
    template: str = JUDGE_PROMPT_TEMPLATE,
) -> str:
    """Format the judge prompt template with criteria, prompt, and response.

    Criteria are joined as bullet points. The user's prompt and the response
    are passed verbatim - they are NOT subject to format string interpolation
    (we use `.replace` over the template's three placeholders rather than
    `str.format`, so curly braces in the user's text don't blow up).
    """
    if not criteria:
        criteria = list(DEFAULT_CRITERIA)
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    # Don't use .format() - it would explode on any '{...}' in the response
    # text and is a needless template-injection vector. Direct replacement is
    # safe and the template has only these three placeholders.
    out = template.replace("{criteria}", criteria_text)
    out = out.replace("{prompt}", original_prompt)
    out = out.replace("{response}", response_to_eval)
    return out


# ===== Single judge call =====


async def score_with_judge(
    judge_provider: BaseProvider,
    judge_model: str,
    original_prompt: str,
    response_to_eval: str,
    criteria: list[str],
    template: str = JUDGE_PROMPT_TEMPLATE,
    response_parser: Callable[[str], dict] | None = None,
) -> JudgeScore:
    """Run one judge model on one response, returning a JudgeScore.

    `response_parser` defaults to `parse_judge_response`. Phase 10's
    hallucination preset passes `parse_hallucination_response` here to
    extract the risk_level field alongside score/reasoning.

    Errors (network failures, parse failures, anything) are captured into
    JudgeScore.parse_error - we do NOT raise out of this function so a
    panel of judges keeps functioning when one of them goes wrong.
    """
    parser = response_parser or parse_judge_response

    judge_prompt = build_judge_prompt(original_prompt, response_to_eval, criteria, template)
    start = time.monotonic()
    try:
        result = await judge_provider.complete(
            prompt=judge_prompt,
            model=judge_model,
            temperature=0.0,
        )
    except ModelariumError as e:
        return JudgeScore(
            model=judge_model,
            score=None,
            reasoning="",
            cost_usd=0.0,
            latency_ms=(time.monotonic() - start) * 1000,
            parse_error=f"judge call failed: {redact_secrets(str(e))}",
        )
    except Exception as e:  # noqa: BLE001 - become a parse_error, not a crash
        return JudgeScore(
            model=judge_model,
            score=None,
            reasoning="",
            cost_usd=0.0,
            latency_ms=(time.monotonic() - start) * 1000,
            parse_error=f"judge call failed: {redact_secrets(str(e))}",
        )

    parsed = parser(result.output)
    return JudgeScore(
        model=judge_model,
        score=parsed.get("score"),
        reasoning=parsed.get("reasoning", ""),
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        parse_error=parsed.get("parse_error"),
        risk_level=parsed.get("risk_level"),
    )


# ===== Orchestration =====


async def run_judging(
    items: list[tuple[StreamState, str]],
    judge_models: list[str],
    criteria: list[str] | None,
    provider_factory: Callable[[str], BaseProvider],
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
    response_parser: Callable[[str], dict] | None = None,
    skip_self_eval: bool = True,
    concurrency: int = 5,
) -> list[JudgeResult]:
    """Run every (state, judge) pairing in parallel and aggregate.

    `items` is a list of (StreamState, original_user_prompt) tuples. Returns
    a list of `JudgeResult` parallel to `items`. State with errors (failed
    main calls) get an empty JudgeResult - we don't waste judge calls on
    them.

    `skip_self_eval=True` (the default) drops the pairing where a judge's
    model ID equals the state's model ID. The skipped model is recorded in
    the resulting `JudgeResult.skipped_models` so the display layer can
    surface it.

    Empty `criteria` falls back to `DEFAULT_CRITERIA`.
    """
    if not items or not judge_models:
        return [JudgeResult() for _ in items]

    criteria = list(criteria) if criteria else list(DEFAULT_CRITERIA)

    # One semaphore + one instance per UNIQUE judge provider (not per judge
    # model). Lets us keep rate-limit hygiene when a single provider hosts
    # several judges in the panel.
    judge_providers_needed: dict[str, str] = {m: get_provider_for_model(m) for m in judge_models}
    provider_names = sorted(set(judge_providers_needed.values()))
    instances: dict[str, BaseProvider] = {n: provider_factory(n) for n in provider_names}
    semaphores: dict[str, asyncio.Semaphore] = {
        n: asyncio.Semaphore(concurrency) for n in provider_names
    }

    async def _score_with_sem(
        provider: BaseProvider,
        semaphore: asyncio.Semaphore,
        judge_model: str,
        original_prompt: str,
        response_text: str,
    ) -> JudgeScore:
        async with semaphore:
            return await score_with_judge(
                provider,
                judge_model,
                original_prompt,
                response_text,
                criteria,
                template,
                response_parser=response_parser,
            )

    async def _score_state(state: StreamState, original_prompt: str) -> JudgeResult:
        if state.error:
            # Don't judge a failed main call. Empty JudgeResult.
            return JudgeResult()

        tasks = []
        skipped: list[str] = []
        for judge_model in judge_models:
            if skip_self_eval and judge_model == state.model:
                skipped.append(judge_model)
                continue
            provider_name = judge_providers_needed[judge_model]
            tasks.append(
                _score_with_sem(
                    instances[provider_name],
                    semaphores[provider_name],
                    judge_model,
                    original_prompt,
                    state.text,
                )
            )

        judges: list[JudgeScore] = list(await asyncio.gather(*tasks)) if tasks else []
        return _aggregate(judges, skipped_models=skipped)

    return list(await asyncio.gather(*[_score_state(state, prompt) for state, prompt in items]))


def _aggregate(judges: list[JudgeScore], *, skipped_models: list[str] | None = None) -> JudgeResult:
    """Compute average and std_dev across successfully-scored judges."""
    successful = [j for j in judges if j.score is not None]
    if successful:
        scores = [j.score for j in successful if j.score is not None]
        average = statistics.mean(scores) if scores else None
        std = statistics.stdev(scores) if len(scores) >= 2 else None
    else:
        average = None
        std = None
    return JudgeResult(
        judges=judges,
        average_score=average,
        std_dev=std,
        skipped_models=list(skipped_models or []),
    )


# ===== UI helpers =====


def print_tos_disclosure(console: Console) -> None:
    """Print the judge-use ToS reminder as a yellow panel.

    Always shown when judging is enabled unless --no-judge-tos is passed.
    The wording is the build prompt's verbatim text.
    """
    console.print(
        Panel(
            TOS_DISCLOSURE,
            title="Judge ToS",
            border_style="yellow",
        )
    )


def total_judge_cost(results: list[JudgeResult]) -> float:
    """Sum the cost across every JudgeScore in every JudgeResult."""
    return sum(j.cost_usd for r in results for j in r.judges)


def total_judge_calls(results: list[JudgeResult]) -> int:
    """Count every JudgeScore (successful or failed) across all results."""
    return sum(len(r.judges) for r in results)
