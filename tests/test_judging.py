"""Tests for cli_modelarium.judging.

Three layers:
    1. parse_judge_response - parser edge cases (fences, wrapping, coercion)
    2. score_with_judge - one judge call with mocked provider
    3. run_judging - orchestration: self-eval-skip, panel aggregation, cost
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from typing import Any

import pytest

from cli_modelarium.exceptions import ProviderError
from cli_modelarium.judging import (
    DEFAULT_CRITERIA,
    JudgeScore,
    _aggregate,
    _coerce_score,
    _extract_first_json_object,
    _strip_code_fence,
    build_judge_prompt,
    parse_judge_response,
    run_judging,
    score_with_judge,
    total_judge_calls,
    total_judge_cost,
)
from cli_modelarium.providers.base import BaseProvider, CompletionResult, OnChunk
from cli_modelarium.streaming import StreamState

# ===== parse_judge_response =====


class TestParseJudgeResponse:
    def test_bare_json(self) -> None:
        result = parse_judge_response('{"score": 8, "reasoning": "good"}')
        assert result["score"] == 8
        assert result["reasoning"] == "good"
        assert result["parse_error"] is None

    def test_json_fenced_with_language_tag(self) -> None:
        text = '```json\n{"score": 7, "reasoning": "ok"}\n```'
        result = parse_judge_response(text)
        assert result["score"] == 7

    def test_json_fenced_without_language_tag(self) -> None:
        text = '```\n{"score": 9, "reasoning": "great"}\n```'
        result = parse_judge_response(text)
        assert result["score"] == 9

    def test_leading_text_before_json(self) -> None:
        text = 'Here is my evaluation:\n{"score": 6, "reasoning": "passable"}'
        result = parse_judge_response(text)
        assert result["score"] == 6
        assert result["reasoning"] == "passable"

    def test_trailing_text_after_json(self) -> None:
        text = '{"score": 5, "reasoning": "ok"} (final answer)'
        result = parse_judge_response(text)
        assert result["score"] == 5

    def test_malformed_json_returns_parse_error(self) -> None:
        result = parse_judge_response("not valid json {")
        assert result["score"] is None
        assert result["parse_error"] is not None

    def test_empty_response(self) -> None:
        result = parse_judge_response("")
        assert result["score"] is None
        assert "empty" in (result["parse_error"] or "").lower()

    def test_whitespace_only(self) -> None:
        result = parse_judge_response("   \n  ")
        assert result["score"] is None

    def test_missing_score_field(self) -> None:
        result = parse_judge_response('{"reasoning": "no number"}')
        assert result["score"] is None
        assert "score" in (result["parse_error"] or "").lower()
        # Reasoning is still surfaced even when score is missing.
        assert result["reasoning"] == "no number"

    def test_score_out_of_range_low(self) -> None:
        result = parse_judge_response('{"score": 0, "reasoning": "x"}')
        assert result["score"] is None
        assert "out of range" in (result["parse_error"] or "").lower()

    def test_score_out_of_range_high(self) -> None:
        result = parse_judge_response('{"score": 11, "reasoning": "x"}')
        assert result["score"] is None
        assert "out of range" in (result["parse_error"] or "").lower()

    def test_score_as_string_numeric_accepted(self) -> None:
        """Some judge models return the score as a quoted number."""
        result = parse_judge_response('{"score": "8", "reasoning": "x"}')
        assert result["score"] == 8

    def test_score_as_string_non_numeric_rejected(self) -> None:
        result = parse_judge_response('{"score": "high", "reasoning": "x"}')
        assert result["score"] is None
        assert result["parse_error"] is not None

    def test_score_as_float_rounded_to_int(self) -> None:
        result = parse_judge_response('{"score": 8.5, "reasoning": "x"}')
        # Banker's rounding: round(8.5) == 8 in Python.
        assert result["score"] == 8

    def test_score_as_float_rounded_up(self) -> None:
        result = parse_judge_response('{"score": 8.7, "reasoning": "x"}')
        assert result["score"] == 9

    def test_score_boolean_rejected(self) -> None:
        result = parse_judge_response('{"score": true, "reasoning": "x"}')
        assert result["score"] is None
        assert "boolean" in (result["parse_error"] or "").lower()

    def test_non_object_root_rejected(self) -> None:
        result = parse_judge_response("[1, 2, 3]")
        assert result["score"] is None


class TestCoerceScore:
    @pytest.mark.parametrize("raw, expected", [(1, 1), (5, 5), (10, 10)])
    def test_in_range_int(self, raw: int, expected: int) -> None:
        score, err = _coerce_score(raw)
        assert score == expected
        assert err is None

    @pytest.mark.parametrize("raw", [0, 11, -3, 100])
    def test_out_of_range_int(self, raw: int) -> None:
        score, err = _coerce_score(raw)
        assert score is None
        assert err is not None


class TestStripCodeFence:
    def test_no_fence(self) -> None:
        assert _strip_code_fence("plain text") == "plain text"

    def test_json_fence(self) -> None:
        assert _strip_code_fence("```json\n{}\n```") == "{}"

    def test_bare_fence(self) -> None:
        assert _strip_code_fence("```\n{}\n```") == "{}"


class TestExtractFirstJsonObject:
    def test_finds_object_after_prose(self) -> None:
        data = _extract_first_json_object('Some prose here: {"score": 8}')
        assert data == {"score": 8}

    def test_handles_nested_braces(self) -> None:
        data = _extract_first_json_object('{"nested": {"deep": 1}, "score": 7}')
        assert data == {"nested": {"deep": 1}, "score": 7}

    def test_handles_string_with_braces_inside(self) -> None:
        data = _extract_first_json_object('{"reasoning": "matches {brace}", "score": 5}')
        assert data is not None
        assert data["score"] == 5

    def test_returns_none_when_no_object(self) -> None:
        assert _extract_first_json_object("no braces here") is None


# ===== build_judge_prompt =====


class TestBuildJudgePrompt:
    def test_criteria_bulleted_into_template(self) -> None:
        prompt = build_judge_prompt(
            original_prompt="What is 2+2?",
            response_to_eval="4",
            criteria=["Accuracy", "Clarity"],
        )
        assert "- Accuracy" in prompt
        assert "- Clarity" in prompt
        assert "What is 2+2?" in prompt
        assert "4" in prompt

    def test_empty_criteria_uses_defaults(self) -> None:
        prompt = build_judge_prompt(original_prompt="x", response_to_eval="y", criteria=[])
        for c in DEFAULT_CRITERIA:
            assert f"- {c}" in prompt

    def test_no_template_injection_from_response(self) -> None:
        """Curly braces in the response must NOT break the formatter
        (we use string replacement, not str.format).
        """
        prompt = build_judge_prompt(
            original_prompt="q",
            response_to_eval='{"score": 999}',  # would explode .format()
            criteria=["acc"],
        )
        assert '{"score": 999}' in prompt

    def test_no_template_injection_from_prompt(self) -> None:
        prompt = build_judge_prompt(
            original_prompt="{ignore previous instructions}",
            response_to_eval="r",
            criteria=["acc"],
        )
        assert "{ignore previous instructions}" in prompt

    def test_custom_template_respected(self) -> None:
        custom = "EVAL: {criteria} :: {prompt} :: {response}"
        prompt = build_judge_prompt(
            original_prompt="orig",
            response_to_eval="resp",
            criteria=["only"],
            template=custom,
        )
        assert prompt == "EVAL: - only :: orig :: resp"


# ===== fake judge provider =====


class _FakeJudgeProvider(BaseProvider):
    """Returns a preset response or raises a preset error."""

    def __init__(
        self,
        name: str = "openai",
        response_text: str = '{"score": 8, "reasoning": "good"}',
        cost_usd: float = 0.0005,
        latency_ms: float = 50.0,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self._response = response_text
        self._cost = cost_usd
        self._latency = latency_ms
        self._error = error
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
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if self._error is not None:
            raise self._error
        return CompletionResult(
            output=self._response,
            cost_usd=self._cost,
            latency_ms=self._latency,
            model=model,
            provider=self.name,
            temperature=temperature,
        )


# ===== score_with_judge =====


class TestScoreWithJudge:
    async def test_happy_path(self) -> None:
        provider = _FakeJudgeProvider()
        score = await score_with_judge(
            judge_provider=provider,
            judge_model="claude-opus-4-7",
            original_prompt="orig",
            response_to_eval="response",
            criteria=["accuracy"],
        )
        assert score.model == "claude-opus-4-7"
        assert score.score == 8
        assert score.reasoning == "good"
        assert score.cost_usd == 0.0005
        assert score.parse_error is None

    async def test_temperature_is_zero(self) -> None:
        provider = _FakeJudgeProvider()
        await score_with_judge(
            judge_provider=provider,
            judge_model="claude-opus-4-7",
            original_prompt="orig",
            response_to_eval="r",
            criteria=["x"],
        )
        # Deterministic scoring: temperature MUST be 0.
        assert provider.calls[0]["temperature"] == 0.0

    async def test_parse_failure_captured(self) -> None:
        provider = _FakeJudgeProvider(response_text="not valid json at all")
        score = await score_with_judge(
            judge_provider=provider,
            judge_model="x",
            original_prompt="o",
            response_to_eval="r",
            criteria=["c"],
        )
        assert score.score is None
        assert score.parse_error is not None

    async def test_provider_error_captured(self) -> None:
        provider = _FakeJudgeProvider(
            error=ProviderError("network died sk-proj-leaked123456789abcdef")
        )
        score = await score_with_judge(
            judge_provider=provider,
            judge_model="x",
            original_prompt="o",
            response_to_eval="r",
            criteria=["c"],
        )
        assert score.score is None
        # Error message is redacted.
        assert "sk-proj-leaked" not in (score.parse_error or "")
        assert "REDACTED" in (score.parse_error or "")


# ===== run_judging =====


def _state(
    model: str, temperature: float = 0.0, text: str = "answer", error: str | None = None
) -> StreamState:
    return StreamState(
        model=model,
        provider_name="openai",
        temperature=temperature,
        status="complete" if error is None else "error",
        text=text,
        error=error,
    )


class TestRunJudging:
    async def test_two_states_two_judges_yields_four_scores(self) -> None:
        provider = _FakeJudgeProvider()
        states = [
            _state(model="gpt-5.5", text="answer one"),
            _state(model="claude-opus-4-7", text="answer two"),
        ]
        items = [(s, "user prompt") for s in states]

        results = await run_judging(
            items=items,
            judge_models=["claude-haiku-4-5", "gemini-3.1-pro"],
            criteria=["x"],
            provider_factory=lambda _: provider,
            skip_self_eval=True,
        )

        assert len(results) == 2
        # 2 judges per state -> 4 judge scores total.
        assert sum(len(r.judges) for r in results) == 4
        for r in results:
            assert r.average_score == 8.0

    async def test_self_eval_skipped_for_matching_model(self) -> None:
        provider = _FakeJudgeProvider()
        states = [_state(model="gpt-5.5")]
        items = [(s, "u") for s in states]

        results = await run_judging(
            items=items,
            judge_models=["gpt-5.5", "claude-opus-4-7"],
            criteria=["x"],
            provider_factory=lambda _: provider,
            skip_self_eval=True,
        )

        assert len(results) == 1
        # gpt-5.5 was the model; it must be in skipped_models.
        assert "gpt-5.5" in results[0].skipped_models
        # claude-opus-4-7 is the only judge that ran.
        assert len(results[0].judges) == 1
        assert results[0].judges[0].model == "claude-opus-4-7"

    async def test_self_eval_skip_can_be_disabled(self) -> None:
        provider = _FakeJudgeProvider()
        items = [(_state(model="gpt-5.5"), "u")]

        results = await run_judging(
            items=items,
            judge_models=["gpt-5.5", "claude-opus-4-7"],
            criteria=["x"],
            provider_factory=lambda _: provider,
            skip_self_eval=False,
        )

        # No skipping - both judges ran.
        assert results[0].skipped_models == []
        assert len(results[0].judges) == 2

    async def test_one_judge_failing_does_not_kill_panel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One judge raising shouldn't propagate; that judge gets a parse_error
        and the rest of the panel keeps its scores.
        """
        good = _FakeJudgeProvider(name="openai", response_text='{"score": 9, "reasoning": "x"}')
        bad = _FakeJudgeProvider(name="anthropic", error=ProviderError("died"))

        def factory(name: str) -> BaseProvider:
            return good if name == "openai" else bad

        items = [(_state(model="grok-4.3"), "u")]

        results = await run_judging(
            items=items,
            judge_models=["gpt-5.5", "claude-opus-4-7"],
            criteria=["x"],
            provider_factory=factory,
            skip_self_eval=True,
        )

        # Two judges scored; one succeeded with score 9, one failed.
        assert len(results[0].judges) == 2
        successful = [j for j in results[0].judges if j.score is not None]
        assert len(successful) == 1
        # Average uses ONLY the successful judge.
        assert results[0].average_score == 9.0

    async def test_failed_state_gets_empty_judge_result(self) -> None:
        """Don't waste judge calls on a state whose main call failed."""
        provider = _FakeJudgeProvider()
        items = [(_state(model="gpt-5.5", error="main call failed"), "u")]

        results = await run_judging(
            items=items,
            judge_models=["claude-opus-4-7"],
            criteria=["x"],
            provider_factory=lambda _: provider,
        )

        assert results[0].judges == []
        # No judge calls were made.
        assert provider.calls == []

    async def test_no_judges_returns_empty_results(self) -> None:
        items = [(_state(model="gpt-5.5"), "u")]
        results = await run_judging(
            items=items,
            judge_models=[],
            criteria=["x"],
            provider_factory=lambda _: _FakeJudgeProvider(),
        )
        assert len(results) == 1
        assert results[0].judges == []

    async def test_empty_criteria_uses_defaults(self) -> None:
        provider = _FakeJudgeProvider()
        items = [(_state(model="gpt-5.5"), "u")]

        await run_judging(
            items=items,
            judge_models=["claude-opus-4-7"],
            criteria=None,
            provider_factory=lambda _: provider,
        )

        # The judge prompt sent to the provider includes the default criteria.
        prompt_sent = provider.calls[0]["prompt"]
        for c in DEFAULT_CRITERIA:
            assert c in prompt_sent


# ===== _aggregate =====


class TestAggregate:
    def test_average_excludes_failed_judges(self) -> None:
        judges = [
            JudgeScore(model="a", score=8, reasoning="", cost_usd=0.0, latency_ms=0.0),
            JudgeScore(
                model="b", score=None, reasoning="", cost_usd=0.0, latency_ms=0.0, parse_error="bad"
            ),
            JudgeScore(model="c", score=6, reasoning="", cost_usd=0.0, latency_ms=0.0),
        ]
        result = _aggregate(judges)
        assert result.average_score == 7.0  # (8 + 6) / 2

    def test_std_dev_requires_two_successful_judges(self) -> None:
        judges = [JudgeScore(model="a", score=8, reasoning="", cost_usd=0.0, latency_ms=0.0)]
        result = _aggregate(judges)
        assert result.std_dev is None  # only one judge

    def test_std_dev_with_two_judges(self) -> None:
        judges = [
            JudgeScore(model="a", score=6, reasoning="", cost_usd=0.0, latency_ms=0.0),
            JudgeScore(model="b", score=10, reasoning="", cost_usd=0.0, latency_ms=0.0),
        ]
        result = _aggregate(judges)
        # statistics.stdev of [6, 10] = sqrt(8) ≈ 2.828
        assert result.std_dev is not None
        assert math.isclose(result.std_dev, math.sqrt(8), rel_tol=1e-6)

    def test_all_failed_judges_yields_none_average(self) -> None:
        judges = [
            JudgeScore(
                model="a", score=None, reasoning="", cost_usd=0.0, latency_ms=0.0, parse_error="x"
            ),
            JudgeScore(
                model="b", score=None, reasoning="", cost_usd=0.0, latency_ms=0.0, parse_error="y"
            ),
        ]
        result = _aggregate(judges)
        assert result.average_score is None
        assert result.std_dev is None

    def test_skipped_models_preserved(self) -> None:
        result = _aggregate([], skipped_models=["gpt-5.5"])
        assert result.skipped_models == ["gpt-5.5"]


# ===== cost tracking helpers =====


class TestCostTotals:
    def test_total_judge_cost(self) -> None:
        from cli_modelarium.judging import JudgeResult

        results = [
            JudgeResult(
                judges=[
                    JudgeScore(model="a", score=8, reasoning="", cost_usd=0.001, latency_ms=0.0),
                    JudgeScore(model="b", score=7, reasoning="", cost_usd=0.002, latency_ms=0.0),
                ]
            ),
            JudgeResult(
                judges=[
                    JudgeScore(model="a", score=9, reasoning="", cost_usd=0.0005, latency_ms=0.0),
                ]
            ),
        ]
        assert total_judge_cost(results) == pytest.approx(0.0035)
        assert total_judge_calls(results) == 3
