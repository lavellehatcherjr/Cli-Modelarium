"""Tests for cli_modelarium.hallucination.

Covers:
    * parse_facts_csv and load_expected_facts (input parsing + file loading)
    * build_hallucination_criteria (with/without facts)
    * parse_hallucination_response (risk_level extraction + score derivation)
    * aggregate_risk_levels and annotate_risk_levels (worst-wins aggregation)
    * resolve_hallucination_config (flag combination validation)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli_modelarium.exceptions import BatchValidationError, ModelariumError
from cli_modelarium.hallucination import (
    EXPECTED_FACTS_MAX_BYTES,
    HALLUCINATION_CRITERIA_BASE,
    HALLUCINATION_TEMPLATE,
    HALLUCINATION_TOS_EXTENSION,
    HALLUCINATION_WITH_FACTS,
    HALLUCINATION_WITHOUT_FACTS,
    HallucinationConfig,
    aggregate_risk_levels,
    annotate_risk_levels,
    build_hallucination_criteria,
    load_expected_facts,
    parse_facts_csv,
    parse_hallucination_response,
    resolve_hallucination_config,
    risk_level_from_score,
)
from cli_modelarium.judging import JudgeResult, JudgeScore


# ===== parse_facts_csv =====


class TestParseFactsCsv:
    def test_simple_list(self) -> None:
        assert parse_facts_csv("fact1,fact2,fact3") == ["fact1", "fact2", "fact3"]

    def test_whitespace_stripped(self) -> None:
        assert parse_facts_csv("  fact1 ,  fact2  ") == ["fact1", "fact2"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_facts_csv("") == []

    def test_comma_escape(self) -> None:
        # Source: r"a,b,c\,d" -> "a,b,c\,d" -> ["a", "b", "c,d"]
        assert parse_facts_csv(r"a,b,c\,d") == ["a", "b", "c,d"]

    def test_drops_empty_pieces(self) -> None:
        assert parse_facts_csv("a,,b,") == ["a", "b"]


# ===== load_expected_facts =====


class TestLoadExpectedFacts:
    def test_txt_one_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("fact one\nfact two\nfact three\n", encoding="utf-8")

        facts = load_expected_facts(str(path))

        assert facts == ["fact one", "fact two", "fact three"]

    def test_txt_comments_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("# header\nfact one\n# mid comment\nfact two\n", encoding="utf-8")

        facts = load_expected_facts(str(path))

        assert facts == ["fact one", "fact two"]

    def test_txt_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("\n\nfact one\n   \nfact two\n", encoding="utf-8")

        facts = load_expected_facts(str(path))

        assert facts == ["fact one", "fact two"]

    def test_json_array_of_strings(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        path.write_text(json.dumps(["fact one", "fact two"]), encoding="utf-8")

        facts = load_expected_facts(str(path))

        assert facts == ["fact one", "fact two"]

    def test_json_non_array_root_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"not": "an array"}), encoding="utf-8")

        with pytest.raises(BatchValidationError):
            load_expected_facts(str(path))

    def test_json_non_string_element_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        path.write_text(json.dumps(["ok", 42]), encoding="utf-8")

        with pytest.raises(BatchValidationError):
            load_expected_facts(str(path))

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("", encoding="utf-8")

        assert load_expected_facts(str(path)) == []

    def test_unknown_extension_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.yaml"
        path.write_text("fact", encoding="utf-8")

        with pytest.raises(BatchValidationError):
            load_expected_facts(str(path))

    def test_duplicate_facts_case_insensitive_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("Built in 1887\nbuilt in 1887\n", encoding="utf-8")

        with pytest.raises(BatchValidationError) as exc_info:
            load_expected_facts(str(path))

        # Error mentions "duplicate" so users can find the offending entry.
        assert "duplicate" in str(exc_info.value).lower()

    def test_file_too_large_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_bytes(b"x\n" * (EXPECTED_FACTS_MAX_BYTES // 2 + 1))

        with pytest.raises(ValueError):
            load_expected_facts(str(path))


# ===== build_hallucination_criteria =====


class TestBuildHallucinationCriteria:
    def test_with_facts_produces_bullets(self) -> None:
        criteria = build_hallucination_criteria(["Built 1887-1889", "Located in Paris"])

        assert len(criteria) == 1
        # The fact bullets are embedded in the criterion text.
        assert "  - Built 1887-1889" in criteria[0]
        assert "  - Located in Paris" in criteria[0]
        # And the WITH_FACTS structure (verify a key phrase from the constant).
        assert "consistent with these known facts" in criteria[0]

    def test_without_facts_uses_without_template(self) -> None:
        criteria = build_hallucination_criteria(None)

        assert len(criteria) == 1
        # WITHOUT_FACTS template is substituted.
        assert "factual knowledge" in criteria[0]
        # No "known facts" phrasing.
        assert "consistent with these known facts" not in criteria[0]

    def test_empty_list_treated_as_no_facts(self) -> None:
        # Per spec: empty facts list -> WITHOUT_FACTS template.
        # build_hallucination_criteria itself uses `if facts:` so empty -> without.
        criteria = build_hallucination_criteria([])
        assert "factual knowledge" in criteria[0]

    def test_criterion_contains_response_format_instruction(self) -> None:
        criteria = build_hallucination_criteria(None)
        assert "Respond ONLY with JSON" in criteria[0]
        # Includes risk_level in the format.
        assert "risk_level" in criteria[0]

    def test_no_leftover_placeholders(self) -> None:
        """After substitution, no {placeholder} should remain to confuse the judge."""
        criteria = build_hallucination_criteria(["fact"])
        assert "{reference_check}" not in criteria[0]
        assert "{facts}" not in criteria[0]


# ===== parse_hallucination_response =====


class TestParseHallucinationResponse:
    def test_valid_json_with_all_three_fields(self) -> None:
        result = parse_hallucination_response(
            '{"score": 8, "risk_level": "Low", "reasoning": "looks accurate"}'
        )
        assert result["score"] == 8
        assert result["risk_level"] == "Low"
        assert result["reasoning"] == "looks accurate"
        assert result["parse_error"] is None

    def test_risk_level_case_normalized(self) -> None:
        result = parse_hallucination_response(
            '{"score": 4, "risk_level": "MEDIUM", "reasoning": "x"}'
        )
        assert result["risk_level"] == "Medium"

    def test_risk_level_lowercase_accepted(self) -> None:
        result = parse_hallucination_response(
            '{"score": 2, "risk_level": "high", "reasoning": "x"}'
        )
        assert result["risk_level"] == "High"

    def test_risk_level_missing_derived_from_score(self) -> None:
        # 8 -> Low (7-10)
        result = parse_hallucination_response('{"score": 8, "reasoning": "x"}')
        assert result["risk_level"] == "Low"
        assert result["parse_error"] is None

    def test_risk_level_derived_for_each_band(self) -> None:
        cases = [(10, "Low"), (7, "Low"), (6, "Medium"), (4, "Medium"),
                 (3, "High"), (1, "High")]
        for score, expected in cases:
            result = parse_hallucination_response(
                f'{{"score": {score}, "reasoning": "x"}}'
            )
            assert result["risk_level"] == expected, f"score={score}"

    def test_invalid_risk_level_value_sets_parse_error(self) -> None:
        result = parse_hallucination_response(
            '{"score": 5, "risk_level": "Critical", "reasoning": "x"}'
        )
        assert result["parse_error"] is not None
        assert "Critical" in result["parse_error"]
        # Risk level is None when invalid.
        assert result["risk_level"] is None

    def test_risk_level_wrong_type_sets_parse_error(self) -> None:
        result = parse_hallucination_response(
            '{"score": 5, "risk_level": 99, "reasoning": "x"}'
        )
        assert result["parse_error"] is not None
        assert result["risk_level"] is None

    def test_score_out_of_range_sets_parse_error(self) -> None:
        result = parse_hallucination_response(
            '{"score": 11, "risk_level": "Low", "reasoning": "x"}'
        )
        # parse_judge_response rejects out-of-range scores.
        assert result["score"] is None
        assert result["parse_error"] is not None

    def test_markdown_fenced_json(self) -> None:
        text = '```json\n{"score": 9, "risk_level": "Low", "reasoning": "x"}\n```'
        result = parse_hallucination_response(text)
        assert result["score"] == 9
        assert result["risk_level"] == "Low"

    def test_leading_text_before_json(self) -> None:
        text = 'Analysis follows: {"score": 6, "risk_level": "Medium", "reasoning": "x"}'
        result = parse_hallucination_response(text)
        assert result["score"] == 6
        assert result["risk_level"] == "Medium"

    def test_empty_response(self) -> None:
        result = parse_hallucination_response("")
        assert result["score"] is None
        assert result["risk_level"] is None


# ===== risk_level_from_score =====


class TestRiskLevelFromScore:
    @pytest.mark.parametrize(
        "score, expected",
        [
            (10, "Low"), (8, "Low"), (7, "Low"),
            (6, "Medium"), (5, "Medium"), (4, "Medium"),
            (3, "High"), (2, "High"), (1, "High"),
        ],
    )
    def test_score_to_level(self, score: int, expected: str) -> None:
        assert risk_level_from_score(score) == expected

    def test_none_score_returns_none(self) -> None:
        assert risk_level_from_score(None) is None

    def test_out_of_range_returns_none(self) -> None:
        assert risk_level_from_score(0) is None
        assert risk_level_from_score(11) is None
        assert risk_level_from_score(-1) is None


# ===== aggregate_risk_levels (worst-wins) =====


class TestAggregateRiskLevels:
    def _judge(self, risk_level: str | None) -> JudgeScore:
        return JudgeScore(
            model="m", score=5, reasoning="", cost_usd=0.0, latency_ms=0.0,
            risk_level=risk_level,
        )

    def test_any_high_wins(self) -> None:
        judges = [self._judge("Low"), self._judge("High"), self._judge("Medium")]
        assert aggregate_risk_levels(judges) == "High"

    def test_no_high_medium_wins(self) -> None:
        judges = [self._judge("Low"), self._judge("Medium")]
        assert aggregate_risk_levels(judges) == "Medium"

    def test_all_low(self) -> None:
        judges = [self._judge("Low"), self._judge("Low")]
        assert aggregate_risk_levels(judges) == "Low"

    def test_empty_returns_none(self) -> None:
        assert aggregate_risk_levels([]) is None

    def test_all_none_returns_none(self) -> None:
        judges = [self._judge(None), self._judge(None)]
        assert aggregate_risk_levels(judges) is None

    def test_mixed_some_none_ignores_them(self) -> None:
        judges = [self._judge(None), self._judge("Medium"), self._judge(None)]
        assert aggregate_risk_levels(judges) == "Medium"


class TestAnnotateRiskLevels:
    def test_mutates_in_place(self) -> None:
        jr = JudgeResult(judges=[
            JudgeScore(model="a", score=2, reasoning="", cost_usd=0.0, latency_ms=0.0, risk_level="High"),
            JudgeScore(model="b", score=8, reasoning="", cost_usd=0.0, latency_ms=0.0, risk_level="Low"),
        ])
        results = [jr]
        annotate_risk_levels(results)
        assert results[0].aggregated_risk_level == "High"


# ===== resolve_hallucination_config =====


class TestResolveHallucinationConfig:
    def test_not_checking_returns_none(self) -> None:
        cfg = resolve_hallucination_config(
            check_hallucination=False,
            expected_facts=None,
            expected_facts_file=None,
            hallucination_template=None,
            judge_models_present=False,
        )
        assert cfg is None

    def test_facts_without_check_rejected(self) -> None:
        with pytest.raises(BatchValidationError):
            resolve_hallucination_config(
                check_hallucination=False,
                expected_facts="a,b,c",
                expected_facts_file=None,
                hallucination_template=None,
                judge_models_present=False,
            )

    def test_template_without_check_rejected(self) -> None:
        with pytest.raises(BatchValidationError):
            resolve_hallucination_config(
                check_hallucination=False,
                expected_facts=None,
                expected_facts_file=None,
                hallucination_template="/tmp/template.txt",
                judge_models_present=False,
            )

    def test_check_without_judges_rejected(self) -> None:
        with pytest.raises(BatchValidationError) as exc_info:
            resolve_hallucination_config(
                check_hallucination=True,
                expected_facts=None,
                expected_facts_file=None,
                hallucination_template=None,
                judge_models_present=False,
            )
        assert "--judge" in str(exc_info.value)

    def test_check_with_judges_no_facts_uses_without_template(self) -> None:
        cfg = resolve_hallucination_config(
            check_hallucination=True,
            expected_facts=None,
            expected_facts_file=None,
            hallucination_template=None,
            judge_models_present=True,
        )
        assert cfg is not None
        assert cfg.facts is None
        assert "factual knowledge" in cfg.criteria[0]
        assert cfg.template == HALLUCINATION_TEMPLATE

    def test_check_with_inline_facts(self) -> None:
        cfg = resolve_hallucination_config(
            check_hallucination=True,
            expected_facts="fact one,fact two",
            expected_facts_file=None,
            hallucination_template=None,
            judge_models_present=True,
        )
        assert cfg is not None
        assert cfg.facts == ["fact one", "fact two"]
        assert "  - fact one" in cfg.criteria[0]

    def test_facts_and_facts_file_together_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "f.txt"
        path.write_text("x", encoding="utf-8")

        with pytest.raises(BatchValidationError):
            resolve_hallucination_config(
                check_hallucination=True,
                expected_facts="a",
                expected_facts_file=str(path),
                hallucination_template=None,
                judge_models_present=True,
            )

    def test_template_and_facts_together_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(BatchValidationError):
            resolve_hallucination_config(
                check_hallucination=True,
                expected_facts="a,b",
                expected_facts_file=None,
                hallucination_template="/tmp/template.txt",
                judge_models_present=True,
            )

    def test_empty_facts_csv_normalized_to_none(self) -> None:
        cfg = resolve_hallucination_config(
            check_hallucination=True,
            expected_facts="   ",  # all whitespace -> empty list -> None
            expected_facts_file=None,
            hallucination_template=None,
            judge_models_present=True,
        )
        assert cfg is not None
        assert cfg.facts is None
        # Uses WITHOUT_FACTS template path.
        assert "factual knowledge" in cfg.criteria[0]

    def test_custom_template_loaded(self, tmp_path: Path) -> None:
        template_path = tmp_path / "tpl.txt"
        template_path.write_text("MY CUSTOM RUBRIC", encoding="utf-8")

        cfg = resolve_hallucination_config(
            check_hallucination=True,
            expected_facts=None,
            expected_facts_file=None,
            hallucination_template=str(template_path),
            judge_models_present=True,
        )
        assert cfg is not None
        assert cfg.criteria == ["MY CUSTOM RUBRIC"]
        assert cfg.facts is None


# ===== ToS extension constant exists =====


class TestToSExtension:
    def test_extension_text_present(self) -> None:
        # The wording exists and mentions "guidance, not ground truth".
        assert "guidance" in HALLUCINATION_TOS_EXTENSION.lower()
        assert "ground truth" in HALLUCINATION_TOS_EXTENSION.lower()
