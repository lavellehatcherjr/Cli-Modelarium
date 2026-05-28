"""Tests for cli_modelarium.batch - parsers, validation, size limits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli_modelarium.batch import (
    MAX_PROMPTS_PER_BATCH,
    MAX_TOTAL_CALLS,
    BatchPrompt,
    _count_total_calls,
    build_batch_states,
    check_batch_size_limits,
    detect_output_format,
    estimate_batch_cost,
    load_batch_file,
    output_overlaps_input,
)
from cli_modelarium.exceptions import (
    BatchSizeError,
    BatchValidationError,
)

# ===== .txt parser =====


class TestTxtParser:
    def test_one_prompt_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.txt"
        path.write_text("first prompt\nsecond prompt\nthird prompt\n", encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert len(prompts) == 3
        assert prompts[0].prompt == "first prompt"
        assert prompts[1].prompt == "second prompt"
        assert prompts[2].prompt == "third prompt"

    def test_auto_assigns_sequential_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("a\nb\nc\n", encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert [p.id for p in prompts] == ["p1", "p2", "p3"]

    def test_comment_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text(
            "# header comment\nreal prompt\n# another comment\nsecond\n",
            encoding="utf-8",
        )

        prompts = load_batch_file(str(path))

        assert len(prompts) == 2
        assert prompts[0].prompt == "real prompt"
        assert prompts[1].prompt == "second"

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("a\n\n\nb\n   \nc\n", encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert [p.prompt for p in prompts] == ["a", "b", "c"]

    def test_inline_hash_is_not_a_comment(self, tmp_path: Path) -> None:
        """Only leading `#` counts. `What's #1?` must be a real prompt."""
        path = tmp_path / "p.txt"
        path.write_text("What's #1?\n# comment\n", encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert len(prompts) == 1
        assert prompts[0].prompt == "What's #1?"

    def test_leading_trailing_whitespace_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("   padded   \n", encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert prompts[0].prompt == "padded"

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")

        assert load_batch_file(str(path)) == []

    def test_only_comments_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "comments.txt"
        path.write_text("# only\n# comments\n", encoding="utf-8")

        assert load_batch_file(str(path)) == []

    def test_bom_is_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.txt"
        path.write_bytes(b"\xef\xbb\xbffirst prompt\nsecond\n")

        prompts = load_batch_file(str(path))

        # The BOM must not pollute the first prompt.
        assert prompts[0].prompt == "first prompt"


# ===== .json parser =====


class TestJsonParser:
    def test_valid_array(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "math-1", "prompt": "what is 2+2?"},
                    {"id": "math-2", "prompt": "what is 3+3?", "system": "you are precise"},
                ]
            ),
            encoding="utf-8",
        )

        prompts = load_batch_file(str(path))

        assert len(prompts) == 2
        assert prompts[0].id == "math-1"
        assert prompts[0].prompt == "what is 2+2?"
        assert prompts[0].system is None
        assert prompts[1].id == "math-2"
        assert prompts[1].system == "you are precise"

    def test_missing_id_auto_generated(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps([{"prompt": "a"}, {"prompt": "b"}]), encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert [p.id for p in prompts] == ["p1", "p2"]

    def test_missing_prompt_field_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")

        with pytest.raises(BatchValidationError) as exc_info:
            load_batch_file(str(path))

        assert "prompt" in str(exc_info.value).lower()

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps([{"id": "dupe", "prompt": "a"}, {"id": "dupe", "prompt": "b"}]),
            encoding="utf-8",
        )

        with pytest.raises(BatchValidationError) as exc_info:
            load_batch_file(str(path))

        assert "duplicate" in str(exc_info.value).lower()
        assert "dupe" in str(exc_info.value)

    def test_non_array_root_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"not": "an array"}), encoding="utf-8")

        with pytest.raises(BatchValidationError) as exc_info:
            load_batch_file(str(path))

        assert "array" in str(exc_info.value).lower()

    def test_malformed_json_raises_json_decode_error(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text("not valid json {", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_batch_file(str(path))

    def test_assertions_preserved_as_raw_dicts(self, tmp_path: Path) -> None:
        """Phase 9 will execute these; for now we carry them through verbatim."""
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "a1",
                        "prompt": "list 3 colors",
                        "assertions": [
                            {"type": "json_valid"},
                            {"type": "max_length_chars", "value": 200},
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

        prompts = load_batch_file(str(path))

        assert len(prompts[0].assertions) == 2
        assert prompts[0].assertions[0] == {"type": "json_valid"}
        assert prompts[0].assertions[1] == {"type": "max_length_chars", "value": 200}

    def test_non_dict_element_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps(["bare string", {"prompt": "ok"}]), encoding="utf-8")

        with pytest.raises(BatchValidationError):
            load_batch_file(str(path))

    def test_non_string_prompt_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(json.dumps([{"prompt": 42}]), encoding="utf-8")

        with pytest.raises(BatchValidationError):
            load_batch_file(str(path))

    def test_non_list_assertions_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps([{"prompt": "ok", "assertions": "not a list"}]),
            encoding="utf-8",
        )

        with pytest.raises(BatchValidationError):
            load_batch_file(str(path))


# ===== format detection by extension =====


class TestFormatDetection:
    def test_txt_uses_txt_parser(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("hello\n", encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert len(prompts) == 1
        assert prompts[0].prompt == "hello"

    def test_json_uses_json_parser(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        path.write_text(json.dumps([{"prompt": "hi"}]), encoding="utf-8")

        prompts = load_batch_file(str(path))

        assert len(prompts) == 1
        assert prompts[0].prompt == "hi"

    def test_no_extension_rejected_with_hint(self, tmp_path: Path) -> None:
        path = tmp_path / "no_ext"
        path.write_text("hello\n", encoding="utf-8")

        with pytest.raises(BatchValidationError) as exc_info:
            load_batch_file(str(path))

        message = str(exc_info.value).lower()
        assert ".txt" in message and ".json" in message

    def test_unknown_extension_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("hello\n", encoding="utf-8")

        with pytest.raises(BatchValidationError):
            load_batch_file(str(path))


# ===== safe_input_path enforcement =====


class TestSizeLimitOnInputFile:
    def test_file_too_large_rejected_before_parsing(self, tmp_path: Path) -> None:
        from cli_modelarium.io_safety import BATCH_INPUT_MAX_BYTES

        path = tmp_path / "huge.txt"
        path.write_bytes(b"x\n" * (BATCH_INPUT_MAX_BYTES // 2 + 1))

        with pytest.raises(ValueError) as exc_info:
            load_batch_file(str(path))

        assert "too large" in str(exc_info.value).lower()


# ===== check_batch_size_limits =====


class TestSizeLimits:
    def _prompts(self, n: int) -> list[BatchPrompt]:
        return [BatchPrompt(id=f"p{i}", prompt=f"prompt {i}") for i in range(n)]

    def test_under_limit_passes(self) -> None:
        total = check_batch_size_limits(
            prompts=self._prompts(10),
            models=["gpt-5.5"],
            temperatures=[0.0],
            command_system_prompts=[None],
        )
        assert total == 10

    def test_over_prompt_limit_without_force_raises(self) -> None:
        with pytest.raises(BatchSizeError) as exc_info:
            check_batch_size_limits(
                prompts=self._prompts(MAX_PROMPTS_PER_BATCH + 1),
                models=["gpt-5.5"],
                temperatures=[0.0],
                command_system_prompts=[None],
            )
        message = str(exc_info.value)
        assert "--force-large" in message
        assert str(MAX_PROMPTS_PER_BATCH) in message

    def test_force_large_bypasses_prompt_limit(self) -> None:
        # Build something well over the prompt cap but small enough for the
        # total-call cap.
        total = check_batch_size_limits(
            prompts=self._prompts(MAX_PROMPTS_PER_BATCH + 5),
            models=["gpt-5.5"],
            temperatures=[0.0],
            command_system_prompts=[None],
            force_large=True,
        )
        assert total == MAX_PROMPTS_PER_BATCH + 5

    def test_over_total_call_limit_without_force_raises(self) -> None:
        # 100 prompts x 100 models x 1 temp = 10_100 > MAX_TOTAL_CALLS
        models = [f"model-{i}" for i in range(100)]
        # _count_total_calls doesn't validate model existence, so we can
        # use any string.
        with pytest.raises(BatchSizeError) as exc_info:
            check_batch_size_limits(
                prompts=self._prompts(102),
                models=models,
                temperatures=[0.0],
                command_system_prompts=[None],
            )
        assert "--force-large" in str(exc_info.value)
        assert str(MAX_TOTAL_CALLS) in str(exc_info.value)

    def test_per_prompt_system_overrides_count_one_each(self) -> None:
        prompts = [
            BatchPrompt(id="a", prompt="a", system="custom-a"),  # 1 SP
            BatchPrompt(id="b", prompt="b"),  # uses command sps
        ]
        # command_sps has 3 prompts. Prompt a has its own, so it's 1*1*1=1
        # call. Prompt b uses all 3, so 3*1*1=3. Total = 4.
        total = check_batch_size_limits(
            prompts=prompts,
            models=["m1"],
            temperatures=[0.0],
            command_system_prompts=["s1", "s2", "s3"],
        )
        assert total == 4


# ===== _count_total_calls =====


class TestCountTotalCalls:
    def test_simple_matrix(self) -> None:
        n = _count_total_calls(
            prompts=[BatchPrompt(id="a", prompt="a")],
            models=["m1", "m2"],
            temperatures=[0.0, 0.5, 1.0],
            command_system_prompts=[None],
        )
        assert n == 6  # 1 prompt x 2 models x 3 temps

    def test_no_temperatures_treated_as_one(self) -> None:
        n = _count_total_calls(
            prompts=[BatchPrompt(id="a", prompt="a")],
            models=["m1"],
            temperatures=[],
            command_system_prompts=[None],
        )
        assert n == 1

    def test_no_models_treated_as_one(self) -> None:
        # Defensive against an empty list slipping through validation.
        n = _count_total_calls(
            prompts=[BatchPrompt(id="a", prompt="a")],
            models=[],
            temperatures=[0.0],
            command_system_prompts=[None],
        )
        assert n == 1


# ===== estimate_batch_cost =====


class TestEstimateCost:
    def test_local_models_contribute_zero(self) -> None:
        est = estimate_batch_cost(
            prompts=[BatchPrompt(id="p1", prompt="hi")],
            models=["local/llama-3.3"],
            temperatures=[0.0],
            command_system_prompts=[None],
        )
        assert est == 0.0

    def test_real_model_cost_nonzero(self) -> None:
        est = estimate_batch_cost(
            prompts=[BatchPrompt(id="p1", prompt="hi")],
            models=["gpt-5.5"],
            temperatures=[0.0],
            command_system_prompts=[None],
        )
        # 500 in @ $5/M + 500 out @ $30/M = 0.0025 + 0.015 = 0.0175
        assert est == pytest.approx(0.0175)

    def test_unknown_model_skipped_silently(self) -> None:
        est = estimate_batch_cost(
            prompts=[BatchPrompt(id="p1", prompt="hi")],
            models=["totally-fake-model"],
            temperatures=[0.0],
            command_system_prompts=[None],
        )
        assert est == 0.0


# ===== build_batch_states =====


class TestBuildBatchStates:
    def test_one_state_per_combination(self) -> None:
        prompts = [
            BatchPrompt(id="p1", prompt="a"),
            BatchPrompt(id="p2", prompt="b"),
        ]
        pairs = build_batch_states(
            prompts,
            models=["gpt-5.5"],
            temperatures=[0.0, 1.0],
            command_system_prompts=[None],
        )
        # 2 prompts x 1 model x 2 temps = 4 states
        assert len(pairs) == 4

    def test_per_prompt_system_used_for_that_prompt_only(self) -> None:
        prompts = [
            BatchPrompt(id="a", prompt="a", system="per-prompt-sp"),
            BatchPrompt(id="b", prompt="b"),
        ]
        pairs = build_batch_states(
            prompts,
            models=["gpt-5.5"],
            temperatures=[0.0],
            command_system_prompts=["cmd-sp-1", "cmd-sp-2"],
        )
        # Prompt a: 1 SP (own) -> 1 state. Prompt b: 2 SPs (cmd) -> 2 states.
        assert len(pairs) == 3
        a_states = [s for s, bp in pairs if bp.id == "a"]
        b_states = [s for s, bp in pairs if bp.id == "b"]
        assert len(a_states) == 1
        assert a_states[0].system_prompt == "per-prompt-sp"
        assert len(b_states) == 2
        assert {s.system_prompt for s in b_states} == {"cmd-sp-1", "cmd-sp-2"}


# ===== format detection helpers =====


class TestDetectOutputFormat:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("out.csv", "csv"),
            ("out.json", "json"),
            ("out.md", "markdown"),
            ("out.markdown", "markdown"),
            ("OUT.CSV", "csv"),  # case-insensitive
            ("noext", None),
            ("out.txt", None),  # .txt is a batch INPUT, not output
            ("out.yaml", None),
        ],
    )
    def test_extension_routing(self, filename: str, expected: str | None) -> None:
        assert detect_output_format(Path(filename)) == expected


class TestOutputOverlapsInput:
    def test_same_file_detected(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("hi", encoding="utf-8")
        assert output_overlaps_input(p, p)

    def test_different_files_not_overlapping(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.csv"
        a.write_text("a", encoding="utf-8")
        b.write_text("b", encoding="utf-8")
        assert not output_overlaps_input(a, b)

    def test_relative_vs_absolute_same_path_detected(self, tmp_path: Path) -> None:
        absolute = tmp_path / "x.txt"
        absolute.write_text("hi", encoding="utf-8")
        relative = Path(absolute.name)
        # Resolve both against tmp_path to simulate user passing relative.
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert output_overlaps_input(absolute, relative)
        finally:
            os.chdir(old_cwd)
