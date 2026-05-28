"""End-to-end tests: example files must work with the real loaders.

These prove the example files are real working configurations - not just
placeholders that look right. If we change the parser/loader contract,
these tests fail and we update the examples.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli_modelarium.assertions import parse_assertion_config
from cli_modelarium.batch import load_batch_file
from cli_modelarium.hallucination import load_expected_facts

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize(
    "filename",
    ["batch_evaluation.json", "hallucination_test.json", "ci_eval_suite.json"],
)
def test_batch_file_loads(filename: str) -> None:
    """Every example batch file must load through the real `load_batch_file`."""
    prompts = load_batch_file(str(EXAMPLES_DIR / filename))
    assert len(prompts) > 0
    # Every prompt has the required fields.
    for bp in prompts:
        assert bp.id
        assert bp.prompt


def test_batch_evaluation_assertions_parse() -> None:
    """Every assertion in batch_evaluation.json must parse via the real validator.

    This ensures the example demonstrates valid assertion configs - not
    placeholders that would fail at runtime when a user copies them.
    """
    prompts = load_batch_file(str(EXAMPLES_DIR / "batch_evaluation.json"))
    for bp in prompts:
        for raw in bp.assertions:
            # Must not raise - parse_assertion_config validates the config.
            cfg = parse_assertion_config(raw)
            assert cfg.type


def test_ci_eval_suite_assertions_parse() -> None:
    """Same for the CI eval suite - all assertions must be real valid configs."""
    prompts = load_batch_file(str(EXAMPLES_DIR / "ci_eval_suite.json"))
    asserted_types: set[str] = set()
    for bp in prompts:
        for raw in bp.assertions:
            cfg = parse_assertion_config(raw)
            asserted_types.add(cfg.type)
    # The example should exercise a variety of assertion types to be useful
    # documentation - not just one type.
    assert len(asserted_types) >= 4


def test_expected_facts_example_with_comments_stripped() -> None:
    """The example facts file uses `#` comments; verify they're stripped."""
    facts = load_expected_facts(str(EXAMPLES_DIR / "expected_facts_example.txt"))
    # The file has 2 comment lines and 4 facts.
    assert len(facts) == 4
    # No comment text leaked in.
    for fact in facts:
        assert "#" not in fact[:1], "leading-# comment should have been stripped"


def test_hallucination_example_prompts_have_no_assertions_field() -> None:
    """Hallucination test inputs typically don't carry assertions - they're
    scored by the judge. Verify the example reflects that pattern.
    """
    prompts = load_batch_file(str(EXAMPLES_DIR / "hallucination_test.json"))
    for bp in prompts:
        # Assertions list should be empty - judge does the scoring.
        assert bp.assertions == []
