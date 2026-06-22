"""Existence and parseability checks for documentation, CI, and example files.

YAML tests are skipped cleanly when PyYAML isn't installed - CI itself
runs the YAML files, which is the ultimate validation, and downstream
developers who care can `pip install pyyaml`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

REPO_ROOT = Path(__file__).parent.parent


# ===== top-level documentation =====


class TestRootDocumentation:
    @pytest.mark.parametrize(
        "filename",
        ["SECURITY.md", "CONTRIBUTING.md", "README.md", "LICENSE", "NOTICE"],
    )
    def test_file_exists(self, filename: str) -> None:
        path = REPO_ROOT / filename
        assert path.exists(), f"missing top-level file: {filename}"
        assert path.is_file()
        assert path.stat().st_size > 0, f"{filename} is empty"

    def test_security_md_mentions_disclosure_process(self) -> None:
        text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
        # Must instruct against public disclosure and provide a reporting path.
        assert "Do NOT" in text or "do not" in text.lower()
        assert "report" in text.lower()
        # Mentions all the security primitives the rest of the codebase implements.
        assert "keyring" in text.lower()
        assert "redaction" in text.lower() or "redact" in text.lower()
        assert "localhost" in text.lower()

    def test_contributing_md_mentions_test_and_lint(self) -> None:
        text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "pytest" in text.lower()
        assert "ruff" in text.lower()


# ===== GitHub configuration =====


class TestGitHubConfig:
    def test_ci_workflow_exists(self) -> None:
        path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert path.exists(), "missing .github/workflows/ci.yml"
        assert path.stat().st_size > 0

    def test_dependabot_exists(self) -> None:
        path = REPO_ROOT / ".github" / "dependabot.yml"
        assert path.exists(), "missing .github/dependabot.yml"
        assert path.stat().st_size > 0

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_ci_workflow_parses_as_yaml(self) -> None:
        text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        parsed = _yaml.safe_load(text)
        # Minimal contract: `on` and `jobs` keys, matrix structure intact.
        # YAML treats bare `on` as the boolean True - the spec keeps it
        # this way so we tolerate either key.
        assert "jobs" in parsed
        on_key = "on" if "on" in parsed else True
        assert on_key in parsed
        assert "test" in parsed["jobs"]
        matrix = parsed["jobs"]["test"]["strategy"]["matrix"]
        # The spec requires 3 OS x 4 Python versions = 12 jobs.
        assert len(matrix["os"]) == 3
        assert len(matrix["python-version"]) == 4

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_dependabot_parses_as_yaml(self) -> None:
        text = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        parsed = _yaml.safe_load(text)
        assert parsed["version"] == 2
        # Both pip and github-actions ecosystems registered.
        ecosystems = [u["package-ecosystem"] for u in parsed["updates"]]
        assert "pip" in ecosystems
        assert "github-actions" in ecosystems


# ===== examples directory =====


EXAMPLES_DIR = REPO_ROOT / "examples"

EXPECTED_EXAMPLES = [
    "basic_comparison.sh",
    "batch_evaluation.json",
    "hallucination_test.json",
    "expected_facts_example.txt",
    "ci_eval_suite.json",
    "github_actions_workflow.yml",
    "README.md",
    # v0.1.x statistical feature demos
    "reproducibility_analysis.sh",
    "statistical_significance.sh",
    "publication_grade_eval.sh",
    "mcnemar_hallucination.sh",
]


class TestExamplesDirectory:
    def test_examples_dir_exists(self) -> None:
        assert EXAMPLES_DIR.exists()
        assert EXAMPLES_DIR.is_dir()

    @pytest.mark.parametrize("filename", EXPECTED_EXAMPLES)
    def test_example_file_exists(self, filename: str) -> None:
        path = EXAMPLES_DIR / filename
        assert path.exists(), f"missing example: {filename}"
        assert path.stat().st_size > 0, f"empty example: {filename}"

    @pytest.mark.parametrize(
        "filename",
        ["batch_evaluation.json", "hallucination_test.json", "ci_eval_suite.json"],
    )
    def test_json_example_parses(self, filename: str) -> None:
        text = (EXAMPLES_DIR / filename).read_text(encoding="utf-8")
        # All three batch input files are top-level arrays.
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) > 0
        # Each element must have at least an "id" and "prompt" - or just "prompt".
        for item in data:
            assert "prompt" in item

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_github_actions_example_parses(self) -> None:
        text = (EXAMPLES_DIR / "github_actions_workflow.yml").read_text(encoding="utf-8")
        parsed = _yaml.safe_load(text)
        assert "jobs" in parsed
        assert "evaluate" in parsed["jobs"]

    def test_expected_facts_example_loads_via_real_loader(self) -> None:
        """End-to-end: the example facts file must load through the real loader."""
        from cli_modelarium.hallucination import load_expected_facts

        facts = load_expected_facts(str(EXAMPLES_DIR / "expected_facts_example.txt"))

        # Comments stripped, 4 facts loaded (per the file content).
        assert len(facts) == 4
        assert "The first moon landing was July 20, 1969" in facts
        assert "Neil Armstrong was the first person to walk on the moon" in facts
        # Comment lines are NOT in the result.
        for fact in facts:
            assert not fact.startswith("#")

    @pytest.mark.parametrize(
        "filename",
        [
            "basic_comparison.sh",
            "reproducibility_analysis.sh",
            "statistical_significance.sh",
            "publication_grade_eval.sh",
            "mcnemar_hallucination.sh",
        ],
    )
    def test_shell_example_has_shebang(self, filename: str) -> None:
        """All .sh examples must have a shebang to be runnable."""
        text = (EXAMPLES_DIR / filename).read_text(encoding="utf-8")
        assert text.startswith("#!"), f"{filename} missing shebang"
