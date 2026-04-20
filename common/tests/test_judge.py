"""Tests for rubric judge."""
import json
from pathlib import Path
from typing import Callable

import pytest

from common.judge import grade
from common.rubric_schema import Rubric, RubricCriterion


def _rubric() -> Rubric:
    return Rubric(
        task_id="t1",
        criteria=[
            RubricCriterion(
                id="style-001",
                category="code_style",
                description="Type hints everywhere",
                artifacts=["dummy.py"],
            ),
            RubricCriterion(
                id="readability-001",
                category="readability",
                description="No function > 60 LOC",
                artifacts=["dummy.py"],
            ),
        ],
    )


def _always_pass(prompt: str, response_format: str = "json") -> str:
    return json.dumps({"passed": True, "rationale": "all good"})


def _always_fail_then_pass():
    state = {"calls": 0}

    def fn(prompt: str, response_format: str = "json") -> str:
        state["calls"] += 1
        if "style-001" in prompt and state["calls"] == 1:
            return "not json at all"
        return json.dumps({"passed": False, "rationale": "missing type hints on foo"})

    return fn


def _always_malformed(prompt: str, response_format: str = "json") -> str:
    return "not a valid json blob"


def test_grade_all_pass():
    result = grade(_rubric(), "def foo() -> int: pass", _always_pass)
    assert result.passed == 2
    assert result.total == 2
    assert result.score == 1.0
    assert len(result.per_criterion) == 2


def test_grade_retry_on_parse_error(tmp_path):
    fn = _always_fail_then_pass()
    result = grade(_rubric(), "def foo(): pass", fn)
    assert result.total == 2
    assert result.passed == 0
    assert all(r.passed is False for r in result.per_criterion)


def test_grade_double_malformed_marks_null():
    result = grade(_rubric(), "def foo(): pass", _always_malformed)
    assert result.total == 0
    assert result.score == 0.0
    assert all(r.passed is None for r in result.per_criterion)
    assert all(r.rationale == "judge output unparseable" for r in result.per_criterion)


def test_grade_writes_audit_files(tmp_path):
    out = tmp_path / "judge-calls"
    result = grade(_rubric(), "def foo(): pass", _always_pass, out_dir=out)
    assert (out / "style-001.json").exists()
    assert (out / "readability-001.json").exists()
    data = json.loads((out / "style-001.json").read_text())
    assert "prompt" in data and "raw_response" in data and "parsed_judgment" in data


def test_grade_with_empty_solution_text():
    # grade() accepts solution_text as a string; artifact resolution is the caller's
    # responsibility (see Tasks 9-10). Here we verify the engine tolerates an empty
    # solution without crashing and still applies the LLM judgment to each criterion.
    result = grade(_rubric(), "", _always_pass)
    assert result.total == 2
    assert result.passed == 2
    assert result.score == 1.0
