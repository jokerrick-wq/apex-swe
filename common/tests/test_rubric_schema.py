"""Tests for KB + Rubric dataclass validators."""
import json
from pathlib import Path

import pytest

from common.rubric_schema import (
    KnowledgeBase,
    PersonaConfig,
    QAPair,
    Rubric,
    RubricCriterion,
    load_knowledge_base,
    load_rubric,
)


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


# --- KnowledgeBase ---

def test_kb_roundtrip(tmp_path):
    raw = {
        "task_id": "t1",
        "persona": {
            "name": "CTO",
            "communication_style": "formal",
            "disclosure_boundaries": "scope only",
        },
        "out_of_scope_response": "not my area",
        "qa_pairs": [
            {
                "id": "kb-001",
                "disclosure_trigger": "bucket questions",
                "answer": "use crm-migration-2026",
            }
        ],
    }
    kb = load_knowledge_base(_write(tmp_path, "kb.json", raw))
    assert isinstance(kb, KnowledgeBase)
    assert kb.task_id == "t1"
    assert kb.persona.name == "CTO"
    assert len(kb.qa_pairs) == 1
    assert kb.qa_pairs[0].id == "kb-001"


def test_kb_missing_persona_raises(tmp_path):
    raw = {
        "task_id": "t1",
        "out_of_scope_response": "x",
        "qa_pairs": [],
    }
    with pytest.raises(ValueError, match="persona"):
        load_knowledge_base(_write(tmp_path, "kb.json", raw))


def test_kb_missing_qa_answer_raises(tmp_path):
    raw = {
        "task_id": "t1",
        "persona": {"name": "x", "communication_style": "x", "disclosure_boundaries": "x"},
        "out_of_scope_response": "x",
        "qa_pairs": [{"id": "kb-001", "disclosure_trigger": "x"}],
    }
    with pytest.raises(ValueError, match="answer"):
        load_knowledge_base(_write(tmp_path, "kb.json", raw))


def test_kb_invalid_json_raises(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="parse"):
        load_knowledge_base(path)


# --- Rubric ---

def test_rubric_roundtrip(tmp_path):
    raw = {
        "task_id": "t1",
        "criteria": [
            {
                "id": "style-001",
                "category": "code_style",
                "description": "type hints required",
                "artifacts": ["/app/migrate.py"],
            }
        ],
    }
    r = load_rubric(_write(tmp_path, "rubric.json", raw))
    assert isinstance(r, Rubric)
    assert len(r.criteria) == 1
    assert r.criteria[0].category == "code_style"


def test_rubric_unknown_category_raises(tmp_path):
    raw = {
        "task_id": "t1",
        "criteria": [
            {
                "id": "x-001",
                "category": "bogus",
                "description": "x",
                "artifacts": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="category"):
        load_rubric(_write(tmp_path, "rubric.json", raw))


def test_rubric_empty_criteria_raises(tmp_path):
    raw = {"task_id": "t1", "criteria": []}
    with pytest.raises(ValueError, match="criteria"):
        load_rubric(_write(tmp_path, "rubric.json", raw))
