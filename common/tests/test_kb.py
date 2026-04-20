"""Tests for persona-LLM caller."""
import json
from pathlib import Path

import pytest

from common.kb import UNAVAILABLE_MESSAGE, respond_as_persona
from common.rubric_schema import KnowledgeBase, PersonaConfig, QAPair, load_knowledge_base


def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        task_id="t1",
        persona=PersonaConfig(
            name="CTO",
            communication_style="formal and direct",
            disclosure_boundaries="scope only",
        ),
        out_of_scope_response="That's outside my scope.",
        qa_pairs=[
            QAPair(
                id="kb-001",
                disclosure_trigger="Questions about the S3 bucket",
                answer="Use crm-migration-2026.",
            )
        ],
    )


def test_respond_uses_llm(tmp_path):
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Use crm-migration-2026. — CTO"

    out = respond_as_persona(_kb(), "where's the S3 bucket?", fake_llm)
    assert out == "Use crm-migration-2026. — CTO"
    assert "You are CTO" in captured["prompt"]
    assert "Questions about the S3 bucket" in captured["prompt"]
    assert "That's outside my scope." in captured["prompt"]
    assert "where's the S3 bucket?" in captured["prompt"]


def test_respond_on_llm_error_returns_sentinel():
    def bad_llm(prompt: str) -> str:
        raise RuntimeError("rate limit")

    out = respond_as_persona(_kb(), "anything", bad_llm)
    assert out == UNAVAILABLE_MESSAGE


def test_load_knowledge_base_happy_path(tmp_path):
    raw = {
        "task_id": "t1",
        "persona": {"name": "a", "communication_style": "b", "disclosure_boundaries": "c"},
        "out_of_scope_response": "x",
        "qa_pairs": [{"id": "kb-001", "disclosure_trigger": "t", "answer": "a"}],
    }
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(raw))
    kb = load_knowledge_base(path)
    assert kb.task_id == "t1"
