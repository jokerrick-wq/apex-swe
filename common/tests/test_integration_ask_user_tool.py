"""Smoke test for integration ask_user tool (no Docker required)."""
import json
import sys
from pathlib import Path

import pytest

# Integration tools live outside common/; add to path
HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "integration"))


def test_ask_user_tool_returns_answer(tmp_path):
    kb_raw = {
        "task_id": "t1",
        "persona": {
            "name": "CTO",
            "communication_style": "formal",
            "disclosure_boundaries": "scope only",
        },
        "out_of_scope_response": "not my area",
        "qa_pairs": [
            {"id": "kb-001", "disclosure_trigger": "bucket", "answer": "use crm-migration-2026"}
        ],
    }
    kb_path = tmp_path / "knowledge_base.json"
    kb_path.write_text(json.dumps(kb_raw))

    from src.tools.ask_user_tool import AskUserTool

    tool = AskUserTool.from_task_dir(tmp_path)
    tool._llm = lambda prompt: "use crm-migration-2026"

    result = tool.execute(question="where's the bucket?")
    assert result["success"] is True
    assert result["answer"] == "use crm-migration-2026"


def test_ask_user_tool_rejects_empty_question(tmp_path):
    kb_raw = {
        "task_id": "t1",
        "persona": {"name": "x", "communication_style": "x", "disclosure_boundaries": "x"},
        "out_of_scope_response": "x",
        "qa_pairs": [],
    }
    (tmp_path / "knowledge_base.json").write_text(json.dumps(kb_raw))

    from src.tools.ask_user_tool import AskUserTool

    tool = AskUserTool.from_task_dir(tmp_path)
    result = tool.execute(question="")
    assert result["success"] is False
    assert "non-empty" in result["error"]
