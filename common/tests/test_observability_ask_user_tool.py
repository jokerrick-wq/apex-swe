"""Smoke test for observability ask_user tool (no Inspect runtime needed)."""
import asyncio
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "observability"))

from common.rubric_schema import KnowledgeBase, PersonaConfig, QAPair  # noqa: E402


def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        task_id="t1",
        persona=PersonaConfig(name="EL", communication_style="x", disclosure_boundaries="x"),
        out_of_scope_response="no",
        qa_pairs=[QAPair(id="kb-001", disclosure_trigger="x", answer="yes")],
    )


# If inspect_ai isn't installed in this venv, these tests skip gracefully.
inspect_ai_available = True
try:
    import inspect_ai  # noqa: F401
except ImportError:
    inspect_ai_available = False


pytestmark = pytest.mark.skipif(not inspect_ai_available, reason="inspect_ai not installed in this venv")


def test_ask_user_inspect_tool_happy_path():
    from agent.tools.ask_user import ask_user

    tool_fn = ask_user(kb=_kb(), llm_fn=lambda prompt: "yes")
    out = asyncio.run(tool_fn(question="is it yes?"))
    assert out == "yes"


def test_ask_user_inspect_tool_rejects_empty():
    from agent.tools.ask_user import ask_user

    tool_fn = ask_user(kb=_kb(), llm_fn=lambda prompt: "noop")
    out = asyncio.run(tool_fn(question="  "))
    assert "non-empty" in out
