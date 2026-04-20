"""ask_user tool: routes the agent's question to the persona LLM."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Wire common/ into sys.path. Mirrors the pattern in multi_step_runner.py.
# Needed because this tool module is imported before the runner's own wiring runs.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.kb import respond_as_persona  # noqa: E402
from common.rubric_schema import KnowledgeBase, load_knowledge_base  # noqa: E402


def _default_persona_model() -> str:
    return os.environ.get("KOSMOS_PERSONA_MODEL", "claude-opus-4-7")


def _call_persona(prompt: str) -> str:
    """Real LLM call via litellm. Imported lazily so unit tests don't require litellm."""
    import litellm  # type: ignore

    resp = litellm.completion(
        model=_default_persona_model(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


class AskUserTool:
    """Stateful tool holding a loaded knowledge base for a single trial."""

    def __init__(self, kb: KnowledgeBase, llm_call=_call_persona):
        self.kb = kb
        self._llm = llm_call

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> "AskUserTool":
        kb_path = Path(task_dir) / "knowledge_base.json"
        return cls(load_knowledge_base(kb_path))

    def execute(self, question: str, **_unused: Any) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            return {"success": False, "error": "ask_user requires a non-empty 'question' string"}
        answer = respond_as_persona(self.kb, question, self._llm)
        return {"success": True, "answer": answer}
