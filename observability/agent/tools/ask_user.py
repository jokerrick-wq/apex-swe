"""ask_user Inspect tool: persona-LLM router for observability tasks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store

from common.kb import respond_as_persona
from common.rubric_schema import KnowledgeBase, load_knowledge_base


KB_STORE_KEY = "kosmos.ask_user.kb"


def _default_persona_model() -> str:
    return os.environ.get("KOSMOS_PERSONA_MODEL", "claude-opus-4-7")


def _call_persona(prompt: str) -> str:
    import litellm  # type: ignore

    resp = litellm.completion(
        model=_default_persona_model(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def load_kb_for_sample(task_dir: Path | str) -> KnowledgeBase:
    return load_knowledge_base(Path(task_dir) / "knowledge_base.json")


@tool
def ask_user(kb: KnowledgeBase | None = None, llm_fn: Callable[[str], str] | None = None) -> Tool:
    """Ask the user (a persona LLM) a clarifying question about the task."""

    effective_llm = llm_fn or _call_persona

    async def execute(question: str) -> str:
        """Ask the task's stakeholder a clarifying question.

        Args:
            question: The clarification question to ask.
        """
        # Resolve the KB: prefer one injected at tool-construction time; fall
        # back to one stashed in the Inspect sample store by the runner.
        effective_kb = kb
        if effective_kb is None:
            try:
                effective_kb = store().get(KB_STORE_KEY)
            except Exception:
                effective_kb = None
        if effective_kb is None:
            return "[ask_user unavailable: knowledge base not loaded]"
        if not isinstance(question, str) or not question.strip():
            return "[ask_user error: question must be a non-empty string]"
        return respond_as_persona(effective_kb, question, effective_llm)

    return execute
