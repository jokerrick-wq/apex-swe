"""Persona-LLM caller for ask_user. Pure function over a KB + an llm_fn."""
from __future__ import annotations

from typing import Callable

from common.rubric_schema import KnowledgeBase

UNAVAILABLE_MESSAGE = "[user is temporarily unavailable]"

PROMPT_TEMPLATE = """You are {name}. {communication_style}
Disclosure boundaries: {disclosure_boundaries}

You have knowledge relevant to the agent's work. Each entry below has a
disclosure trigger describing the kind of question that entry answers.

Match the agent's question against a trigger. If one matches, answer in
your voice using the entry's content, adapting phrasing to fit your style.
If no trigger matches, respond verbatim with the out-of-scope response.

KNOWLEDGE BASE:
{qa_pairs}

OUT-OF-SCOPE RESPONSE:
{out_of_scope}

AGENT'S QUESTION:
{question}

Answer (in your voice, <= 200 words):
"""


def _format_qa_pairs(kb: KnowledgeBase) -> str:
    lines: list[str] = []
    for pair in kb.qa_pairs:
        lines.append(f"[{pair.id}] Trigger: {pair.disclosure_trigger}")
        lines.append(f"       Answer: {pair.answer}")
        lines.append("")
    return "\n".join(lines).rstrip()


def respond_as_persona(kb: KnowledgeBase, question: str, llm_fn: Callable[[str], str]) -> str:
    """Call the persona LLM with the full KB + the agent's question.

    Returns the persona's text response, or UNAVAILABLE_MESSAGE if the LLM call fails.
    One LLM call per invocation; no retrieval step.
    """
    prompt = PROMPT_TEMPLATE.format(
        name=kb.persona.name,
        communication_style=kb.persona.communication_style,
        disclosure_boundaries=kb.persona.disclosure_boundaries,
        qa_pairs=_format_qa_pairs(kb),
        out_of_scope=kb.out_of_scope_response,
        question=question,
    )
    try:
        return llm_fn(prompt)
    except Exception:
        return UNAVAILABLE_MESSAGE
