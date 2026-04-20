"""Dataclass validators for KB and Rubric JSON files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_CATEGORIES = {"code_style", "readability", "simplicity", "over_engineering"}


@dataclass
class PersonaConfig:
    name: str
    communication_style: str
    disclosure_boundaries: str


@dataclass
class QAPair:
    id: str
    disclosure_trigger: str
    answer: str


@dataclass
class KnowledgeBase:
    task_id: str
    persona: PersonaConfig
    out_of_scope_response: str
    qa_pairs: list[QAPair]


@dataclass
class RubricCriterion:
    id: str
    category: str
    description: str
    artifacts: list[str]


@dataclass
class Rubric:
    task_id: str
    criteria: list[RubricCriterion]


def _read_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse JSON at {path}: {exc}") from exc


def _require(obj: dict, key: str, context: str) -> Any:
    if key not in obj:
        raise ValueError(f"{context} missing required field: {key!r}")
    return obj[key]


def load_knowledge_base(path: Path | str) -> KnowledgeBase:
    raw = _read_json(path)
    ctx = f"knowledge_base at {path}"
    task_id = _require(raw, "task_id", ctx)
    persona_raw = _require(raw, "persona", ctx)
    persona = PersonaConfig(
        name=_require(persona_raw, "name", f"{ctx}.persona"),
        communication_style=_require(persona_raw, "communication_style", f"{ctx}.persona"),
        disclosure_boundaries=_require(persona_raw, "disclosure_boundaries", f"{ctx}.persona"),
    )
    out_of_scope = _require(raw, "out_of_scope_response", ctx)
    qa_raw = _require(raw, "qa_pairs", ctx)
    qa_pairs: list[QAPair] = []
    for idx, item in enumerate(qa_raw):
        qctx = f"{ctx}.qa_pairs[{idx}]"
        qa_pairs.append(QAPair(
            id=_require(item, "id", qctx),
            disclosure_trigger=_require(item, "disclosure_trigger", qctx),
            answer=_require(item, "answer", qctx),
        ))
    return KnowledgeBase(
        task_id=task_id,
        persona=persona,
        out_of_scope_response=out_of_scope,
        qa_pairs=qa_pairs,
    )


def load_rubric(path: Path | str) -> Rubric:
    raw = _read_json(path)
    ctx = f"rubric at {path}"
    task_id = _require(raw, "task_id", ctx)
    criteria_raw = _require(raw, "criteria", ctx)
    if not isinstance(criteria_raw, list) or len(criteria_raw) == 0:
        raise ValueError(f"{ctx}: criteria must be a non-empty list")
    criteria: list[RubricCriterion] = []
    for idx, item in enumerate(criteria_raw):
        cctx = f"{ctx}.criteria[{idx}]"
        category = _require(item, "category", cctx)
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"{cctx}: unknown category {category!r} (valid: {sorted(VALID_CATEGORIES)})"
            )
        criteria.append(RubricCriterion(
            id=_require(item, "id", cctx),
            category=category,
            description=_require(item, "description", cctx),
            artifacts=item.get("artifacts", []),
        ))
    return Rubric(task_id=task_id, criteria=criteria)
