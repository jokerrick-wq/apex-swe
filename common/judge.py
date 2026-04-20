"""LLM-as-a-judge for rubric criteria. One LLM call per criterion."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from common.rubric_schema import Rubric, RubricCriterion

JUDGE_PROMPT_TEMPLATE = """You are evaluating a software engineering solution against a single atomic
code-quality criterion. Your response must be pure JSON with two fields:
"passed" (boolean) and "rationale" (string, 2-4 sentences citing specific
evidence).

CRITERION:
{criterion}

CODE ARTIFACT(S):
{artifacts}

Rules:
- "passed" is true iff the criterion clearly holds in the artifact.
- When uncertain, default to false — this is a quality bar, not a pass-the-test check.
- Rationale must cite specific lines, function names, or constructs from the artifact.
- Do not evaluate correctness; only the stated code-quality criterion.

JSON:
"""

UNPARSEABLE_RATIONALE = "judge output unparseable"


@dataclass
class CriterionResult:
    id: str
    category: str
    description: str
    passed: bool | None
    rationale: str


@dataclass
class RubricGrading:
    passed: int
    total: int
    score: float
    per_criterion: list[CriterionResult]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total": self.total,
            "score": self.score,
            "per_criterion": [
                {
                    "id": r.id,
                    "category": r.category,
                    "description": r.description,
                    "passed": r.passed,
                    "rationale": r.rationale,
                }
                for r in self.per_criterion
            ],
        }


def _try_parse(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "passed" not in parsed or not isinstance(parsed["passed"], bool):
        return None
    if "rationale" not in parsed or not isinstance(parsed["rationale"], str):
        return None
    return parsed


def _write_audit(out_dir: Path, criterion_id: str, prompt: str, raw: str, parsed: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{criterion_id}.json").write_text(
        json.dumps(
            {"prompt": prompt, "raw_response": raw, "parsed_judgment": parsed},
            indent=2,
        )
    )


def _judge_one(
    criterion: RubricCriterion,
    solution_text: str,
    llm_fn: Callable[..., str],
    out_dir: Path | None,
) -> CriterionResult:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        criterion=criterion.description,
        artifacts=solution_text,
    )
    raw = llm_fn(prompt, response_format="json")
    parsed = _try_parse(raw)
    if parsed is None:
        retry_prompt = prompt + "\n\n(Your previous response was not valid JSON. Return only valid JSON now.)"
        raw_retry = llm_fn(retry_prompt, response_format="json")
        parsed = _try_parse(raw_retry)
        if out_dir is not None:
            _write_audit(out_dir, criterion.id, prompt, raw_retry, parsed)
        if parsed is None:
            return CriterionResult(
                id=criterion.id,
                category=criterion.category,
                description=criterion.description,
                passed=None,
                rationale=UNPARSEABLE_RATIONALE,
            )
    else:
        if out_dir is not None:
            _write_audit(out_dir, criterion.id, prompt, raw, parsed)

    return CriterionResult(
        id=criterion.id,
        category=criterion.category,
        description=criterion.description,
        passed=parsed["passed"],
        rationale=parsed["rationale"],
    )


def grade(
    rubric: Rubric,
    solution_text: str,
    llm_fn: Callable[..., str],
    out_dir: Path | str | None = None,
) -> RubricGrading:
    """Grade a solution against a rubric. One LLM call per criterion (plus one retry on parse failure).

    If out_dir is provided, write one <criterion_id>.json per call with the prompt, raw response, and parsed judgment.
    """
    out_path = Path(out_dir) if out_dir is not None else None
    results: list[CriterionResult] = []
    for criterion in rubric.criteria:
        results.append(_judge_one(criterion, solution_text, llm_fn, out_path))

    valid = [r for r in results if r.passed is not None]
    passed = sum(1 for r in valid if r.passed)
    total = len(valid)
    score = (passed / total) if total else 0.0
    return RubricGrading(passed=passed, total=total, score=score, per_criterion=results)
