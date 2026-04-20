"""submit_answer tool: terminates the agent loop and records the summary."""
from __future__ import annotations

from typing import Any


class SubmitAnswerTool:
    """Stateful tool that latches once submit_answer has been called.

    The runner polls `.submitted` between steps to decide whether to break
    out of the loop.
    """

    def __init__(self) -> None:
        self.submitted: bool = False
        self.summary: str = ""

    def execute(self, summary: str = "", **_unused: Any) -> dict[str, Any]:
        self.submitted = True
        self.summary = summary if isinstance(summary, str) else str(summary)
        return {"success": True, "summary": self.summary, "terminated": True}
