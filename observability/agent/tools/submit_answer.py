"""submit_answer Inspect tool. Used in place of react()'s default submit."""
from __future__ import annotations

from inspect_ai.tool import Tool, tool


@tool
def submit_answer() -> Tool:
    """Signal that the task is complete. Ends the agent loop."""

    async def execute(summary: str = "") -> str:
        """Submit your final answer and end the task.

        Args:
            summary: Brief description of what you did.
        """
        return f"Task complete: {summary}" if summary else "Task complete."

    return execute
