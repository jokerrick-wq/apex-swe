import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "integration"))


def test_submit_answer_latches():
    from src.tools.submit_answer_tool import SubmitAnswerTool

    tool = SubmitAnswerTool()
    assert tool.submitted is False

    out = tool.execute(summary="done")
    assert out["success"] is True
    assert out["terminated"] is True
    assert out["summary"] == "done"
    assert tool.submitted is True
    assert tool.summary == "done"


def test_submit_answer_handles_missing_summary():
    from src.tools.submit_answer_tool import SubmitAnswerTool

    tool = SubmitAnswerTool()
    out = tool.execute()
    assert out["success"] is True
    assert out["summary"] == ""
    assert tool.submitted is True
