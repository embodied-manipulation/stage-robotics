"""Fixed, observation-based manipulation guidance shared by restricted prompts."""

from pathlib import Path

GUIDANCE_PROFILE = "task-independent-v1"
GUIDANCE_PATH = Path(__file__).with_name("task_independent_guidance.md")


def task_independent_guidance() -> str:
    """Return the reviewed text without consulting an experience corpus."""
    return GUIDANCE_PATH.read_text(encoding="utf-8").strip()
