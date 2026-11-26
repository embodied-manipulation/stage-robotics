# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independently restricted prompts with optional fixed manipulation guidance."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from subgoal_core.prompts import scene_evidence_and_recovery
from robots.robocasa.prompts import evaluate
from rpent.prompt.utils import BulletList, PromptNode
from robots.robocasa.restrictions import get_restrictions

INTERFACE_PATH = Path(__file__).with_name("interface_only.md")

SCENE_EVIDENCE_AND_RECOVERY = scene_evidence_and_recovery()


def interface_sections() -> dict[str, str]:
    """Read the exact interface text shared by the prompt and audit."""
    sections = {}
    for block in INTERFACE_PATH.read_text(encoding="utf-8").split("## ")[1:]:
        heading, body = block.split("\n", 1)
        sections[heading.strip()] = body.strip()
    return sections


def system_prompt(variables: Mapping[str, object] | None) -> dict[str, PromptNode]:
    """Restrict the upstream prompt without replacing its manipulation policy."""
    flags = get_restrictions()
    no_memory = flags.disable_task_memory and flags.disable_global_memory
    sections = evaluate.system_prompt(None if no_memory else variables)
    if flags.hide_task_feedback:
        sections["Intro"] = sections["Intro"].replace(
            "- **SUCCESS = the task's own _check_success()** (surfaced as state.success).\n"
            "  You drive until state.success == true or the command budget runs out.\n", "")
        sections["Goal"] = evaluate.GOAL.split(" Success is determined", 1)[0]
        sections["Rules"] = BulletList([
            item for item in evaluate.RULES.items
            if "task_progress" not in item and "state.success" not in item
        ])
        sections["Workflow"] = evaluate.WORKFLOW.replace(
            "Check task_language and success_criteria.", "Check task_language."
        ).replace(
            "view_env_state to check task_progress.",
            "view_env_state to inspect the current images and proprioception."
        ).replace(
            "When state.success == true, call finish(status='success').",
            "When you judge the task complete from observations, call finish(status='success')."
        )
    else:
        # The native response has success beside state, never inside it.
        for name in ("Intro", "Goal", "Workflow"):
            sections[name] = sections[name].replace("state.success", "top-level success")
        sections["Rules"] = BulletList([
            item.replace("state.success", "top-level success")
            for item in sections["Rules"].items
        ])
    if flags.hide_task_feedback or flags.disable_privileged_grasp:
        sections["VLA_Rules"] = evaluate.VLA_RULES.replace(
            "Contact,\n   a held object, fixture progress, or a rising task counter means productive\n   progress",
            "Visible task progress means productive\n   progress"
        )
    if flags.disable_bootstrap:
        sections["Primitives"] = evaluate.PRIMITIVES.replace(
            "- reset(): Restart episode (only in EXPLORE mode).\n", "")
    if flags.disable_task_memory and flags.disable_global_memory:
        sections.pop("Memory", None)
    if flags.retain_task_independent_guidance:
        from rpent.prompt.manipulation_guidance import task_independent_guidance

        sections["Task_Independent_Guidance"] = task_independent_guidance()
    sections['Rules'] = BulletList([
        ('Work through recoverable local failures using Scene_Evidence_And_Recovery; '
         'reserve finish(status=\'stuck\') for an evidenced terminal blocker.')
        if 'stuck' in item else item for item in sections['Rules'].items
    ])
    sections['Workflow'] = sections['Workflow'].replace(
        "7. If stuck, call finish(status='stuck', summary='...').",
        '7. REASSESS: Continue a suitable unfinished goal. When a concrete blocker requires recovery, identify a feasible changed approach, '
        'execute it and inspect the outcome. Apply Scene_Evidence_And_Recovery '
        'before choosing a terminal stuck assessment.'
    )
    from research.subgoal_review import guidance
    sections['Subgoal_Review'] = guidance()
    sections['Scene_Evidence_And_Recovery'] = SCENE_EVIDENCE_AND_RECOVERY
    return sections
