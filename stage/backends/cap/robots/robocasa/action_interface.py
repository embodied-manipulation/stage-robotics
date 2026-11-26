"""Opt-in language routing, action guidance and reporting, separate from access limits."""

import os

from subgoal_core.protocol import full_task_prompt
from rpent.prompt.utils import BulletList
from subgoal_core.prompts import (vla_instruction_guidance, pure_vla_workflow,
                                 VLA_OBSERVATION_CHECKPOINTS)

TASK_PROGRESS_REPORTING = (
    "Also include your estimated task_progress as a compact JSON object, with entries "
    "derived from the task instruction and observations; use null for unknown values."
)


def decision_reporting_profile() -> str:
    profile = os.environ.get("STAGE_DECISION_REPORTING", "upstream")
    if profile not in {"upstream", "task-progress-v1"}:
        raise ValueError("STAGE_DECISION_REPORTING must be upstream or task-progress-v1")
    return profile


def vla_prompt_mode() -> str:
    mode = os.environ.get("STAGE_VLA_PROMPT_MODE", "task_language")
    if mode not in {"task_language", "provided"}:
        raise ValueError("STAGE_VLA_PROMPT_MODE must be task_language or provided")
    return mode


def vla_instruction_scope() -> str:
    scope = os.environ.get("STAGE_VLA_INSTRUCTION_SCOPE", "unrestricted")
    if scope not in {"unrestricted", "subgoal_only"}:
        raise ValueError("STAGE_VLA_INSTRUCTION_SCOPE must be unrestricted or subgoal_only")
    if scope == "subgoal_only" and vla_prompt_mode() != "provided":
        raise ValueError("subgoal_only requires STAGE_VLA_PROMPT_MODE=provided")
    return scope




def hybrid_guidance_enabled() -> bool:
    value = os.environ.get("STAGE_HYBRID_GUIDANCE", "0").lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("STAGE_HYBRID_GUIDANCE must be 0, 1, false or true")
    return value in {"1", "true"}


def configure_prompt(sections: dict) -> dict:
    sections = dict(sections)
    if os.environ.get("STAGE_PLANNER_OBSERVATION_PROFILE") == "vla-views-plus-nav/v1":
        sections["Intro"] = sections["Intro"].replace(
            "**THREE cameras**", "**FOUR cameras**"
        ).replace(
            "and wrist (eye_in_hand). agentview = WHAT",
            "wrist (eye_in_hand), and agentview_right (right external RGB). agentview = WHAT",
        )
        sections["Localization"] += (
            "\nADDITIONAL RIGHT VIEW — agentview_right.png is a 256x256 RGB image "
            "from the right external camera also consumed by RLDX. Use it to inspect "
            "occluded objects, cross-check target identity, and assess visible progress. "
            "Match each returned image to its camera/role metadata. This view has no "
            "recorded depth or world map: do not pass its pixels to back_project_batch, "
            "do not request camera='agentview_right' from query_world_map, and do not "
            "reuse its pixel coordinates with another camera. Before an analytic move, "
            "localize the target in a supported calibrated camera at the current state. "
            "If it is occluded there, obtain a suitable view before choosing metric "
            "coordinates; do not invent world xyz from the right RGB alone.\n"
        )
    if decision_reporting_profile() == "task-progress-v1":
        sections["Output"] = BulletList(
            [sections["Output"].items[0], TASK_PROGRESS_REPORTING, *sections["Output"].items[1:]]
        )
    if vla_prompt_mode() == "provided":
        sections["VLA_Rules"] = vla_instruction_guidance(
            'vla_skill or vla_arm', local_outcome_only=vla_instruction_scope() == 'subgoal_only')
        sections['Workflow'] = sections['Workflow'].replace(
            'If the target is far (|xy| > 0.8m), navigate_to first.',
            'If the target is beyond arm reach (about 0.8m from the current robot base), navigate_to first.'
        ).replace(
            '4. MANIPULATE: Use vla_arm / vla_skill for VLA grasping. Then script\n'
            '   free-space carry (move_to with gripper omitted) and release.',
            '4. MANIPULATE: Pursue the active observable outcome with VLA or analytic tools.\n'
            '   Let VLA continue through the movements within that outcome when suitable.\n'
            '   Choose an analytic motion when observed geometry supports it. An executor\n'
            '   change within the same goal does not itself require a Subgoal transition.'
        )
        for old in ('view_env_state to check task_progress.',
                    'view_env_state to inspect the current images and proprioception.'):
            sections['Workflow'] = sections['Workflow'].replace(
                old, 'inspect the observations already returned by the tool. '
                'Use view_env_state when a needed observation is missing.')
        sections['Primitives'] = sections['Primitives'].replace(
            'VLA arm-only skill (base clamped).', 'VLA arm-focused skill (base motion limited).')
        sections["VLA_Observation_Checkpoints"] = VLA_OBSERVATION_CHECKPOINTS
    if hybrid_guidance_enabled():
        sections["Action_Selection"] = (
            "Choose VLA or analytic primitives to pursue the active observable outcome. "
            "A suitable VLA instruction can remain active across approach, grasp, transport "
            "and release when those movements serve the same outcome. "
            "Use analytic positioning when grounded geometry supports a useful correction "
            "or motion within the current goal. Ground geometric targets in current RGB-D "
            "and calibration using back_project_batch or query_world_map. "
            "Changing executor does not by itself complete or replace the Subgoal. "
            "After observing, continue a suitable instruction; advance on completion evidence "
            "or recover when a concrete blocker calls for a changed immediate instruction."
        )
    if os.environ.get('STAGE_CODE_AS_POLICY') == '0':
        sections['Primitives'] = (
            'EXECUTION TOOLS:\n'
            '- vla_skill(prompt): Learned full-body VLA, including base and arm.\n'
            '- vla_arm(prompt): Learned arm-focused VLA with base motion limited.\n'
            'OBSERVATION AND DECISION TOOLS:\n'
            '- view_env_state: Read recorded images and state.\n'
            '- back_project_batch / query_world_map: Read-only geometric observations.\n'
            '- subgoal_review: Record start / continue / advance / recover from current evidence.\n'
            '- finish: Report completion after reviewing the latest state.\n'
            'All physical motion, including navigation, grasp, carry and release, is executed by VLA.'
        )
        sections['Workflow'] = pure_vla_workflow(
            execution='Use vla_skill for full-body execution or vla_arm when base motion should be limited.',
            actions='navigation, gripper changes and placement',
        )
        sections.pop('Action_Selection', None)
        sections.pop('Task_Independent_Guidance', None)
    return sections


_existing_configure_prompt = configure_prompt


def configure_prompt(sections: dict) -> dict:
    """Retain V's behavior text; H changes only advertised executor capability."""
    variant = os.environ.get('STAGE_VARIANT')
    if variant is None or variant == 'V':
        return _existing_configure_prompt(sections)
    if variant not in {'H', 'H_unclassified', 'C'}:
        raise ValueError('Unknown final variant')
    old_codes = os.environ.get('STAGE_CODE_AS_POLICY')
    old_guidance = os.environ.get('STAGE_HYBRID_GUIDANCE')
    try:
        os.environ['STAGE_CODE_AS_POLICY'] = '0'
        os.environ['STAGE_HYBRID_GUIDANCE'] = '0'
        result = _existing_configure_prompt(sections)
    finally:
        for key, value in [('STAGE_CODE_AS_POLICY', old_codes),
                           ('STAGE_HYBRID_GUIDANCE', old_guidance)]:
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result['Primitives'] = result['Primitives'].replace(
        'All physical motion, including navigation, grasp, carry and release, is executed by VLA.',
        'Available executors: VLA and Codes. Codes may compose the exposed native '
        'move_to, move_delta, rotate_pitch, set_gripper, release, scripted_grasp, '
        'navigate_to, move_base and execute_action tools. Use their tool schemas '
        'and current calibrated geometry. Choose either executor for the same active local outcome.')
    if variant == 'C':
        from stage_bridge.codes_only import prompt_sections
        result = prompt_sections(result, simulator='robocasa')
    if variant == 'H_unclassified':
        from stage_bridge.unclassified import neutral_sections
        result = neutral_sections(result)
    return result
