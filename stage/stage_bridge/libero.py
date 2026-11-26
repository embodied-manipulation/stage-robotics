"""LIBERO adapter for the same continuous Subgoal protocol used in RoboCasa.

The native policy encoder, model forward, five-action chunks and observation
preprocessing remain in robots.libero. No task recipes are loaded here.
"""
from pathlib import Path

from rpent.memory.manager import MemoryManager
from rpent.tools import ToolResult, tool
from research import review_window
from subgoal_core.protocol import full_task_prompt
from subgoal_core.prompts import (
    LIBERO_PROMPT_PROTOCOL, LEGACY_LIBERO_PROMPT_PROTOCOL,
    VLA_OBSERVATION_CHECKPOINTS, vla_instruction_guidance,
    pure_vla_workflow, scene_evidence_and_recovery,
)


class EpisodeFiles(MemoryManager):
    """Current-episode files only; no Task/Global Memory or cross-run reads."""
    file_tool_description = ' Only current-episode public files are available.'

    def _authorize(self, path, *, write):
        path = Path(path)
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError('Only current-episode public files are available')
        return resolved


def select_observation(result, state):
    """Deliver the configured native LIBERO images in their original orientation."""
    if not result.images:
        return result
    step = result.data.get('step')
    if type(step) is not int or 'artifacts' not in result.data:
        # Camera-window delivery must come from a recorded native observation.
        result.images = []
        return result
    artifacts = set(result.data['artifacts'])
    images, descriptors = [], []
    from subgoal_core.observations import _window, LIBERO_CAMERAS
    cameras = _window.cameras if _window is not None else LIBERO_CAMERAS
    for camera in cameras:
        names = (f'{camera}.png',) if camera == 'agentview_policy' else (f'{camera}_high.png', f'{camera}.png')
        name = next((n for n in names if n in artifacts), None)
        if name is None:
            raise ValueError(f'Missing calibrated LIBERO {camera} observation')
        images.append(state.load_bytes(name, step=step))
        descriptors.append({'camera': camera, 'artifact': name})
    result.images = images
    result.data['images'] = descriptors
    if 'agentview_policy' not in cameras:
        result.data['artifacts'] = [n for n in result.data['artifacts'] if n != 'agentview_policy.png']
    return result


class PureSkill:
    def __init__(self, primitives, *, local_outcome_only=True):
        self.primitives = primitives
        self.local_outcome_only = local_outcome_only

    @tool(name='pi0_doubled')
    def run(self, prompt: str) -> ToolResult:
        """Execute the accepted Subgoal with Pi0.5 and observe within 40 actions.

        Supports grasp, transport, place, articulated-object and other skills.
        Uses native five-action model chunks. This call can return earlier on
        native task termination or truncation; a return is not Subgoal success.

        Args:
            prompt: Exactly the instruction accepted by subgoal_review.
        """
        primitives = self.primitives
        task_language = getattr(primitives, '_last_obs', {}).get('task_descriptions', '')
        if self.local_outcome_only and full_task_prompt(prompt, task_language):
            return ToolResult(data={'ok': False, 'error_code': 'full_task_prompt_disallowed',
                'error': 'Use the accepted local observable Subgoal, not the complete task instruction.',
                'steps_applied': 0})
        chunks = review_window.remaining_chunks(8, 5)
        used = 0
        primitives.begin_primitive('pi0_doubled')
        try:
            for _ in range(chunks):
                if primitives.env.terminated or primitives.env.truncated:
                    break
                primitives._vlm_chunk(prompt)
                used += 1
        finally:
            primitives.end_primitive()
        return ToolResult(data={
            'name': 'pi0_doubled', 'instruction': prompt, 'chunks_used': used,
            'terminated': primitives.env.terminated, 'truncated': primitives.env.truncated,
            'success_meaning': 'Only top-level native terminated reports task success; '
                               'assess the local Subgoal from current observations.',
        })


def install(toolkit, private, *, prompt_protocol=LIBERO_PROMPT_PROTOCOL, status_protocol=None,
            code_as_policy=False, classification_enabled=True, vla_enabled=True):
    """Reuse shared decision, action-budget and image-delivery enforcement."""
    from research.subgoal_review import install as install_subgoals
    toolkit._tools.pop('pi0_pick', None)
    if vla_enabled:
        toolkit.add_tool(PureSkill(toolkit._primitives,
            local_outcome_only=prompt_protocol != LEGACY_LIBERO_PROMPT_PROTOCOL).run, replace=True)
    else:
        toolkit._tools.pop('pi0_doubled', None)
    original_execute = toolkit.execute_tool

    def execute(name, inputs):
        return select_observation(original_execute(name, inputs), toolkit.state)

    toolkit.execute_tool = execute
    if classification_enabled:
        install_subgoals(toolkit, private, code_as_policy=code_as_policy, status_protocol=status_protocol,
                         vla_enabled=vla_enabled)
    else:
        if not code_as_policy:
            raise ValueError('Classification-off is a Hybrid-only ablation')
        from stage_bridge.unclassified import install as install_unclassified
        install_unclassified(toolkit, private)
    return review_window.install(toolkit, 40, private)


def shared_prompt_sections(task_feedback_protocol=None, *, code_as_policy=False, classification_enabled=True,
                           vla_enabled=True, planner_observation_profile='calibration-two-view/v1'):
    """The XR-1 behavior with only native LIBERO interface names substituted."""
    from research.subgoal_review import guidance
    from rpent.prompt.common import OUTPUT
    sections = {
        'Intro': 'You control a fixed-base Panda robot in a LIBERO-PRO simulation. '
                 'Read the live task_language in view_env_state. Physical execution '
                 'uses the Pi0.5 VLA through pi0_doubled.',
        'Goal': 'Complete the manipulation task described in task_language. '
                'Top-level terminated in a recorded observation is native whole-task success; '
                'truncated is the environment limit. Local Subgoal completion is assessed '
                'from observations and does not by itself establish task success.',
        'Primitives': 'EXECUTION TOOL:\n'
                      '- pi0_doubled(prompt): Learned VLA for the accepted local outcome.\n'
                      'OBSERVATION AND DECISION TOOLS:\n'
                      '- view_env_state: Read recorded images and state.\n'
                      '- back_project / view_camera_meta: Read-only geometry and calibration.\n'
                      '- subgoal_review: Record start / continue / advance / recover.\n'
                      '- finish(status, summary): Report completion after reviewing the latest state.\n'
                      'All physical motion, including grasp, carry and release, is executed by VLA.',
        'VLA_Rules': vla_instruction_guidance('pi0_doubled', local_outcome_only=True),
        'Workflow': pure_vla_workflow(
            execution='Use pi0_doubled for learned manipulation.',
            actions='gripper changes and placement', success_field='terminated'),
        'Observation_Interface': 'The VLA uses its native external and wrist images and '
            'proprioception. The Planner sees calibrated external and wrist images with '
            'native robot state and task feedback. Calibrated images may differ in '
            'orientation from the offline policy PNG. Match geometric queries to the '
            'selected calibrated view. Read-only geometry tools do not move the robot.',
        'Subgoal_Review': guidance(vla_enabled=vla_enabled),
        'Scene_Evidence_And_Recovery': scene_evidence_and_recovery(
            success_field='terminated', progress_source='the task requirements',
            criteria_source='the task_language requirements'),
        'VLA_Observation_Checkpoints': VLA_OBSERVATION_CHECKPOINTS,
        'Episode': 'Complete the goal within one continuous environment episode. Local retries, '
            're-localization, re-staging and changed Subgoals are part of this same attempt. '
            'Use the current scene for recovery and preserve progress already achieved. '
            'The episode provides no reset operation. Only current-episode public files '
            'are available. Task/Global Memory and online RSI are disabled.',
        'Output': OUTPUT,
        'Next': 'Call view_env_state(step=0) and forward its result text. '
            'The framework attaches the visual window to the next Planner request.',
    }
    if task_feedback_protocol:
        from stage_bridge.libero_feedback import PROTOCOL, SYSTEM_GUIDANCE
        if task_feedback_protocol != PROTOCOL:
            raise ValueError(f'Unsupported task feedback: {task_feedback_protocol}')
        sections['Native_Task_Feedback'] = SYSTEM_GUIDANCE
        if not classification_enabled:
            from stage_bridge.unclassified import language_guidance as without_declarations
            sections['Native_Task_Feedback'] = without_declarations(SYSTEM_GUIDANCE)
        sections['Scene_Evidence_And_Recovery'] = scene_evidence_and_recovery(
            success_field='terminated', progress_source='task_progress and grasp_feedback',
            criteria_source='success_criteria')
    if code_as_policy:
        sections['Primitives'] = sections['Primitives'].replace(
            'All physical motion, including grasp, carry and release, is executed by VLA.',
            'Available executors: VLA and Codes. Codes may compose the exposed native '
            'move_to, move_delta, move_pose, rotate_wrist, rotate_pitch, set_gripper '
            'and release tools. Use their tool schemas and current calibrated geometry. '
            'Choose either executor for the same active local outcome.')
        sections['Intro'] = sections['Intro'].replace(
            'uses the Pi0.5 VLA through pi0_doubled.',
            'uses Pi0.5 through pi0_doubled or the exposed analytic tools through Codes.')
    if planner_observation_profile == 'libero-stage-policy-plus-calibrated/v1':
        sections['Observation_Interface'] = (
            'The VLA uses its native external and wrist images and proprioception. '
            'At each observed timestamp, the Planner receives agentview_policy.png '
            '(external image in VLA preprocessing orientation), calibrated agentview_high.png '
            'and wrist_high.png (or their low-resolution fallbacks), with native robot state '
            'and task feedback. These are three images from two cameras of the same state. '
            'Use calibrated agentview/wrist images for pixel selection and match geometric '
            'queries to that view; never use agentview_policy.png pixels for back-projection. '
            'Read-only geometry tools do not move the robot.'
        )
    if not vla_enabled:
        from stage_bridge.codes_only import prompt_sections
        sections = prompt_sections(sections, simulator='libero')
    if not classification_enabled:
        from stage_bridge.unclassified import neutral_sections
        sections = neutral_sections(sections)
    return sections


def prompt_pair(prompt_protocol=LIBERO_PROMPT_PROTOCOL, *, task_feedback_protocol=None,
                language_profile=None, language_profile_path=None, subgoal_status_protocol=None,
                code_as_policy=False, classification_enabled=True, vla_enabled=True,
                planner_observation_profile='calibration-two-view/v1'):
    from stage_bridge.libero_language import guidance as language_guidance
    from subgoal_core.protocol import CONTINUATION_STATUS_PROTOCOL
    from subgoal_core.prompts import CONTINUATION_STATUS_GUIDANCE
    if subgoal_status_protocol not in (None, CONTINUATION_STATUS_PROTOCOL):
        raise ValueError(f'Unsupported Subgoal status protocol: {subgoal_status_protocol}')
    language = language_guidance(prompt_protocol, language_profile, language_profile_path)
    if not vla_enabled:
        language = language.replace('next VLA instruction', 'next local instruction').replace(
            'is passed to the VLA, verbatim.', 'defines the local outcome pursued by generated code.').replace(
            'the VLA prompt', 'the local goal prompt')
    if not classification_enabled:
        from stage_bridge.unclassified import language_guidance as without_declarations
        language = without_declarations(language)
    if prompt_protocol == LEGACY_LIBERO_PROMPT_PROTOCOL:
        if task_feedback_protocol:
            raise ValueError('Full task feedback requires the shared Subgoal prompt')
        system, user = legacy_prompt_pair()
    elif prompt_protocol == LIBERO_PROMPT_PROTOCOL:
        from rpent.prompt.utils import format_prompt
        system, user = (format_prompt(shared_prompt_sections(task_feedback_protocol,
                           code_as_policy=code_as_policy, classification_enabled=classification_enabled,
                           vla_enabled=vla_enabled, planner_observation_profile=planner_observation_profile)),
            'Read the initial state and images, select an observable Subgoal, and classify start.')
    else:
        raise ValueError(f'Unsupported LIBERO prompt protocol: {prompt_protocol}')
    if not classification_enabled:
        user = 'Read the initial state and images, select an observable immediate outcome and instruction.'
    if subgoal_status_protocol and classification_enabled:
        system += '\n\n' + CONTINUATION_STATUS_GUIDANCE
    if language:
        system += '\n\n' + language
    return system, user


def legacy_prompt_pair():
    """Retain the language-candidate pilot's historical prompt identity."""
    from research.subgoal_review import guidance
    system = '''You control a single Panda robot in a LIBERO-PRO simulation.
Read the live task_language in view_env_state. Complete that task using current
observations and the continuous Subgoal protocol below. Physical actions are
executed exclusively by the Pi0.5 VLA through pi0_doubled. Pass the accepted
Subgoal instruction verbatim. The instruction may describe any appropriate
skill, including grasping, carrying, placing, opening, closing or turning.
Choose instructions from current observations and task requirements. A full
task instruction is also valid when it fits the current goal. Keep a productive
instruction stable while the goal is unfinished; change it on completion or
when observed obstacles call for recovery.

The native VLA uses external and wrist images plus robot proprioception. The
Planner sees calibrated external and wrist images in the visual window, with
native robot state and task feedback. These calibrated images may differ in
orientation from the offline policy PNG. Inspect both views for object identity,
contact, holding, placement and articulated-object changes. Depth/calibration
tools are read-only observations, not motion commands.

Top-level terminated in a recorded view_env_state is native task success.
Truncated is the environment limit. A tool return or your visual belief alone
does not prove official success. Local Subgoal completion and task termination
are separate. Finish after reviewing the latest state, or report inability
honestly when the episode can no longer be completed.

Only current-episode public files are available for notes and observations.
No cross-episode Task/Global Memory, per-task recipes, online RSI or resets are
part of this experiment. Begin by calling view_env_state(step=0) and forwarding
its result text so the current visual window can be delivered to the Planner.
''' + guidance()
    return system, 'Call view_env_state(step=0), inspect task_language and the images, then start the first Subgoal.'
