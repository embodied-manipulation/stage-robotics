"""Codes-only capability text; shared decision semantics remain unchanged."""
from subgoal_core.prompts import pure_codes_workflow, CODES_OBSERVATION_CHECKPOINTS


def prompt_sections(sections, *, simulator):
    from stage_bridge.review import guidance
    result = dict(sections)
    libero = simulator == 'libero'
    tools = ('move_to, move_delta, move_pose, rotate_wrist, rotate_pitch, set_gripper, release'
             if libero else 'move_to, move_delta, rotate_pitch, set_gripper, release, '
             'scripted_grasp, navigate_to, move_base, execute_action')
    geometry = 'back_project / view_camera_meta' if libero else 'back_project_batch / query_world_map'
    actions = 'gripper changes and placement' if libero else 'navigation, gripper changes and placement'
    result['Primitives'] = ('EXECUTION TOOLS:\n- '+tools+': Native analytic tools composed in generated code.\n'
        'OBSERVATION AND DECISION TOOLS:\n- view_env_state: Read recorded images and state.\n'
        '- '+geometry+': Read-only geometric observations.\n'
        '- subgoal_review: Record start / continue / advance / recover from current evidence.\n'
        '- finish: Report completion after reviewing the latest state.\n'
        'All physical motion, including '+actions+', is executed by Codes.')
    result.pop('VLA_Rules', None)
    result.pop('VLA_Observation_Checkpoints', None)
    result['Codes_Observation_Checkpoints'] = CODES_OBSERVATION_CHECKPOINTS
    result['Workflow'] = pure_codes_workflow(actions=actions, success_field='terminated' if libero else 'success')
    result['Subgoal_Review'] = guidance(vla_enabled=False)
    if libero:
        result['Intro'] = ('You control a fixed-base Panda robot in a LIBERO-PRO simulation. '
                           'Read the live task_language in view_env_state. Physical execution '
                           'uses the exposed analytic tools through Codes.')
        result['Observation_Interface'] = result['Observation_Interface'].replace(
            'The VLA uses its native external and wrist images and proprioception. ', '')
    else:
        # Only capability descriptions are altered; embodiment/camera advice remains.
        result['Intro'] = result['Intro'].replace('RLDX', 'analytic').replace('VLA', 'Codes')
    from stage_bridge.codes_guidance import configure_sections
    return configure_sections(result, simulator=simulator)
