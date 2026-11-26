"""Continuous Subgoal decision instructions, independent of backend tools."""

CADENCE_GUIDANCE = """
The framework schedules observation checkpoints shared by VLA and analytic tools,
including several calls in one generated program. These checkpoints do not end
the active Subgoal or require a new instruction. After receiving
the current images, assess the current local Subgoal and choose continue, revise
the instruction, switch executor, or finish based on permitted observations.
Keep the same VLA instruction while continuing the same skill. Use one concrete
skill instruction corresponding to the current local goal.
If review_required is returned, stop the current code program and inspect the
returned images before issuing more motion. In code-mode, forward both the result
text (including review_window.review_token) and the returned images to the model. A partially executed analytic call
has not completed its intended motion; recompute its target from current state.
Do not automatically replay relative moves or whole composite grasp routines.
The window is measured in physical simulation actions, not wall-clock seconds.
Native tool completion can return earlier. Feedback never certifies success
unless the selected information condition explicitly supplies that evidence.
"""


def guidance(*, visual_guidance, checkpoint_guidance, code_as_policy, vla_enabled=True):
    motion_delivery = 'tool, forward both result text (review token included) and all images to GPT;'
    if visual_guidance:
        checkpoint_guidance = checkpoint_guidance.replace(
            'returned images before issuing more motion. In code-mode, forward both the result\n'
            'text (including review_window.review_token) and the returned images to the model.',
            'CURRENT images attached to the next Planner request before issuing more motion.\n'
            'In code-mode, forward the result text, including review_window.review_token.\n'
            'The framework attaches the visual window automatically; do not manually reattach images.'
        )
        motion_delivery = (
            'tool, forward the result text (review token included) to GPT. The framework\n'
            'automatically attaches the visual window for the next decision;'
        )
    mode = ('Hybrid: choose VLA or compose native analytic tools in generated code for each local goal.'
            if code_as_policy else
            'Pure Subgoal VLA: physical execution uses only VLA; analytic motion tools are unavailable.')
    if not vla_enabled:
        mode = 'Codes-only Subgoal: physical execution uses only analytic tools composed in generated code; learned VLA tools are unavailable.'
        checkpoint_guidance = checkpoint_guidance.replace('VLA and analytic tools', 'analytic tools').replace('VLA instruction', 'local instruction').replace('switch executor', 'select analytic actions')
    result = checkpoint_guidance + '\n' + mode + f'''
After the initial images and every new motion observation delivered to GPT,
call subgoal_review BEFORE issuing more motion. Use review_window.action_count
as observation_step (physical actions, not the artifact/tool step number).
Classify start / continue / advance / recover, with concise visible evidence
and decision_reason. start/advance/recover need a local prompt and observable
completion_condition. advance/recover also need transition_evidence.
continue preserves the EXACT accepted prompt and goal; omit prompt on continue.
A suitable Subgoal may span any number of observation checkpoints within the
episode budget. Observation frequency does not set its lifetime. Advance on
completion evidence; recover when an obstacle or uncertainty calls for a
different immediate instruction. An executor change alone keeps the same goal.
Classification itself does not perform a physical action. Match every VLA prompt to the accepted
Subgoal prompt; analytic code must pursue that same local goal. After a motion
{motion_delivery}
never run a blind generated loop through multiple review windows.
Finish only after observing and classifying the latest physical state. A model's
completion claim is separate from the private evaluator's result.
''' + visual_guidance
    if not vla_enabled:
        result = result.replace(' An executor change alone keeps the same goal.', '')
        result = result.replace('Match every VLA prompt to the accepted\nSubgoal prompt; analytic code must pursue that same local goal.',
                                'Analytic code must pursue the accepted Subgoal and its completion condition.')
    return result


def pure_codes_workflow(*, actions, success_field='success'):
    return ('Read the task and initial images. Select an observable Subgoal and classify start. '
            'Compose the exposed analytic tools in generated code to pursue the accepted Subgoal, '
            'including '+actions+'. After returned observations, classify continue while the '
            'current goal remains suitable; advance on completion evidence or recover on an '
            'observed blocker. With official feedback, finish success only after the latest '
            'observation reports '+success_field+'.')


# Exact behavioral text extracted from the completed XR-1 configuration.
# Backend bindings change tool, feedback-field and embodiment names only.
SHARED_PROMPT_PROFILE = 'xr1-observation-led-subgoal/v1'
LIBERO_PROMPT_PROTOCOL = 'continuous-subgoal-shared/libero-v2'
LEGACY_LIBERO_PROMPT_PROTOCOL = 'continuous-subgoal-no-memory/libero-v1'

CONTINUATION_STATUS_GUIDANCE = """
SUBGOAL EXECUTION STATUS
Each tool result includes framework-owned subgoal_status for the current goal.
continue_count counts accepted continue decisions at distinct later physical
observation steps; it excludes the decision you are about to make. It is not
the number of policy forwards or tool calls. Re-reading or reclassifying the
same observation does not increase it. start/advance/recover start a new count
at zero. elapsed_actions is the physical action count since this goal started.
Use these values with observed approach, object motion, contact, alignment and
the completion condition to assess whether execution is progressing, making
wrong motions or remaining ineffective. A count alone neither proves blockage
nor triggers recover; there is no continuation-count limit. Explain continued
progress or a concrete reason to change the immediate goal in decision_reason.
"""

_VLA_INSTRUCTION_TEMPLATE = (
    'Pass the current instruction verbatim to {tools}. Choose one immediate, observable outcome '
    'from the task and current scene. Let the policy handle the movements within that outcome. '
    'Keep a suitable instruction stable while its outcome is unfinished or unconfirmed. Assess '
    'changes across observations: approach, improving alignment, contact, holding, transport and '
    'release can all be progress toward the same outcome. Missing contact or object displacement '
    'alone does not establish a blockage. Continue as long as the instruction remains suitable; '
    'there is no continuation-count limit. Advance when observations support the saved completion'
    ' condition. Recover when an observed obstacle, ineffective approach or relevant uncertainty '
    'requires a different immediate instruction; identify what needs to change and why. The '
    'framework schedules observation checkpoints. A returned observation or tool boundary does '
    'not end the Subgoal, establish failure, or require a new instruction.'
)
_LOCAL_OUTCOME_SCOPE = (
    ' This diagnostic uses a local observable outcome rather than the complete multi-stage task. '
    'A placement outcome may include its approach, grasp, transport and release; these movements '
    'need not become separate Subgoals.'
)
VLA_OBSERVATION_CHECKPOINTS = (
    'Execution returns fresh observations at framework-managed checkpoints. Review the scene, '
    'then continue the same Subgoal or select a transition based on its completion condition and '
    'observed obstacles. The checkpoint schedules observation only; it does not impose a Subgoal '
    'duration or retry count.'
)
CODES_OBSERVATION_CHECKPOINTS = VLA_OBSERVATION_CHECKPOINTS
_SCENE_RECOVERY_TEMPLATE = """
Work through recoverable failures within this continuous episode.

1. Establish what actually happened.
Use wrist and external views to identify the object approached, contacted or held,
and the visible support, obstruction and clearance relationships. Record this in
observation_evidence. Confirm a grasp through the intended object's motion with
the gripper and support/contact evidence. Treat unresolved identity or occlusion
as uncertainty and obtain the observation needed for the next action.

2. Turn an unmet goal into a concrete recovery.
Keep a suitable instruction stable while its outcome is unfinished or unconfirmed.
Compare the approach, alignment, contact and object relationships across observations.
Use this scene evidence to judge whether the current approach remains suitable;
observation count and missing contact alone are not reasons to replace the goal.
When the scene reveals a wrong object, lost grasp or blocking prerequisite, choose
recover using that evidence. State in decision_reason the blocking condition, the
change your next action is intended to make, and the observable result that will
test it. Select a feasible change to target identification, approach geometry,
clearance, support, gripper alignment or executor as the current scene warrants.
Execute that bounded recovery, inspect its result, and update your explanation.
Preserve achieved relationships and return to the original requirement once its
prerequisite is satisfied. A different strategy means changing the relevant
physical condition or correcting the grounded target, rather than just renaming
the same ineffective attempt.

3. Use completion feedback to direct the remaining work.
When official feedback is available, {success_field}=false means the overall goal remains
unmet. It is expected during intermediate Subgoals and says nothing by itself
about whether the scene is recoverable. Use {progress_source} and the current images
to select the remaining requirement. If the scene looks complete while {success_field}
is false, inspect {criteria_source} and relevant current geometry, then
correct the unmet condition. Keep official feedback distinct from your own
assessment. If feedback is absent, ground the remaining work in the task and
visible outcomes.

4. Prefer a feasible next recovery over an early stuck decision.
While a relevant recovery remains feasible within the available runtime budget,
perform it and inspect the outcome. Use observations to distinguish a still-suitable ongoing approach from a concrete
blocker requiring recovery. An observation checkpoint or unmet success flag alone
does not establish a blocker. Reserve finish(status='stuck')
for an evidenced terminal blocker: the relevant reachable recovery strategies
have been tried and failed, or current observations establish that the remaining
goal is unreachable or cannot be executed with the available capabilities.
If further progress depends on an observation that cannot be obtained, identify
that limitation. Before finishing, state the unmet requirement, strategies tried,
their observed outcomes, and why no feasible next recovery remains in
decision_reason and the finish summary. Respect runtime budget/termination
signals; use available actions for purposeful progress rather than unchanged
repetition. With official feedback, call finish(status='success') when the latest
top-level {success_field} is true.
"""
_PURE_WORKFLOW_TEMPLATE = (
    'Read the task and initial images. Select an observable Subgoal and classify start. '
    '{execution} After returned observations, classify continue while the current goal remains '
    'suitable; advance on completion evidence or recover on an observed blocker. The policy '
    'executes {actions} from the accepted instruction. With official feedback, finish success '
    'only after the latest observation reports {success_field}.'
)

def vla_instruction_guidance(tools, *, local_outcome_only=True):
    text = _VLA_INSTRUCTION_TEMPLATE.format(tools=tools)
    return text + (_LOCAL_OUTCOME_SCOPE if local_outcome_only else '')


def scene_evidence_and_recovery(*, success_field='success',
                                progress_source='task_progress',
                                criteria_source='the published criteria'):
    return _SCENE_RECOVERY_TEMPLATE.format(success_field=success_field,
        progress_source=progress_source, criteria_source=criteria_source)


def pure_vla_workflow(*, execution, actions, success_field='success'):
    return _PURE_WORKFLOW_TEMPLATE.format(execution=execution, actions=actions,
                                          success_field=success_field)
