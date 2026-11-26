"""Opt-in Codes execution guidance, adapted from ASPIRE's public diagnostic loop.

This changes model-visible instructions only. Native handlers, validators,
classification, observations and the shared action window are unchanged.
"""
from copy import deepcopy
from dataclasses import replace
import os

NATIVE_PROFILE = 'capability-only/v1'
ASPIRE_PROFILE = 'aspire-inspired-codes/v1'
PROFILES = (NATIVE_PROFILE, ASPIRE_PROFILE)


def validate_profile(spec):
    profile = spec.get('codes_prompt_profile', NATIVE_PROFILE)
    if profile not in PROFILES:
        raise ValueError(f'Unknown Codes prompt profile: {profile}')
    if profile != NATIVE_PROFILE and spec.get('final_variant') != 'C':
        raise ValueError('The ASPIRE-inspired prompt candidate is for Codes-only')
    return profile


def enabled():
    profile = os.environ.get('STAGE_CODES_PROMPT_PROFILE', NATIVE_PROFILE)
    if profile not in PROFILES:
        raise ValueError(f'Unknown Codes prompt profile: {profile}')
    return profile == ASPIRE_PROFILE


CODES_EXECUTION = """
You control the robot by composing the available analytic tools in code. Own the
geometry, ordering and checks of the manipulation; a high-level local instruction
is the goal for your code, not a separate motor policy. Use the actual registered
tool schemas and the current robot state to select a short executable fragment.

Ground each fragment in the current observation. Identify the intended object,
its support, the approach direction and available clearance. Compute metric
targets from permitted calibrated observations. In this RoboCasa interface,
move_to uses world-frame XYZ metres and move_delta adds a world-frame displacement
to the current EEF position. Gripper +1 closes, -1 opens, and 'hold' preserves the
current finger width where supported. Carry with the intended grasp retained.
Use fresh returned evidence to update geometry after object or base motion;
re-localize only when the next target is no longer established by that evidence.

Plan the next action around its prerequisite and observable postcondition. Reach
an accessible pregrasp before contact, align the fingers to a graspable part,
establish contact, verify retention with a bounded lift, then carry, place and
release. Keep transit clear of surrounding surfaces and held-object clearance.
Use intermediate waypoints for long Cartesian moves, respecting the native
0.30 m single-move limit. Use current scene geometry to choose base placement and
approach orientation when arm reach or clearance is insufficient. Tool completion
alone does not prove that the object was grasped or that placement succeeded.

Read both the primitive result and the returned images/state before the next
dependent action. Compare the requested target with the measured EEF change,
final_dist, gripper state, object motion and support/contact relationships, using
the fields actually returned. An ok=false result identifies an unachieved primitive
target; it does not by itself establish an impossible task. An ok=true motion
result establishes only that primitive's reported condition. Distinguish target
localization error, unreachable approach, obstruction, tracking error, lost grasp,
and an unmet task relationship using the available evidence. Do not infer an IK
or collision diagnosis that the current interface does not report.

Repair the implicated part of the fragment. For uncertain geometry, obtain the
missing view or corrected target. For poor clearance or reach, change the grounded
waypoint, orientation or base placement. For lost contact, establish a different
grasp and test retention before transport. For tracking that contradicts the
requested displacement, inspect achieved pose and nearby obstruction before
changing the next action; repeatedly increasing force, duration or displacement
is not a diagnosis. Make the changed physical condition and its expected visible
effect explicit in the existing decision_reason. Preserve successful task
relationships while working on the remaining requirement.

The shared 40-action window is a maximum for the entire fragment, including any
native calibration or composite substeps. A window interruption is partial
execution. Use the returned state to determine which phase actually occurred;
continue from that state after the required observed Subgoal review. Do not
restart a relative move or a whole scripted_grasp merely because it was interrupted.
A motion that returns earlier may already supply the observation needed for the
next review. Keep the existing start/continue/advance/recover semantics and the
required image delivery; never run through multiple review windows in a blind loop.

The runtime task_language supplies the goal. Use permitted observations and the
selected feedback to verify completion. Keep code-execution success, object-level
postconditions and native whole-task success distinct. Resolve recoverable failures
within this episode; finish stuck only with the evidenced blocker and attempted
recoveries required by the shared completion rules.
""".strip()

WORKFLOW = (
    'Read the task and current observations, select an observable local outcome, and classify start. '
    'Ground the next fragment in measured geometry and robot state; compose the available native '
    'analytic calls with a clear prerequisite and postcondition. Execute only to the next returned '
    'motion observation or shared window boundary. Inspect primitive diagnostics and actual state, '
    'then classify continue, advance or recover and construct the next fragment from the achieved '
    'state. Preserve the accepted goal on continue. Verify whole-task completion with the existing '
    'selected feedback and finish rules.'
)


def configure_sections(sections, *, simulator):
    if not enabled():
        return sections
    if simulator != 'robocasa':
        raise ValueError('ASPIRE-inspired candidate currently targets RoboCasa only')
    result = dict(sections)
    result['Codes_Execution'] = CODES_EXECUTION
    result['Workflow'] = WORKFLOW
    return result


def configure_tool(declaration):
    if not enabled():
        return declaration
    if declaration.name == 'scripted_grasp':
        description = declaration.description.replace(
            'A fallback when the VLA closed-loop grasp is unavailable. For hard objects prefer vla_arm.',
            'Inspect the returned stage and images to determine which phases actually executed. '
            'The shared action window may interrupt this sequence; it does not resume automatically.')
        return replace(declaration, description=description)
    if declaration.name == 'rotate_pitch':
        description = (
            'Apply an incremental rotation about the arm controller X-axis over n physical steps. '
            'target_pitch is the signed total increment in radians, clamped to [-1.5, 1.5]; '
            'it is not an absolute world-frame pitch target. Repeated calls accumulate rotation. '
            'The command requests no translation; inspect the measured EEF pose and images for '
            'actual displacement or contact before the next dependent motion. '
            'gripper is the held command (+1 close, -1 open; default +1).'
        )
        schema = deepcopy(declaration._input_schema)
        schema['properties']['target_pitch']['description'] = (
            'Signed rotation increment about the controller X-axis, radians, clamped to +/-1.5; '
            'not an absolute world pitch target (default 0.6).')
        return replace(declaration, description=description, _input_schema=schema)
    return declaration
