"""Stable language goals and explicit, observation-based termination claims.

These checks validate declared decisions, not whether an image proves a claim.
No environment predicates, ground-truth object poses, or task names are used.
"""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata


DECISIONS = ('start', 'continue', 'advance', 'recover')
PROGRESS = ('progressing', 'complete', 'blocked', 'uncertain')
VISIBILITY = ('visible', 'occluded', 'unknown', 'not_applicable')
MAX_OBJECTS = 8
RECORD_ONLY = 'record_only/v1'
DECISION_REASON_VERSION = 'brief_decision_reason/v1'
DECISION_REASON_SCHEMA = {
    'type': 'string', 'minLength': 1,
    'description': ('One short sentence explaining why the selected mode and next action fit '
                    'the current task progress or obstacle. Give the decision justification; '
                    'keep observed facts in observation_evidence and robot instructions in prompt.'),
}


def decision_reason_required(config):
    return (config or {}).get('decision_reason') == DECISION_REASON_VERSION


def decision_reason_error(value, *, required=False):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        return 'decision_reason must be one short, nonempty decision justification'
    return None


def with_decision_reason(spec, config):
    """Extend only declared new tool contracts; historical schemas stay intact."""
    if decision_reason_required(config):
        spec = deepcopy(spec)
        spec['input_schema']['properties']['decision_reason'] = deepcopy(DECISION_REASON_SCHEMA)
        if spec['name'] == 'finish':
            spec['input_schema']['properties']['decision_reason']['description'] = (
                'One short sentence explaining why ending now with the selected status is appropriate. '
                'Keep the outcome in summary and observed facts in assessment.evidence.')
        spec['input_schema']['required'].append('decision_reason')
    return spec


def records_model_errors(config):
    return (config or {}).get('semantic_error_policy') == RECORD_ONLY


def transition_model_errors(arguments):
    """Audit contradictory model declarations; never infer visual ground truth."""
    errors = []
    decision = arguments.get('decision')
    if decision == 'advance':
        if arguments.get('progress') != 'complete':
            errors.append('advance_without_complete_progress')
        if arguments.get('target_visibility') in ('occluded', 'unknown'):
            errors.append('advance_without_visible_completion')
    if decision == 'recover' and arguments.get('progress') not in ('blocked', 'uncertain'):
        errors.append('recover_without_blocked_or_uncertain_progress')
    observation = arguments.get('object_observation')
    if isinstance(observation, dict) and observation.get('visibility') in ('occluded', 'unknown') and observation.get('location') is not None:
        errors.append('object_location_without_visibility')
    return errors


def model_error_metrics(events):
    """Private metrics for executed decisions, separate from rollout validity."""
    rows, counts, decisions = [], {}, 0
    for event in events:
        if event.get('type') not in ('planner_intervention', 'final_visual_assessment'):
            continue
        decisions += 1
        state = event.get('subgoal_state') or {}
        errors = event.get('model_errors', state.get('model_errors', []))
        if errors:
            rows.append({'event_index': event.get('event_index'),
                         'step': event.get('observation_step'),
                         'decision': event.get('decision', 'finish'),
                         'assessment_subgoal_id': state.get('assessment_subgoal_id'),
                         'codes': errors})
        for code in errors:
            counts[code] = counts.get(code, 0) + 1
    return {'schema': 'stage.planner-model-errors/v1',
            'basis': 'model_declaration_consistency_not_visual_ground_truth',
            'decisions': decisions, 'error_decisions': len(rows),
            'error_count': sum(counts.values()), 'counts': counts, 'events': rows}

ACTION_PROPERTIES = {
    'decision': {'type': 'string', 'enum': list(DECISIONS)},
    'progress': {'type': 'string', 'enum': list(PROGRESS),
                 'description': 'Current observed progress of the active subgoal, before this action.'},
    'target_visibility': {'type': 'string', 'enum': list(VISIBILITY)},
    'observation_evidence': {'type': 'string', 'minLength': 1,
                             'description': 'Public scene evidence from the latest observations.'},
    'prompt': {'type': ['string', 'null'],
               'description': 'Required for start/advance/recover; omit for continue to reuse exact text.'},
    'completion_condition': {'type': ['string', 'null'],
                             'description': 'Observable outcome of the new subgoal; unchanged during continue.'},
    'transition_evidence': {'type': ['string', 'null'],
                           'description': 'Evidence of completion for advance, or a concrete obstacle/uncertainty for recover.'},
    'object_observation': {
        'type': ['object', 'null'], 'additionalProperties': False,
        'properties': {
            'object_name': {'type': 'string', 'minLength': 1},
            'visibility': {'type': 'string', 'enum': ['visible', 'occluded', 'unknown']},
            'location': {'type': ['string', 'null'],
                         'description': 'Observed scene relation, required only when visible; omit otherwise.'},
        },
        'required': ['object_name', 'visibility'],
        'description': 'Optional model-reported object observation, never a ground-truth location.',
    },
}
ACTION_REQUIRED = ['decision', 'progress', 'target_visibility', 'observation_evidence']

FINISH_ASSESSMENT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'reason': {'type': 'string', 'enum': [
            'completed', 'unrecoverable', 'recovery_exhausted', 'budget_exhausted']},
        'progress': {'type': 'string', 'enum': list(PROGRESS)},
        'target_visibility': {'type': 'string', 'enum': list(VISIBILITY)},
        'evidence': {'type': 'string', 'minLength': 1,
                     'description': 'Visible support for completion or the concrete barrier to further execution.'},
        'recovery_evidence': {'type': ['string', 'null'],
                              'description': 'What bounded recovery was tried and what obstacle remained.'},
    },
    'required': ['reason', 'progress', 'target_visibility', 'evidence'],
}


# Keep the original prompt above for saved v1 episode specifications.


# v3 is based on the reviewed Claude Opus 5 candidate. Keep v1/v2 text unchanged
# so saved episode specifications reproduce their original Planner instructions.


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f'{name} must be one of {", ".join(choices)}')
    return value


def _objects(arguments, previous, record_only=False):
    objects = deepcopy((previous or {}).get('objects', {}))
    if not isinstance(objects, dict) or len(objects) > MAX_OBJECTS:
        raise ValueError('previous objects must be a bounded object record')
    observation = arguments.get('object_observation')
    if observation is None:
        return objects
    if not isinstance(observation, dict) or set(observation) - {'object_name', 'visibility', 'location'}:
        raise ValueError('object_observation must contain object_name, visibility and optional location')
    name = _text(observation.get('object_name'), 'object_observation.object_name')
    visibility = _choice(observation.get('visibility'), VISIBILITY[:3], 'object_observation.visibility')
    step = arguments.get('observation_step')
    if type(step) is not int or step < 0:
        raise ValueError('object_observation requires a non-negative observation_step')
    location = observation.get('location')
    record = deepcopy(objects.get(name, {'last_seen_step': None, 'last_seen_location': None}))
    if visibility == 'visible':
        location = _text(location, 'visible object_observation.location')
        record.update(last_seen_step=step, last_seen_location=location)
    elif location is not None:
        if not record_only:
            raise ValueError('occluded/unknown object_observation must omit location; last seen evidence is retained')
        location = _text(location, 'object_observation.location')
    # Retain the raw report without promoting an unseen location to last-seen evidence.
    if record_only:
        record['reported_location'] = location
    record['visibility'] = visibility
    record['last_report_step'] = step
    # Retain the eight most recently reported objects, without changing their
    # last-seen timestamp merely because the planner reports them as unknown.
    objects.pop(name, None)
    objects[name] = record
    while len(objects) > MAX_OBJECTS:
        objects.pop(next(iter(objects)))
    return objects


CONTINUATION_STATUS_PROTOCOL = 'subgoal-continuation-status/v1'


def continuation_status(state, action_count):
    """Framework-owned duration and accepted continuations, before the next decision."""
    if state is None:
        return {'protocol': CONTINUATION_STATUS_PROTOCOL, 'active': False}
    return {
        'protocol': CONTINUATION_STATUS_PROTOCOL, 'active': True,
        'subgoal_id': state['active_subgoal_id'], 'prompt': state['active_prompt'],
        'completion_condition': state['completion_condition'],
        'continue_count': state['continue_count'],
        'subgoal_start_action': state['subgoal_start_action'],
        'current_action': action_count,
        'elapsed_actions': action_count - state['subgoal_start_action'],
        'last_review_action': state['assessment_observation_step'],
    }


def resolve_transition(arguments, previous=None, *, record_only=False, require_decision_reason=False,
                       track_continuation=False):
    """Return a new accepted state or raise ValueError without mutating inputs."""
    if not isinstance(arguments, dict):
        raise ValueError('transition arguments must be an object')
    if set(arguments) - (set(ACTION_PROPERTIES) | {'observation_step', 'decision_reason'}):
        raise ValueError('transition contains unsupported fields; subgoal IDs are assigned by the harness')
    reason = arguments.get('decision_reason')
    if error := decision_reason_error(reason, required=require_decision_reason):
        raise ValueError(error)
    observation_step = arguments.get('observation_step')
    if type(observation_step) is not int or observation_step < 0:
        raise ValueError('observation_step must be a non-negative integer')
    decision = _choice(arguments.get('decision'), DECISIONS, 'decision')
    progress = _choice(arguments.get('progress'), PROGRESS, 'progress')
    visibility = _choice(arguments.get('target_visibility'), VISIBILITY, 'target_visibility')
    evidence = _text(arguments.get('observation_evidence'), 'observation_evidence')
    transition_evidence = arguments.get('transition_evidence')
    if transition_evidence is not None:
        transition_evidence = _text(transition_evidence, 'transition_evidence')
    if previous is None:
        if decision != 'start':
            raise ValueError('the first transition must use decision=start')
        sequence, recovery_count = 0, 0
    else:
        if not isinstance(previous, dict):
            raise ValueError('previous must be an accepted transition state')
        for key in ('active_prompt', 'active_subgoal_id', 'completion_condition'):
            _text(previous.get(key), f'previous.{key}')
        sequence, recovery_count = previous.get('sequence'), previous.get('recovery_count')
        if type(sequence) is not int or sequence < 1 or type(recovery_count) is not int or recovery_count < 0:
            raise ValueError('previous must have valid sequence and recovery_count')
        if decision == 'start':
            raise ValueError('start is only allowed for the first subgoal')

    if decision == 'continue':
        active_prompt = previous['active_prompt']
        condition = previous['completion_condition']
        supplied_prompt = arguments.get('prompt')
        expected_prompt = active_prompt
        if supplied_prompt is not None and supplied_prompt != expected_prompt:
            raise ValueError(f'{decision} cannot change prompt; omit it to use {expected_prompt!r}')
        supplied_condition = arguments.get('completion_condition')
        if supplied_condition is not None and supplied_condition != condition:
            raise ValueError(f'{decision} cannot change completion_condition')
        active_id = previous['active_subgoal_id']
        parent_id = previous.get('parent_subgoal_id')
        prompt = expected_prompt
    else:
        prompt = _text(arguments.get('prompt'), f'{decision}.prompt')
        condition = _text(arguments.get('completion_condition'), f'{decision}.completion_condition')
        if not record_only and decision == 'advance' and progress != 'complete':
            raise ValueError('advance requires progress=complete for the previous subgoal')
        if not record_only and decision == 'advance' and visibility not in ('visible', 'not_applicable'):
            raise ValueError('advance requires visible completion evidence; an occluded or unknown target is not confirmed complete')
        if not record_only and decision == 'recover' and progress not in ('blocked', 'uncertain'):
            raise ValueError('recover requires progress=blocked or uncertain')
        if decision in ('advance', 'recover') and transition_evidence is None:
            raise ValueError(f'{decision} requires transition_evidence from current observations')
        sequence += 1
        active_prompt, active_id = prompt, f'sg{sequence:03d}'
        parent_id = previous['active_subgoal_id'] if decision == 'recover' else None
        recovery_count += int(decision == 'recover')

    duration = {}
    if track_continuation:
        if decision == 'continue':
            # Repeated classification of one physical observation is not another
            # execution attempt. Rejected transitions never reach this point.
            duration = dict(
                continue_count=previous['continue_count'] + int(
                    observation_step > previous['assessment_observation_step']),
                subgoal_start_action=previous['subgoal_start_action'])
        else:
            duration = dict(continue_count=0, subgoal_start_action=observation_step)

    return {
        'decision': decision, 'prompt': prompt, 'active_prompt': active_prompt,
        'subgoal_id': active_id, 'active_subgoal_id': active_id,
        'completion_condition': condition, 'progress': progress,
        'assessment_subgoal_id': previous['active_subgoal_id'] if previous else active_id,
        'assessment_observation_step': observation_step,
        'target_visibility': visibility, 'observation_evidence': evidence,
        **({'decision_reason': reason.strip()} if reason is not None else {}),
        'transition_evidence': transition_evidence, 'sequence': sequence,
        'recovery_count': recovery_count, 'parent_subgoal_id': parent_id,
        'objects': _objects(arguments, previous, record_only),
        **duration,
        **({'model_errors': transition_model_errors(arguments)} if record_only else {}),
    }


def finish_assessment_error(status, assessment, previous, budget_exhausted=False, *, record_only=False):
    """Return None or a protocol error; this does not judge image correctness."""
    try:
        _choice(status, ('success', 'failure', 'stuck'), 'status')
        if not isinstance(assessment, dict):
            raise ValueError('finish requires a structured assessment')
        if set(assessment) - set(FINISH_ASSESSMENT_SCHEMA['properties']):
            raise ValueError('finish assessment contains unsupported fields')
        reason = _choice(assessment.get('reason'),
                         FINISH_ASSESSMENT_SCHEMA['properties']['reason']['enum'], 'assessment.reason')
        progress = _choice(assessment.get('progress'), PROGRESS, 'assessment.progress')
        visibility = _choice(assessment.get('target_visibility'), VISIBILITY, 'assessment.target_visibility')
        _text(assessment.get('evidence'), 'assessment.evidence')
        recovery_evidence = assessment.get('recovery_evidence')
        if recovery_evidence is not None:
            _text(recovery_evidence, 'assessment.recovery_evidence')
        if record_only:
            return None
        if reason == 'budget_exhausted':
            if not budget_exhausted:
                raise ValueError('budget_exhausted requires the actual action limit to be reached')
            if status != 'stuck' or progress != 'uncertain':
                raise ValueError('budget_exhausted requires status=stuck and progress=uncertain')
            return None
        if visibility in ('unknown', 'occluded') or progress == 'uncertain':
            raise ValueError('an uncertain or out-of-view target is not evidence of completion or irrecoverable failure; continue bounded recovery or use budget_exhausted at the limit')
        if status == 'success':
            if reason != 'completed' or progress != 'complete':
                raise ValueError('success requires reason=completed and progress=complete')
            return None
        if reason not in ('unrecoverable', 'recovery_exhausted') or progress != 'blocked':
            raise ValueError('failure/stuck requires a visible blocked condition and reason=unrecoverable or recovery_exhausted')
        if reason == 'recovery_exhausted':
            _text(recovery_evidence, 'assessment.recovery_evidence')
            count = previous.get('recovery_count') if isinstance(previous, dict) else None
            if type(count) is not int or count < 1:
                raise ValueError('recovery_exhausted requires at least one accepted recovery transition')
            _text(previous.get('parent_subgoal_id'), 'current recovery parent_subgoal_id')
    except ValueError as error:
        return str(error)
    return None


def finish_model_errors(status, assessment, previous, budget_exhausted=False):
    """Run only after structural validation; retain semantic failure as data."""
    error = finish_assessment_error(status, assessment, previous, budget_exhausted)
    return ['inconsistent_finish_assessment'] if error else []


def full_task_prompt(prompt: str, task_language: str) -> bool:
    """Detect copied task text, including case, punctuation and wrapper changes.

    Semantic paraphrases are governed by the Planner instruction, not a lexical test.
    """
    def words(value):
        return re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold())
    provided, task = words(prompt), words(task_language)
    return bool(task) and any(provided[i:i + len(task)] == task
                              for i in range(len(provided) - len(task) + 1))
