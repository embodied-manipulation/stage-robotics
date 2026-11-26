"""Explicit per-episode minimum effort before an unsuccessful Planner finish."""
from rpent.tools import ToolResult
from research.common import append_json

PROTOCOL = 'minimum-vla-actions-before-stuck/v1'


def guidance(minimum, *, code_as_policy=False, vla_enabled=True):
    text = f'''Minimum exploration effort for this episode:
Before finishing unsuccessfully (including stuck), execute at least {minimum}
VLA physical simulation actions in this continuous episode. Count actions, not
Planner calls, model forwards, observations, or Subgoal changes. At 40 actions
per full observation window, {minimum} actions is {minimum / 40:g} full windows.
This is a minimum before giving up, not a success requirement or a stopping
target. Native whole-task success still ends execution immediately; never keep
moving after success to meet the minimum. Respect native truncation and hard
runtime limits, and report infrastructure failures honestly.
Use the additional actions for grounded task execution or recovery. Reinspect
the remaining native goal conditions, preserve achieved relationships, and try
feasible corrections when the current approach is ineffective. Keep productive
instructions stable. Do not pad the count with idle motion, blind repetition,
or unnecessary Subgoal switches. Continue beyond {minimum} if useful progress
or feasible recovery remains. The usual 40-action review cadence still applies.
'''
    if not vla_enabled:
        return text.replace('VLA physical', 'Codes physical')
    return text.replace('VLA physical', 'VLA or Codes physical') if code_as_policy else text


def install(toolkit, private, *, minimum, action_count, code_as_policy=False, vla_enabled=True):
    if type(minimum) is not int or minimum < 0:
        raise ValueError('Minimum VLA actions must be a nonnegative integer')
    if not minimum:
        return
    original = toolkit._tools['finish']
    env = toolkit._primitives.env

    def finish(**inputs):
        actions = action_count()
        success, truncated = bool(env.terminated), bool(env.truncated)
        error = None
        if inputs.get('status') == 'success' and not success:
            error = 'Native whole-task success is required for finish(status="success").'
        elif not success and not truncated and actions < minimum:
            error = (f'Only {actions}/{minimum} required VLA actions have been executed. '
                     'Continue purposeful task execution or grounded recovery before giving up.')
        append_json(private/'minimum_effort_decisions.jsonl', dict(protocol=PROTOCOL,
            counted_executors=['Codes'] if not vla_enabled else ['VLA', 'Codes'] if code_as_policy else ['VLA'],
            action=actions, minimum=minimum, status=inputs.get('status'),
            native_success=success, truncated=truncated, allowed=error is None))
        if error:
            if not vla_enabled:
                error = error.replace('VLA actions', 'Codes actions')
            elif code_as_policy:
                error = error.replace('VLA actions', 'physical executor actions')
            return ToolResult(data=dict(error=error, code='minimum_effort_finish_rejected',
                actions_executed=actions, minimum_vla_actions_before_stuck=minimum,
                remaining_minimum_actions=max(0, minimum-actions)))
        return original(**inputs)

    toolkit._tools['finish'] = original.with_handler(finish)
