"""Measured sensor history owned by accepted goals, shared by XR-1 and RLDX."""
from collections import deque

PROTOCOL = 'explicit-subgoal-transitions/v1'


class TransitionOwnedHistory:
    transition_owned_history = True

    def accept_subgoal(self, decision, prompt):
        self._load()
        reset = decision in {'start', 'advance', 'recover'}
        if reset:
            self._seed_hist(prompt)
            self._history_generation = getattr(self, '_history_generation', 0) + 1
            self._vla_ran_in_stage = False
            self._analytic_since_seed = False
            self._analytic_frames = deque([False] * len(self._hist), maxlen=self._hist.maxlen)
            self._stage_reseed_count = 0
        elif decision != 'continue':
            raise ValueError(f'Unsupported history transition: {decision}')
        elif not self._hist or prompt != self._last_prompt:
            raise ValueError('continue requires the existing history and accepted prompt')
        self._last_prompt = prompt
        return {'protocol': PROTOCOL, 'reset': reset, 'decision': decision,
                'generation': self._history_generation}

    def record_analytic_step(self):
        if self._hist and self._last_prompt is not None:
            if self._vla_ran_in_stage:
                super()._record_frame(self._last_prompt)
                self._analytic_frames.append(True)
            else:
                self._analytic_since_seed = True

    def _record_frame(self, prompt):
        super()._record_frame(prompt)
        self._analytic_frames.append(False)

    def begin_vla_call(self):
        reseeded = not self._vla_ran_in_stage and self._analytic_since_seed
        if reseeded:
            self._seed_hist(self._last_prompt)
            self._analytic_frames = deque([False] * len(self._hist), maxlen=self._hist.maxlen)
            self._analytic_since_seed = False
            self._stage_reseed_count += 1
        self._last_history_event = dict(
            history_mode=('stage_continuous' if self._vla_ran_in_stage else
                          'stage_first_call_reseed' if reseeded else 'stage_first_call'),
            analytic_frames_in_history=sum(self._analytic_frames),
            history_generation=self._history_generation,
            stage_reseed_count=self._stage_reseed_count)
        return reseeded

    def run(self, prompt, *args, **kwargs):
        if kwargs.get('force_reset'):
            raise ValueError('History resets belong to start/advance/recover')
        if not self._hist or prompt != self._last_prompt:
            raise ValueError('VLA prompt must equal the accepted Subgoal instruction')
        # The native string-based reset predicate is now always false. Native
        # encoding, observation shapes, prediction and execution remain intact.
        result = super().run(prompt, *args, **kwargs)
        if result.get('steps_applied', 0) > 0:
            self._vla_ran_in_stage = True
        return result


def skill_class(policy, protocol):
    from robots.robocasa.vla_skill import VLASkill
    from research.xr1_skill import XR1Skill
    from research.vla_transition_skill import VLATransitionSkill
    if protocol == PROTOCOL:
        selected = {'xr1': XR1Skill, 'rldx': VLATransitionSkill}.get(policy)
        if selected is None or not getattr(selected, 'transition_owned_history', False):
            raise ValueError(f'{policy} does not implement {PROTOCOL}')
        return selected
    if protocol is not None:
        raise ValueError(f'Unsupported policy history protocol: {protocol}')
    return XR1Skill if policy == 'xr1' else VLASkill
