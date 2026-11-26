"""Opt-in Pi0.5 action-overlap guidance, isolated by episode and accepted goal.

Simulation stays synchronous: no physical steps elapse during inference, so
delay_steps is zero. Preserve native 10-step prediction / 5-step execution.
No image-history or KV-cache changes are made here.
"""
from pathlib import Path
import threading
import time

import numpy as np

from research.common import append_json

PROTOCOL = 'pi05-subgoal-rtc-overlap/v1'


def configuration():
    return dict(protocol=PROTOCOL, enabled=True, delay_steps=0,
                guidance_mode='approx', guidance_clip=5.0,
                prediction_horizon=10, execution_horizon=5,
                reset_on=['start', 'advance', 'recover'],
                preserve_on=['continue'], asynchronous=False)


def request_options(options, *, episode_key, state, action_count, prompt):
    if not state or state['prompt'] != prompt:
        raise ValueError('RTC requires the accepted Subgoal and exact instruction')
    return {**(options or {}), 'rtc': dict(protocol=PROTOCOL,
        episode_key=episode_key, subgoal_id=state['subgoal_id'],
        observation_action=action_count)}


class OverlapPolicy:
    """Keep normalized *model-space* tails, never decoded actuator commands."""
    def __init__(self, model, private, *, context_type=None):
        if context_type is None:
            from rlinf.models.embodiment.openpi.sampling.rtc_guidance import RTCGuidanceContext
            context_type = RTCGuidanceContext
        self.model, self.private, self.context_type = model, Path(private), context_type
        if model.action_horizon != 10 or model.action_chunk != 5 or not model.rtc_enabled:
            raise ValueError('RTC trial requires native horizon 10, execution 5 and enabled guidance')
        self.streams = {}
        self.lock = threading.Lock()
        self.requests = 0

    def predict(self, obs, options):
        import torch
        metadata = (options or {}).get('rtc', {})
        if metadata.get('protocol') != PROTOCOL:
            raise ValueError('RTC service requires explicit per-request RTC identity')
        key, goal = metadata['episode_key'], metadata['subgoal_id']
        action = metadata['observation_action']
        if not isinstance(key, str) or not key or not isinstance(goal, str) or not goal:
            raise ValueError('RTC episode and Subgoal identities must be nonempty')
        if type(action) is not int or action < 0:
            raise ValueError('RTC observation_action must be a physical action count')
        prompts = obs['task_descriptions']
        if len(prompts) != 1:
            raise ValueError('RTC service expects one environment per RPC')
        prompt = str(prompts[0])
        with self.lock:
            previous = self.streams.get(key)
            analytic_handoff = metadata.get('reset_after_analytic') is True
            same_goal = previous is not None and previous['goal'] == goal and not analytic_handoff
            executed = action - previous['action'] if previous is not None else 0
            if previous is not None and executed != 5 and not analytic_handoff:
                raise ValueError(f'RTC expected 5 executed actions, received {executed}')
            if analytic_handoff and previous is not None and executed <= 5:
                raise ValueError('RTC analytic handoff requires actual intervening physical actions')
            if same_goal and previous['prompt'] != prompt:
                raise ValueError('A continued RTC Subgoal cannot change its prompt')
            context = None
            if same_goal:
                context = self.context_type(prev_model_actions=previous['model_actions'],
                                            executed_horizon=executed, delay_steps=0)
            begin = time.perf_counter()
            with torch.no_grad():
                actions, result = self.model.predict_action_batch(
                    obs, mode='eval', rtc_context=context)
            raw = result['model_actions'].detach().clone()
            actions = actions.detach().cpu().numpy().astype(np.float32)
            if (tuple(raw.shape) != (1, 10, 32) or actions.shape != (1, 5, 7)
                    or not bool(torch.isfinite(raw).all()) or not np.isfinite(actions).all()):
                raise ValueError('RTC produced an invalid/nonfinite model or execution chunk')
            overlap = raw.shape[1] - executed if same_goal else 0
            row = dict(protocol=PROTOCOL, request=self.requests + 1, episode_key=key,
                subgoal_id=goal, prompt=prompt, observation_action=action,
                context_reused=same_goal,
                reset_reason='analytic_handoff' if analytic_handoff else (
                    None if same_goal else ('new_subgoal' if previous else 'new_episode')),
                previous_prediction_action=previous['action'] if previous else None,
                executed_horizon=executed, overlap_steps=overlap, delay_steps=0,
                prediction_horizon=10, execution_horizon=5, guidance_mode='approx',
                guidance_clip=float(self.model.rtc_guidance_clip), finite=True,
                model_action_abs_max=float(raw.abs().max()),
                first_env_action=actions[0, 0].tolist(), last_env_action=actions[0, -1].tolist(),
                inference_seconds=time.perf_counter() - begin)
            if same_goal:
                tail = context.get_prev_remaining()
                row['overlap_model_rmse'] = float((raw[:, :overlap] - tail).square().mean().sqrt())
            if previous:
                row['boundary_env_action_l2'] = float(np.linalg.norm(actions[0, 0] - previous['last_action']))
            self.requests += 1
            self.streams[key] = dict(goal=goal, prompt=prompt, action=action,
                                     model_actions=raw, last_action=actions[0, -1].copy())
            append_json(self.private / 'rtc_requests.jsonl', row)
            return actions


def facade_class(native, spec):
    if spec.get('policy_rtc') != configuration():
        raise ValueError('Unrecognized RTC trial configuration')

    class RTCFacade(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Pi0Eval already contains the sampler; enable only this opted-in service.
            self._model.rtc_enabled = True
            self._model.rtc_guidance_mode = 'approx'
            self._model.rtc_guidance_clip = spec['policy_rtc']['guidance_clip']
            self.overlap = OverlapPolicy(self._model, Path(spec['rtc_audit_dir']))

        def get_status(self):
            return {**super().get_status(), 'policy_rtc': configuration(),
                    'prediction_horizon': self._model.action_horizon}

        def predict(self, obs, options=None):
            return self.overlap.predict(obs, options)

    return RTCFacade
