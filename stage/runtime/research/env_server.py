"""Matched RoboCasa reset, action budget, and evaluator artifacts outside Planner IO."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import time
import sys
from types import SimpleNamespace

import numpy as np

from robots.robocasa.env_server import RoboCasaEnvFacade
from research.common import append_json, read_json, write_json


ROBOT_OBSERVATION_FIELDS = frozenset({
    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_base_pos', 'robot0_base_quat',
    'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos',
    'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos',
    'robot0_joint_pos_sin', 'robot0_joint_vel',
})
# The four world-frame poses above require localization; the allowlist alone
# does not establish a real-robot sensor implementation. See the handoff guide.


class ResearchEnvFacade(RoboCasaEnvFacade):
    """Add measurement around the upstream simulator and native action mapping."""

    def __init__(self, spec: dict):
        from research import custom_ab_tasks  # register recovered tasks before native make()
        self.spec = spec
        if spec.get('feedback_sampling', 'native') not in {'native', 'step_cached'}:
            raise ValueError('feedback_sampling must be native or step_cached')
        profile_path = Path(spec['output_dir']) / 'runtime_profile.json'
        runtime_profile = None
        self._preserve_reset_contexts = os.environ.get('STAGE_PRESERVE_RESET_CONTEXTS') == '1'
        if profile_path.exists():
            runtime_profile = read_json(profile_path)
            if (not isinstance(runtime_profile, dict)
                    or set(runtime_profile) != {'fix_id', 'preserve_reset_contexts'}
                    or runtime_profile['fix_id'] != 'preserve_reset_contexts'
                    or runtime_profile['preserve_reset_contexts'] is not True):
                raise ValueError(f'Unsupported episode runtime profile: {profile_path}')
            self._preserve_reset_contexts = True
        try:
            from robots.robocasa.restrictions import get_restrictions
            self.restrictions = get_restrictions()
        except ModuleNotFoundError:
            # Unmodified upstream control source has no restriction extension.
            self.restrictions = SimpleNamespace(hide_task_feedback=False, disable_success_stop=False, disable_privileged_grasp=False)
        self.final_predicates = {}
        self.zero_command_steps = 0
        self.private = Path(spec['output_dir']) / 'private'
        self.private.mkdir(parents=True, exist_ok=True)
        self.action_count = 0
        self.sources = Counter()
        self.source = 'unattributed'
        self.source_tool = 'initialization'
        self.official_any = False
        self.official_terminal = False
        self.first_success_step = None
        self._episode_ready = False
        self._states = []
        self._actions = []
        self._closed = False
        self._retired_simulators = []
        self._retired_renderers = []
        # Match the existing 340 scene protocol, independently of policy choice.
        super().__init__(task_name=spec['task'], split=spec['split'], seed=7)
        self._retain_reset_renderer()
        self.env.reset()
        self.env.sim.model.opt.noslip_iterations = 1
        self.env.rng = np.random.default_rng(spec['seed'])
        self._retain_reset_renderer()
        self._observation = self.env.reset()
        self.env.sim.model.opt.noslip_iterations = 1
        self.seed = spec['seed']
        self._meta['seed'] = spec['seed']
        self._episode_ready = True
        if self._cached_feedback:
            self._initialize_cached_feedback()
        self._save_initial()
        self._save_metrics()
        if spec.get('resume_from'):
            from research.episode_resume import restore_environment
            restore_environment(self)
        if self._preserve_reset_contexts:
            write_json(self.private / 'renderer_lifetime.json', {
                'version': 'retain-reset-renderers/v1',
                'preserve_reset_contexts': self._preserve_reset_contexts,
                'runtime_profile_path': str(profile_path) if runtime_profile is not None else None,
                'runtime_profile': runtime_profile,
                'retained_reset_simulators': len(self._retired_simulators),
                'retained_reset_renderers': len(self._retired_renderers),
                'release_boundary': 'one-episode environment process exit',
                'rng_state_or_action_changes': False,
            })

    def _retain_reset_renderer(self):
        if self._preserve_reset_contexts:
            # Old cyclic renderers can otherwise be collected while the new GL
            # context is current. Their native destructor frees framebuffer IDs
            # without switching to its owning context. Match the existing
            # Subgoal lifetime guard and let process exit release these objects.
            sim = self.env.sim
            if not any(retired is sim for retired in self._retired_simulators):
                self._retired_simulators.append(sim)
            # MjSim.free() explicitly deletes this attribute during reset, so
            # retaining the simulator alone does not retain its GL context.
            # Keep a direct reference until this episode process exits.
            renderer = getattr(sim, '_render_context_offscreen', None)
            if renderer is not None and not any(old is renderer for old in self._retired_renderers):
                self._retired_renderers.append(renderer)

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc['env.get_observation'] = self.get_observation
        self._rpc['env.set_action_source'] = self.set_action_source
        self._rpc['env.success_for_stop'] = self.success_for_stop
        self._rpc['research.finalize'] = self.finalize
        self._rpc['research.action_count'] = self.get_action_count
        self._readonly_methods.update({'env.get_observation', 'env.success_for_stop', 'research.action_count'})

    def _sanitize(self, observation):
        if not (self.restrictions.hide_task_feedback
                or getattr(self.restrictions, 'restrict_observation_fields', False)):
            return observation
        return {key: value for key, value in observation.items() if key in ROBOT_OBSERVATION_FIELDS}

    @property
    def _cached_feedback(self):
        return self.spec.get('feedback_sampling', 'native') == 'step_cached'

    @contextmanager
    def _capture_predicate_locals(self):
        predicate_code = self.env._check_success.__func__.__code__
        captured = {}

        def trace(frame, event, arg):
            if event == 'call' and frame.f_code is predicate_code:
                def local(frame, event, arg):
                    if event == 'return':
                        captured.update({key: value.item() if isinstance(value, np.generic) else value
                                         for key, value in frame.f_locals.items()
                                         if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))})
                    return local
                return local
            return None

        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            yield captured
        finally:
            sys.settrace(previous_trace)

    def _initialize_cached_feedback(self):
        # One identical initial evaluation in both arms, before any agent action.
        with self._capture_predicate_locals() as captured:
            success = bool(self.env._check_success())
        self.final_predicates = captured
        self.official_terminal = self.official_any = success
        self.first_success_step = 0 if success else None

    def get_observation(self):
        return self._sanitize(self._observation)

    def reset(self):
        """Native bootstrap calls attach to the already sampled matched episode."""
        if self._episode_ready:
            append_json(self.private / 'bootstrap_attach.jsonl', {'time': time.time(), 'action_count': self.action_count})
            return self.get_observation()
        return super().reset()

    def set_action_source(self, source: str, tool: str):
        if source not in {'vla', 'analytic', 'unattributed'}:
            raise ValueError(f'Unknown action source: {source}')
        self.source, self.source_tool = source, str(tool)
        return {'ok': True}

    def check_success(self):
        if self.restrictions.hide_task_feedback:
            raise PermissionError('task feedback disabled')
        if self._cached_feedback:
            return self.official_terminal
        return super().check_success()

    def success_for_stop(self):
        if self.restrictions.disable_success_stop:
            raise PermissionError('success stopping disabled')
        if self._cached_feedback:
            return self.official_terminal
        return super().check_success()

    def get_success_criteria_text(self):
        if self.restrictions.hide_task_feedback:
            raise PermissionError('task feedback disabled')
        return super().get_success_criteria_text()

    def get_task_progress(self):
        if self.restrictions.hide_task_feedback:
            raise PermissionError('task feedback disabled')
        if self._cached_feedback:
            return super().get_task_progress(predicate_locals=self.final_predicates)
        return super().get_task_progress()

    def grasp_contact(self):
        if self.restrictions.disable_privileged_grasp:
            raise PermissionError('privileged grasp disabled')
        return super().grasp_contact()

    def get_action_count(self):
        return {'actions': self.action_count, 'budget': self.spec['action_budget'],
                'exhausted': self.action_count >= self.spec['action_budget']}

    def _state(self):
        data = self.env.sim.data
        return np.concatenate((np.asarray(data.qpos).copy(), np.asarray(data.qvel).copy()))

    def _save_initial(self):
        data = self.env.sim.data
        np.savez_compressed(self.private / 'initial_state.npz', qpos=np.asarray(data.qpos),
                            qvel=np.asarray(data.qvel), ctrl=np.asarray(data.ctrl))
        self._states.append(self._state())
        write_json(self.private / 'scene.json', {
            'task': self.task_name, 'seed': self.seed, 'split': self.split,
            'task_language': self.get_task_language(), 'layout_id': self.env.layout_id,
            'style_id': self.env.style_id, 'environment_reset': self.spec['environment_reset'],
            'noslip_iterations': int(self.env.sim.model.opt.noslip_iterations),
            'source_revision': self.spec['source_revision'],
        })
        # Saving the initial visual evidence does not enter the Planner prompt.
        import imageio.v2 as imageio
        for camera in self.cameras:
            image = self.render_camera(camera, 256, 256, False)
            imageio.imwrite(self.private / f'initial_{camera}.png', image[::-1])

    def step(self, flat_action):
        if self.action_count >= self.spec['action_budget']:
            raise RuntimeError('physical action budget exhausted')
        started = time.perf_counter()
        with self._capture_predicate_locals() as captured:
            obs, reward, done, info = super().step(flat_action)
        self.final_predicates = captured
        self._observation = obs
        self.action_count += 1
        self.sources[self.source] += 1
        # Kitchen.reward already calls the official predicate once at each step.
        # Reuse that result so instrumentation adds no predicate invocation.
        success = bool(reward)
        self.official_terminal = success
        self.official_any |= success
        if success and self.first_success_step is None:
            self.first_success_step = self.action_count
        action = np.asarray(flat_action, dtype=np.float64).reshape(-1)
        self._actions.append(action.copy())
        self.zero_command_steps += int(bool(np.all(action == 0)))
        self._states.append(self._state())
        append_json(self.private / 'actions.jsonl', {
            'step': self.action_count, 'source': self.source, 'tool': self.source_tool,
            'action': action.tolist(), 'official_success': success,
            'sim_time': float(self.env.sim.data.time), 'step_seconds': time.perf_counter() - started,
        })
        self._save_metrics()
        exhausted = self.action_count >= self.spec['action_budget']
        if (self.restrictions.hide_task_feedback
                or getattr(self.restrictions, 'restrict_observation_fields', False)):
            return self._sanitize(obs), 0.0, False, {'stage_action_budget_exhausted': exhausted}
        return obs, reward, done, {**info, 'stage_action_budget_exhausted': exhausted}

    def chunk_step(self, flat_actions, *, return_all_frames=False):
        observations, final = [], None
        for action in flat_actions:
            final = self.step(action)
            observations.append(final[0])
            if self.action_count >= self.spec['action_budget']:
                break
        if final is None:
            raise ValueError('empty action chunk')
        return ((observations if return_all_frames else final[0]), *final[1:])

    def _save_metrics(self):
        total = self.action_count
        write_json(self.private / 'environment_metrics.json', {
            'schema': 'stage.action-source/v1', 'actions': total,
            'action_budget': self.spec['action_budget'], 'budget_exhausted': total >= self.spec['action_budget'],
            'action_steps_by_source': dict(self.sources),
            'vla_action_fraction': self.sources['vla'] / total if total else None,
            'official_process_any': self.official_any, 'official_terminal': self.official_terminal,
            'first_success_step': self.first_success_step, 'success_source': 'native Kitchen.reward/_check_success',
            'final_predicate_states': self.final_predicates,
            'zero_command_steps': self.zero_command_steps,
            'physical_success': None, 'physical_metric_status': 'not_evaluated',
        })

    def finalize(self):
        self._save_metrics()
        np.savez_compressed(self.private / 'trajectory.npz',
                            actions=np.asarray(self._actions), states=np.asarray(self._states),
                            nq=self.env.sim.model.nq, nv=self.env.sim.model.nv)
        return {'ok': True, 'actions': self.action_count}

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.finalize()
            finally:
                super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    # Reject failed native RGB reads before corrupt pixels reach either executor.
    # Reuse the existing checked-opengl-readback/v1 guard; valid pixels are unchanged.
    from research.render_readback_validation import install
    install()
    ResearchEnvFacade(read_json(args.episode)).serve(transport='http', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
