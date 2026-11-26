"""Actual XR1 history/skill and toolkit dispatch with a measured-frame CPU fixture."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_review_window import Primitives, TestToolkit
from rpent.tools import ToolResult, tool
from robots.robocasa.primitives import RoboCasaPrimitives, _physical_action
from research.xr1_skill import XR1Skill
from research.xr1_server import XR1Facade, action_fields
from research.subgoal_review import install
import research.review_window as review


class SensorEnvironment:
    def __init__(self):
        self.actions = []

    @property
    def current_raw_obs(self):
        t = len(self.actions)
        return {'language': 'Arrange tea and prepare coffee.',
                'robot0_base_pos': np.zeros(3), 'robot0_base_quat': np.array([0, 0, 0, 1]),
                'robot0_base_to_eef_pos': np.full(3, t),
                'robot0_base_to_eef_quat': np.array([0, 0, 0, 1]),
                'robot0_gripper_qpos': np.array([.02, -.02])}

    @property
    def eef_pos(self):
        return np.array([0, 0, len(self.actions) * .001])

    @property
    def gripper_qpos(self):
        return self.current_raw_obs['robot0_gripper_qpos']

    def render_camera(self, *args, **kwargs):
        return np.full((2, 2, 3), len(self.actions) % 256, dtype=np.uint8)

    def step(self, action):
        self.actions.append(action)


class Policy:
    def __init__(self):
        self.inputs = []

    def get_modality_config(self):
        return XR1Facade.get_modality_config(self)

    def predict(self, observation, options):
        self.inputs.append(copy.deepcopy(observation))
        return action_fields(np.zeros((16, 12)))


class MeasuredPrimitives(Primitives):
    def __init__(self):
        self.env = SensorEnvironment()
        self.policy = Policy()
        self._vla_skill = XR1Skill(self.env, self.policy)
        self._vla_skill.restrictions = SimpleNamespace(disable_privileged_grasp=True,
                                                 disable_success_stop=True)
        self._vla_skill._build_env_action = lambda *args, **kw: np.zeros(12)
        self._recording = False
        self.record_frame = lambda: None
        self._vla_desync = True

    @tool
    @_physical_action
    def move_to(self, steps: int = 3) -> ToolResult:
        """Measured analytic motion fixture."""
        self._vla_desync = True
        for _ in range(steps):
            self.env.step(np.zeros(12))
        return ToolResult(data={'steps': steps})

    @tool
    @_physical_action
    def vla_skill(self, prompt: str, max_chunks: int = 5,
                   force_reset: bool = False) -> ToolResult:
        """Run the actual native skill, without a learned model or simulator."""
        result = RoboCasaPrimitives.run_vla_skill(
            self, None, max_chunks, True, prompt, force_reset, 8, 999, .012)
        return ToolResult(data=result)


class TransitionHistoryTests(unittest.TestCase):
    prompt = 'Place the white cup on the tray.'

    def setUp(self):
        self.flags = patch.dict(os.environ, {'STAGE_VLA_PROMPT_MODE': 'provided',
            'STAGE_VLA_INSTRUCTION_SCOPE': 'subgoal_only', 'VLA_MAX_CHUNKS': '5',
            'VLA_ACTION_STEPS_PER_CHUNK': '8', 'VLA_SETTLE_PATIENCE': '999'})
        self.flags.start()
        review._window = None
        self.tmp = tempfile.TemporaryDirectory()
        from rpent.utils.logging import init_output_dir
        init_output_dir(Path(self.tmp.name))
        self.tk = TestToolkit()
        self.tk._tools.clear()
        self.p = self.tk._primitives = MeasuredPrimitives()
        for name in ('move_to', 'vla_skill', 'finish'):
            self.tk.add_tool(getattr(self.p, name))
        install(self.tk, self.tmp.name, code_as_policy=True)
        review.install(self.tk, 40, self.tmp.name)
        self.see()
        self.classify('start')

    def tearDown(self):
        self.tk.close()
        review._window = None
        self.flags.stop()
        self.tmp.cleanup()

    def see(self):
        result = review._window.observe(ToolResult(images=[b'image']), 'view_env_state')
        review.on_planner_request({'input': [{'type': 'input_text', 'text': json.dumps(result.data)},
                                            {'type': 'input_image', 'image_url': 'test'}]})

    def classify(self, decision):
        args = dict(observation_step=review._window.total, decision=decision,
                    progress='complete' if decision == 'advance' else 'progressing',
                    target_visibility='visible', observation_evidence='Cup is visible.',
                    decision_reason='Pursue the local cup placement.')
        if decision != 'continue':
            args.update(prompt=self.prompt, completion_condition='Cup rests on tray.')
        if decision in ('advance', 'recover'):
            args['transition_evidence'] = 'The observed state requires this local goal.'
        result = self.tk.execute_tool('subgoal_review', args)
        self.assertFalse(result.is_error, result.data)
        return result.data['vla_history']

    def vla(self, **kwargs):
        result = self.tk.execute_tool('vla_skill', {'prompt': self.prompt, **kwargs})
        self.assertFalse(result.is_error, result.data)
        return result

    def times(self):
        obs = self.p.policy.inputs[-1]
        video = obs['video.robot0_agentview_left'][0, :, 0, 0, 0].tolist()
        state = obs['state.end_effector_position_relative'][0, :, 0].tolist()
        self.assertEqual(video, state)
        return video

    def test_five_chunks_and_continue_keep_measured_history(self):
        self.vla()
        self.assertEqual(len(self.p.env.actions), 40)
        self.assertEqual(len(self.p.policy.inputs), 5)
        self.see()
        self.assertFalse(self.classify('continue')['reset'])
        self.vla(max_chunks=1)
        self.assertEqual(self.times(), [34, 36, 38, 40])
        self.assertEqual(self.p._vla_skill._history_generation, 1)

    def test_analytic_frames_are_recorded_without_continue_reseed(self):
        self.vla(max_chunks=1)
        self.tk.execute_tool('move_to', {'steps': 3})
        self.see()
        self.assertFalse(self.classify('continue')['reset'])
        # Even a legacy desync flag cannot override the explicit transition.
        self.p._vla_desync = True
        self.vla(max_chunks=1)
        self.assertEqual(self.times(), [5, 7, 9, 11])
        self.assertEqual(self.p._vla_skill._history_generation, 1)
        self.assertEqual(self.p._vla_skill._last_history_event['history_mode'], 'stage_continuous')

    def test_codes_before_first_vla_in_stage_reseeds(self):
        self.tk.execute_tool('move_to', {'steps': 3})
        self.see(); self.classify('continue'); self.vla(max_chunks=1)
        self.assertEqual(self.times(), [3, 3, 3, 3])
        self.assertEqual(self.p._vla_skill._last_history_event['history_mode'], 'stage_first_call_reseed')
        self.assertEqual(self.p._vla_skill._history_generation, 1)

    def test_advance_then_codes_then_vla_reseeds(self):
        self.vla(max_chunks=1); self.see(); self.classify('advance')
        self.tk.execute_tool('move_to', {'steps':3})
        self.see(); self.classify('continue'); self.vla(max_chunks=1)
        self.assertEqual(self.times(), [11,11,11,11])
        self.assertEqual(self.p._vla_skill._history_generation, 2)

    def test_zero_step_vla_does_not_mark_stage(self):
        self.p.env.action_budget_exhausted = True
        self.vla(max_chunks=1)
        self.assertFalse(self.p._vla_skill._vla_ran_in_stage)
        self.p.env.action_budget_exhausted = False
        self.tk.execute_tool('move_to', {'steps':3})
        self.see(); self.classify('continue'); self.vla(max_chunks=1)
        self.assertEqual(self.times(), [3,3,3,3])

    def test_same_text_advance_recover_reset_once_and_force_reset_is_rejected(self):
        self.vla(max_chunks=1)
        self.see()
        self.assertEqual(self.classify('advance')['generation'], 2)
        self.assertFalse(self.classify('continue')['reset'])
        self.vla(max_chunks=1)
        self.assertEqual(self.times(), [8, 8, 8, 8])
        self.see()
        self.assertEqual(self.classify('recover')['generation'], 3)
        self.vla(max_chunks=1)
        self.assertEqual(self.times(), [16, 16, 16, 16])
        result = self.tk.execute_tool('vla_skill', {'prompt': self.prompt, 'force_reset': True})
        self.assertTrue(result.is_error)
        self.assertEqual(len(self.p.env.actions), 24)
        self.assertEqual(self.p._vla_skill._history_generation, 3)


if __name__ == '__main__':
    unittest.main()
