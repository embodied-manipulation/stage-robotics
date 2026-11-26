import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from stage_bridge.libero_rtc import OverlapPolicy, request_options


class Context:
    def __init__(self, prev_model_actions, executed_horizon, delay_steps):
        self.prev_model_actions = prev_model_actions
        self.executed_horizon, self.delay_steps = executed_horizon, delay_steps

    def get_prev_remaining(self):
        return self.prev_model_actions[:, self.executed_horizon:]


class Model:
    action_horizon, action_chunk = 10, 5
    rtc_enabled, rtc_guidance_clip = True, 5.0

    def __init__(self):
        self.contexts = []

    def predict_action_batch(self, obs, mode, rtc_context):
        self.contexts.append(rtc_context)
        raw = torch.arange(10 * 32, dtype=torch.float32).reshape(1, 10, 32)
        raw = raw + len(self.contexts) * 2000
        return torch.ones((1, 5, 7)), {'model_actions': raw}


class RTCTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Model()
        self.policy = OverlapPolicy(self.model, self.directory.name, context_type=Context)

    def predict(self, action, key='episode-a', goal='sg001', prompt='close the drawer'):
        options = request_options({'mode': 'eval'}, episode_key=key,
            state=dict(subgoal_id=goal, prompt=prompt), action_count=action, prompt=prompt)
        return self.policy.predict({'task_descriptions': [prompt]}, options)

    def test_continue_across_planner_boundary_uses_shifted_model_tail(self):
        for action in range(0, 46, 5):
            self.assertEqual(self.predict(action).shape, (1, 5, 7))
        self.assertIsNone(self.model.contexts[0])
        for context in self.model.contexts[1:]:
            self.assertEqual(context.get_prev_remaining().shape, (1, 5, 32))
            self.assertEqual(context.delay_steps, 0)
        # The call at 40 is the next Planner interval; retain the previous call at 35.
        self.assertEqual(float(self.model.contexts[8].get_prev_remaining()[0, 0, 0]), 16160)
        rows = [json.loads(s) for s in (Path(self.directory.name)/'rtc_requests.jsonl').read_text().splitlines()]
        self.assertTrue(rows[8]['context_reused'])
        self.assertEqual(rows[8]['previous_prediction_action'], 35)

    def test_two_lanes_never_exchange_action_tails(self):
        self.predict(0, key='a')
        self.predict(0, key='b')
        self.predict(5, key='a')
        self.predict(5, key='b')
        self.assertEqual(float(self.model.contexts[2].get_prev_remaining()[0, 0, 0]), 2160)
        self.assertEqual(float(self.model.contexts[3].get_prev_remaining()[0, 0, 0]), 4160)

    def test_new_goal_clears_tail_even_if_wording_is_unchanged(self):
        self.predict(0)
        self.predict(5)
        self.predict(10, goal='sg002')
        self.assertIsNone(self.model.contexts[-1])
        self.predict(15, goal='sg002')
        self.assertIsNotNone(self.model.contexts[-1])

    def test_four_interleaved_episodes_keep_their_own_tails(self):
        for key in ('a', 'b', 'c', 'd'):
            self.predict(0, key=key)
        for key in ('d', 'b', 'a', 'c'):
            self.predict(5, key=key)
        self.assertEqual([float(c.get_prev_remaining()[0, 0, 0])
                          for c in self.model.contexts[4:]], [8160, 4160, 2160, 6160])

    def test_repeated_or_out_of_sync_observation_is_rejected(self):
        self.predict(0)
        for action in [0, 3, 10]:
            with self.assertRaisesRegex(ValueError, '5 executed actions'):
                self.predict(action)
        self.assertEqual(len(self.model.contexts), 1)

    def test_prompt_change_requires_new_goal(self):
        self.predict(0)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            self.predict(5, prompt='put the bottle on the table')
        self.predict(5, goal='sg002', prompt='put the bottle on the table')
        self.assertIsNone(self.model.contexts[-1])

    def test_unaccepted_prompt_is_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'accepted Subgoal'):
            request_options({}, episode_key='a', state=None, action_count=0, prompt='x')

    def test_exhausted_prediction_cannot_silently_disable_guidance(self):
        self.model.action_horizon = 5
        with self.assertRaisesRegex(ValueError, 'horizon 10'):
            OverlapPolicy(self.model, self.directory.name, context_type=Context)


if __name__ == '__main__':
    unittest.main()
