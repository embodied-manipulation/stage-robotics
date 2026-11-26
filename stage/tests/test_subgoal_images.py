import base64
import json
from pathlib import Path
import tempfile
import unittest

from rpent.tools import ToolResult
from research import subgoal_images as visual
from research import review_window as review
from research.planner_transport import RequestBoundary


def image_count(value):
    if isinstance(value, dict):
        return int(value.get('type') == 'input_image') + sum(image_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(image_count(v) for v in value)
    return 0


class SubgoalImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.public = self.root / 'public'
        self.private = self.root / 'private'
        self.private.mkdir()
        self.window = visual.SubgoalImages(visual.configuration(4), self.private, self.public)

    def tearDown(self):
        visual._window = None
        review._window = None
        self.tmp.cleanup()

    def result(self, action):
        descriptors, images = [], []
        for camera in self.window.cameras:
            name = camera + '.png'
            pixels = f'PNG-{camera}-{action}'.encode()
            path = self.public / name / f'{action:02d}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pixels)
            descriptors.append(dict(camera=camera, artifact=name))
            images.append(pixels)
        return ToolResult(data=dict(step=action, images=descriptors), images=images)

    def observe(self, action):
        result = self.result(action)
        self.window.observe(result, action)
        self.assertEqual(result.images, [])

    def test_anchor_survives_eviction_and_deduplicates(self):
        self.observe(0)
        self.window.transition('start', {'subgoal_id': 'sg001'}, 0)
        _, first = self.window.rewrite({'input': []})
        self.assertEqual(first['retained_image_count'], 3)
        for action in (40, 80, 120, 160, 200):
            self.observe(action)
        out, info = self.window.rewrite({'input': []})
        self.assertEqual(info['retained_steps'], [0, 80, 120, 160, 200])
        self.assertEqual(image_count(out), 15)
        urls = [p['image_url'] for p in out['input'][-1]['content'] if p['type'] == 'input_image']
        self.assertEqual(base64.b64decode(urls[0].split(',')[1]), b'PNG-agentview-0')
        self.assertEqual(base64.b64decode(urls[-1].split(',')[1]), b'PNG-wrist-200')

    def test_continue_transition_and_historical_read(self):
        self.observe(0)
        self.window.transition('start', {'subgoal_id': 'sg001'}, 0)
        self.observe(40)
        self.window.transition('continue', {'subgoal_id': 'sg001'}, 40)
        self.assertEqual(self.window.anchor['action'], 0)
        self.window.observe(self.result(0), 40)
        self.assertEqual([f['action'] for f in self.window.frames], [0, 40])
        for decision, action in [('advance', 40), ('recover', 80)]:
            self.observe(action)
            self.window.transition(decision, {'subgoal_id': decision}, action)
            _, info = self.window.rewrite({'input': []})
            self.assertEqual(info['retained_steps'], [action])
            self.assertEqual(info['anchor_step'], action)

    def test_preserve_text_reasoning_and_tool_pair_without_pixel_text(self):
        self.observe(0)
        old = {'input': [{'type': 'function_call', 'call_id': 'abc', 'arguments': '{}'},
                        {'type': 'function_call_output', 'call_id': 'abc', 'output': [
                            {'type': 'input_text', 'text': json.dumps({'content': [
                                {'type': 'text', 'text': 'Keep this evidence'},
                                {'type': 'image', 'data': 'encoded-pixels'}]})},
                            {'type': 'input_image', 'image_url': 'old-pixels'}]},
                        {'type': 'reasoning', 'encrypted_content': 'opaque-reasoning'}]}
        out, _ = self.window.rewrite(old)
        self.assertEqual(out['input'][0], old['input'][0])
        self.assertEqual(out['input'][2], old['input'][2])
        self.assertEqual(out['input'][1]['call_id'], 'abc')
        self.assertIn('Keep this evidence', json.dumps(out))
        self.assertNotIn('encoded-pixels', json.dumps(out))
        self.assertNotIn('old-pixels', json.dumps(out))
        self.assertEqual(image_count(out), 3)

    def test_actual_request_boundary_unlocks_review_with_injected_images(self):
        visual._window = self.window
        review._window = review.ReviewWindow()
        result = review._window.observe(self.result(0), 'view_env_state')
        self.assertFalse(result.images)
        boundary = RequestBoundary(self.private / 'model_inputs', enforce=False, expected_effort='high')
        payload = {'model': 'gpt-6-astra', 'reasoning': {'effort': 'high'},
                   'input': [{'type': 'message', 'role': 'user', 'content': [
                       {'type': 'input_text', 'text': json.dumps(result.data)}]}]}
        out, meta = boundary(payload)
        self.assertEqual(review._window.last_grant, 0)
        self.assertEqual(image_count(out), 3)
        self.assertEqual(meta['retained_steps'], [0])
        saved = json.loads((self.private / 'model_inputs/request_0001.json').read_text())
        self.assertEqual(image_count(saved), 3)
        # A stale historical read cannot grant a later physical window.
        review._window.total = 40
        review._window.used = 40
        review._window.observe(self.result(40), 'view_env_state')
        boundary({**payload, 'input': [{'type': 'input_text', 'text': review._window.token}]})
        review._window.total = 80
        review._window.used = 40
        stale = review._window.observe(self.result(0), 'view_env_state')
        boundary({**payload, 'input': [{'type': 'input_text', 'text': json.dumps(stale.data)}]})
        self.assertEqual(review._window.last_grant, 40)
        self.assertEqual(review._window.used, 40)

    def test_compaction_does_not_gain_images_or_tools(self):
        visual._window = self.window
        self.observe(0)
        boundary = RequestBoundary(self.private / 'model_inputs', enforce=True, expected_effort='high')
        out, meta = boundary({'model': 'gpt-6-astra', 'reasoning': {'effort': 'high'}, 'input': [
            {'type': 'message', 'role': 'user', 'content': [
                {'type': 'input_text', 'text': 'You are performing a CONTEXT CHECKPOINT COMPACTION.'}]}]})
        self.assertTrue(meta['tool_free_compaction'])
        self.assertEqual(image_count(out), 0)

    def test_checkpoint_restores_anchor_and_window(self):
        self.observe(0)
        self.window.transition('start', {'subgoal_id': 'sg001'}, 0)
        for step in (40, 80, 120, 160, 200):
            self.observe(step)
        restored = visual.SubgoalImages(self.window.config, self.private, self.public)
        restored.restore(self.root)
        self.assertEqual(restored.rewrite({'input': []}), self.window.rewrite({'input': []}))

    def test_configurable_size_and_incomplete_camera_set(self):
        with self.assertRaises(ValueError):
            visual.configuration(0)
        self.window = visual.SubgoalImages(visual.configuration(2), self.private, self.public)
        self.observe(0)
        self.window.transition('start', {'subgoal_id': 'sg001'}, 0)
        for step in (40, 80, 120):
            self.observe(step)
        out, meta = self.window.rewrite({'input': []})
        self.assertEqual(meta['retained_steps'], [0, 80, 120])
        self.assertEqual(image_count(out), 9)
        result = self.result(160)
        result.data['images'].pop()
        with self.assertRaises(ValueError):
            self.window.observe(result, 160)

    def test_libero_two_view_anchor_is_delivered_at_actual_request_boundary(self):
        config = visual.configuration(4, cameras=visual.LIBERO_CAMERAS)
        visual.configure({'planner_observation_profile': 'calibration-two-view/v1',
                          'planner_image_window': config}, self.private, self.public)
        self.window = visual._window
        review._window = review.ReviewWindow()
        first = review._window.observe(self.result(0), 'view_env_state')
        boundary = RequestBoundary(self.private / 'model_inputs', enforce=False)
        def deliver(result):
            return boundary({'model': 'gpt-6-astra', 'reasoning': {'effort': 'low'},
                             'input': [{'type': 'input_text', 'text': json.dumps(result.data)}]})
        deliver(first)
        self.window.transition('start', {'subgoal_id': 'sg001'}, 0)
        for action in (40, 80, 120, 160, 200):
            review._window.total = action
            review._window.used = 40
            result = review._window.observe(self.result(action), 'pi0_doubled')
            out, metadata = deliver(result)
        self.assertEqual(metadata['retained_steps'], [0, 80, 120, 160, 200])
        self.assertEqual(metadata['retained_image_count'], 10)
        self.assertEqual(image_count(out), 10)
        self.assertEqual(review._window.last_grant, 200)
        self.assertNotIn('navview', json.dumps(out))
        self.assertIn('at most 10 images', visual.guidance())
        self.window.transition('recover', {'subgoal_id': 'sg002'}, 200)
        self.assertEqual(self.window.rewrite({'input': []})[1]['retained_steps'], [200])
        with self.assertRaises(ValueError):
            visual.configure({'planner_observation_profile': 'native-three-view/v1',
                              'planner_image_window': config}, self.private, self.public)


if __name__ == '__main__':
    unittest.main()
