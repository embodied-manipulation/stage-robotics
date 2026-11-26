import json
from pathlib import Path
import tempfile
import unittest

from test_review_window import TestToolkit
from rpent.tools import ToolResult
import research.review_window as review
from research.subgoal_review import install


class SubgoalTests(unittest.TestCase):
    def setUp(self):
        review._window = None
        self.tmp = tempfile.TemporaryDirectory()
        from rpent.utils.logging import init_output_dir
        init_output_dir(Path(self.tmp.name))
        self.tk = install(TestToolkit(), Path(self.tmp.name), code_as_policy=True)
        review.install(self.tk, 40, Path(self.tmp.name))

    def tearDown(self):
        self.tk.close()
        review._window = None
        self.tmp.cleanup()

    def see(self):
        result = review._window.observe(ToolResult(images=[b'image']), 'view_env_state')
        review.on_planner_request({'input': [{'type': 'input_text', 'text': json.dumps(result.data)},
                                            {'type': 'input_image', 'image_url': 'test'}]})

    def classify(self, decision='start', **kwargs):
        args = dict(observation_step=review._window.total, decision=decision,
                    progress='progressing', target_visibility='visible',
                    observation_evidence='The white cup is visible.',
                    decision_reason='Move the observed cup to clear the dispenser.')
        if decision in ('start', 'advance', 'recover'):
            args.update(prompt='Place the white cup on the tray.',
                        completion_condition='White cup rests on tray.')
        args.update(kwargs)
        return self.tk.execute_tool('subgoal_review', args)

    def test_model_delivery_then_classification_is_required(self):
        self.assertTrue(self.classify().is_error)
        self.assertTrue(self.tk.execute_tool('move_to', {'steps': 1}).is_error)
        self.assertEqual(review._window.total, 0)
        self.see()
        self.assertFalse(self.classify().is_error)
        self.tk.execute_tool('move_to', {'steps': 100})
        self.assertEqual(review._window.total, 40)
        self.see()
        self.assertTrue(self.tk.execute_tool('move_to', {'steps': 1}).is_error)
        self.assertEqual(review._window.total, 40)
        self.assertFalse(self.classify('continue').is_error)
        self.tk.execute_tool('move_to', {'steps': 1})
        self.assertEqual(review._window.total, 41)

    def test_four_decisions_and_removed_wait(self):
        self.see()
        self.assertFalse(self.classify().is_error)
        state = self.tk.subgoal_review.state
        self.assertFalse(self.classify('continue').is_error)
        self.assertEqual(self.tk.subgoal_review.state['active_prompt'], state['active_prompt'])
        self.assertTrue(self.classify('continue', prompt='Grasp red mug.').is_error)
        self.assertFalse(self.classify('advance', progress='complete', transition_evidence='Cup rests on tray.').is_error)
        self.assertFalse(self.classify('recover', progress='blocked', transition_evidence='Cup slipped.').is_error)
        self.assertTrue(self.classify('wait').is_error)

    def test_prompt_binding_shared_actions_and_final_review(self):
        self.see(); self.classify()
        self.assertTrue(self.tk.execute_tool('vla_skill', {'prompt': 'Wrong object.'}).is_error)
        self.tk.execute_tool('move_to', {'steps': 8})
        self.tk.execute_tool('vla_skill', {'prompt': 'Place the white cup on the tray.', 'max_chunks': 20})
        self.assertEqual(review._window.total, 40)
        self.assertEqual(self.tk._primitives.forwards, 4)
        self.assertTrue(self.tk.execute_tool('finish', {}).data['review_required'])
        self.see(); self.classify('continue')
        self.assertFalse(self.tk.execute_tool('finish', {}).is_error)

    def test_pure_mode_removes_analytic_execution(self):
        from research.subgoal_review import ANALYTIC
        pure = install(TestToolkit(), Path(self.tmp.name), code_as_policy=False)
        self.assertFalse(ANALYTIC & {tool.name for tool in pure.list_tools()})
        self.assertTrue(pure.execute_tool('move_to', {'steps': 1}).is_error)
        self.assertEqual(len(pure._primitives.env.actions), 0)
        pure.close()

    def test_pure_prompt_only_advertises_vla_execution(self):
        import os
        from unittest.mock import patch
        from robots.robocasa.prompt_bundle import system_prompt
        with patch.dict(os.environ, STAGE_CODE_AS_POLICY='0',
                        STAGE_RESTRICTED='1', STAGE_HYBRID_GUIDANCE='0',
                        STAGE_VLA_PROMPT_MODE='provided', STAGE_VLA_INSTRUCTION_SCOPE='subgoal_only'):
            sections = system_prompt()
            self.assertIn('All physical motion', sections['Primitives'])
            self.assertNotIn('navigate_to(', sections['Primitives'])
            self.assertNotIn('move_to', sections['Workflow'])
            self.assertNotIn('Action_Selection', sections)
            self.assertNotIn('Task_Independent_Guidance', sections)


if __name__ == '__main__':
    unittest.main()
