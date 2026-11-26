import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_review_window import TestToolkit
from stage_bridge.review import install, ANALYTIC, VLA
from stage_bridge.final_config import VARIANTS, validate, capabilities, validate_pairs
from stage_bridge.execution_trace import ExecutionTrace
from rpent.tools import ToolResult, tool
from research import review_window


class CodesOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name);review_window._window=None
        from rpent.utils.logging import init_output_dir
        init_output_dir(self.path)

    def toolkit(self,**kwargs):
        tk=TestToolkit();self.addCleanup(tk.close)
        return install(tk,self.path,**kwargs)

    def test_c_variant_registered_and_pairing(self):
        self.assertEqual(VARIANTS['C'],(False,True,True))
        c=dict(final_variant='C',vla_enabled=False,code_as_policy=True,classification_enabled=True,
               review_window_actions=40,external_memory=False,pair_id='p',key='C/p')
        self.assertEqual(validate(c),(True,True));self.assertEqual(capabilities(c),(False,True,True))
        for bad in (dict(c,vla_enabled=True),{k:v for k,v in c.items() if k!='vla_enabled'}):
            with self.assertRaises(ValueError):validate(bad)
        v=dict(c,final_variant='V',vla_enabled=True,code_as_policy=False,key='V/p')
        h=dict(c,final_variant='H',vla_enabled=True,key='H/p')
        self.assertEqual(validate_pairs([v,h,c]),1)
        with self.assertRaises(ValueError):validate_pairs([v,h,c,dict(v,pair_id='other',key='V/other')])

    def test_c_toolkit_has_no_vla_tools_and_identical_classifier(self):
        arms={name:self.toolkit(code_as_policy=codes,vla_enabled=vla)
              for name,(vla,codes,labels) in VARIANTS.items() if labels}
        schemas={name:{t.name:t for t in tk.list_tools()} for name,tk in arms.items()}
        self.assertFalse(VLA & schemas['C'].keys())
        self.assertTrue(ANALYTIC & schemas['C'].keys())
        self.assertEqual(ANALYTIC & schemas['C'].keys(),ANALYTIC & schemas['H'].keys())
        for name in ('V','H'):
            self.assertEqual(schemas['C']['subgoal_review'].input_schema,schemas[name]['subgoal_review'].input_schema)

    def test_c_libero_toolkit_has_no_vla_tools(self):
        from stage_bridge.libero import install as install_libero
        tk=TestToolkit();self.addCleanup(tk.close)
        install_libero(tk,self.path,code_as_policy=True,vla_enabled=False)
        self.assertFalse(VLA & {t.name for t in tk.list_tools()})
        self.assertIn('move_to',{t.name for t in tk.list_tools()})
        review_window._window=None

    def test_c_vla_call_is_violation(self):
        tk=self.toolkit(code_as_policy=True,vla_enabled=False)
        self.assertTrue(tk.execute_tool('vla_skill',{'prompt':'cup'}).is_error)
        self.assertEqual(json.loads((self.path/'protocol_violation.json').read_text())['variant'],'C')
        with self.assertRaises(RuntimeError):
            ExecutionTrace(self.path,variant='C',classification_enabled=True).record(
                tk,tool='pi0_doubled',before=0,after=0)

    def test_c_trace_records_codes_only(self):
        tk=self.toolkit(code_as_policy=True,vla_enabled=False)
        tk.subgoal_review.state=dict(subgoal_id='sg001',prompt='place cup')
        ExecutionTrace(self.path,variant='C',classification_enabled=True).record(tk,tool='move_to',before=0,after=3)
        row=json.loads((self.path/'execution_segments.jsonl').read_text())
        self.assertEqual(row['executor'],'Codes');self.assertEqual(row['physical_actions'],3)

    def test_libero_effort_counts_codes_for_c(self):
        from stage_bridge.libero_effort import install as effort, guidance
        @tool
        def finish(status:str,summary:str)->ToolResult:
            """Finish the episode."""
            return ToolResult(data={'_finish':True})
        tk=SimpleNamespace(_tools={'finish':finish},_primitives=SimpleNamespace(
            env=SimpleNamespace(terminated=False,truncated=False)))
        actions=999
        effort(tk,self.path,minimum=1000,action_count=lambda:actions,code_as_policy=True,vla_enabled=False)
        self.assertTrue(tk._tools['finish'](status='stuck',summary='blocked').is_error)
        actions=1000
        self.assertTrue(tk._tools['finish'](status='stuck',summary='blocked').data['_finish'])
        self.assertEqual(json.loads((self.path/'minimum_effort_decisions.jsonl').read_text().splitlines()[-1])['counted_executors'],['Codes'])
        self.assertIn('Codes physical',guidance(1000,code_as_policy=True,vla_enabled=False))
