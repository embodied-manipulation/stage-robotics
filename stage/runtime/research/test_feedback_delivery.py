"""Native feedback contracts through saved state, tool and actual request recorder."""
import json
from types import SimpleNamespace
import pytest
from feedback import project, state_class
from native_prompt import prompt_pair
from research.feedback_delivery import record_delivery
from research.planner_transport import RequestBoundary
from robots.libero.tools import view_env_state as libero_view
from robots.libero.toolkit import EnvState
from robots.robocasa.tools import PublicEnvState, public_result, view_env_state as robocasa_view
from robots.robocasa.prompts.restricted import system_prompt
from rpent.prompt.utils import format_prompt
from rpent.tools import ToolResult


def configure(tmp_path, monkeypatch, with_feedback):
    monkeypatch.setenv('STAGE_RESTRICTED', '1')
    p = tmp_path / 'restrictions.json'
    p.write_text(json.dumps({'hide_task_feedback': not with_feedback,
                            'task_language_names_only': not with_feedback,
                            'disable_privileged_grasp': not with_feedback,
                            'disable_success_stop': not with_feedback,
                            'retain_task_independent_guidance': False}))
    monkeypatch.setenv('STAGE_RESTRICTIONS_FILE', str(p))


def saved_request(tmp_path, data):
    payload = {'model': 'gpt-6-astra', 'reasoning': {'effort': 'low'},
               'input': [{'type': 'custom_tool_call_output', 'call_id': 'cpu-test',
                         'output': [{'type': 'input_text', 'text': ToolResult(data=data).to_text()}]}]}
    RequestBoundary(tmp_path / 'model_inputs', enforce=False)(payload)
    saved = json.loads(next((tmp_path / 'model_inputs').glob('request_*.json')).read_text())
    return json.loads(saved['input'][0]['output'][0]['text'])


@pytest.mark.parametrize('with_feedback', [False, True])
@pytest.mark.parametrize('success', [None, False, True])
def test_libero_native_fields_and_names_reach_model(tmp_path, monkeypatch, with_feedback, success):
    configure(tmp_path, monkeypatch, with_feedback)
    snapshot = {'success': success, 'feedback_action': 7, 'success_stop_triggered': success is True}
    rpc = SimpleNamespace(call=lambda method: snapshot)
    parent = EnvState if with_feedback else PublicEnvState
    state = state_class(parent, rpc, with_feedback, tmp_path)(tmp_path / 'public')
    with state.record_step(state={'robot0_eef_pos': [0, 0, 0], 'object_names': ['bottle']},
                           result={'success': True}, extras={'task_language': 'test'}):
        pass
    data = libero_view(state=state).data
    for _ in range(2):
        data = project(public_result(data), with_feedback=with_feedback)
    record_delivery(data, with_feedback=with_feedback, benchmark='libero', private=tmp_path, tool='view_env_state')
    delivered = saved_request(tmp_path, data)
    assert not {'success', 'libero_terminated', 'task_progress', 'feedback_source',
                'feedback_available', 'success_stop_triggered'} & delivered.keys()
    if with_feedback:
        assert delivered['terminated'] is (success is True)
        assert delivered['truncated'] is False
        assert delivered['state']['object_names'] == ['bottle']
        assert delivered['log']['result']['success'] is True  # native pick heuristic
    else:
        assert not {'terminated', 'truncated'} & delivered.keys()
        assert 'object_names' not in delivered['state']
        assert 'success' not in delivered['log']['result']


@pytest.mark.parametrize('with_feedback', [False, True])
@pytest.mark.parametrize('success', [False, True])
def test_robocasa_native_progress_and_grasp_to_request(tmp_path, monkeypatch, with_feedback, success):
    configure(tmp_path, monkeypatch, with_feedback)
    state = (EnvState if with_feedback else PublicEnvState)(tmp_path / 'public')
    extras = {'task_language': 'test', 'success': success, 'task_progress': {'placed_count': 1}}
    with state.record_step(state={'robot0_eef_pos': [0, 0, 0]}, terminated=success,
                           result={'grasp_contact': True, 'grasp_obj': 'cup'}, extras=extras):
        pass
    data = public_result(robocasa_view(state=state).data)
    record_delivery(data, with_feedback=with_feedback, benchmark='robocasa', private=tmp_path, tool='view_env_state')
    delivered = saved_request(tmp_path, data)
    assert not {'feedback_source', 'feedback_available'} & delivered.keys()
    if with_feedback:
        assert delivered['success'] is success and delivered['robocasa_terminated'] is success
        assert delivered['task_progress']['placed_count'] == 1
        assert delivered['log']['result']['grasp_contact'] is True
        assert delivered['log']['result']['grasp_obj'] == 'cup'
    else:
        assert not {'success', 'robocasa_terminated', 'task_progress'} & delivered.keys()
        assert not {'grasp_contact', 'grasp_obj'} & delivered['log']['result'].keys()


def test_missing_feedback_is_a_delivery_error_not_a_pass(tmp_path):
    with pytest.raises(RuntimeError, match='delivery contract failed'):
        record_delivery({'step': 1, 'state': {}}, with_feedback=True,
                        benchmark='libero', private=tmp_path, tool='move_to')
    row = json.loads((tmp_path / 'feedback_delivery.jsonl').read_text())
    assert row['ok'] is False and 'terminated' in row['missing']


def test_with_prompts_preserve_native_experience_and_finish(tmp_path, monkeypatch):
    from robots.robocasa.prompts import evaluate as rc_native
    from robots.libero.prompts import evaluate as li_native
    configure(tmp_path, monkeypatch, True)
    sections = system_prompt(None)
    for key, expected in [('VLA_Rules', rc_native.VLA_RULES), ('Localization', rc_native.LOCALIZATION),
                          ('Navigation', rc_native.NAVIGATION), ('Gripper_Rules', rc_native.GRIPPER_RULES)]:
        assert sections[key] == expected
    rc = format_prompt(sections, variables={'output_dir': '/public', 'recipe_tag': 'test'})
    assert 'state.success' not in rc and 'top-level success' in rc
    spec = {'suite': 'libero_10_task', 'task': 0, 'seed': 0, 'enable_sam3': False,
            'feedback_arm': 'with_success_feedback', 'retain_task_independent_guidance': True}
    li, _ = prompt_pair(spec, tmp_path/'public', tmp_path/'prompt')
    assert li_native.KEY_HYPERPARAMETERS in li
    assert 'battle-tested on seed 0' in li and 'object_names' in li
    assert 'terminated:false + strategy_notes' in li
    assert 'Brief reasoning before each tool call (1-2 sentences): observation → decision.' in li
    assert 'Report visual assessment separately' not in li + rc
    assert 'Placement from the current grasp' not in li + rc
    assert '`segment`' not in li and 'SAM3 localization aid' not in li
    assert 'task_progress' not in li and 'libero_terminated' not in li
    assert 'success_stop_triggered' not in li and 'feedback_source' not in li
    assert 'READ SEED-0 STRATEGY REFERENCES' not in li
    assert '/restricted-unavailable' not in li
    assert 'consult BOTH' not in li


def test_libero_transient_success_stays_terminated_in_native_public_state(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, True)
    snapshot = {'success': False, 'feedback_action': 5, 'success_stop_triggered': True}
    state = state_class(EnvState, SimpleNamespace(call=lambda method: snapshot), True, tmp_path)(tmp_path/'public')
    with state.record_step(state={'robot0_eef_pos': [0,0,0]}, terminated=True):
        pass
    data = project(public_result(libero_view(state=state).data), with_feedback=True)
    delivered = saved_request(tmp_path, data)
    assert delivered['terminated'] is True
    assert not {'success', 'success_stop_triggered'} & delivered.keys()
    private = json.loads((tmp_path/'feedback_snapshots.jsonl').read_text())
    assert private['success'] is False and private['terminated'] is True


def test_robocasa_stop_reuses_cached_official_measurement(monkeypatch):
    from research.env_server import ResearchEnvFacade, RoboCasaEnvFacade
    from robots.robocasa.restrictions import Restrictions
    def forbidden(*args):
        raise AssertionError('stop check must not reevaluate a stateful predicate')
    monkeypatch.setattr(RoboCasaEnvFacade, 'check_success', forbidden)
    env = object.__new__(ResearchEnvFacade)
    env.spec = {'feedback_sampling': 'step_cached'}
    env.restrictions = Restrictions(disable_success_stop=False)
    env.official_terminal = True
    assert env.success_for_stop() is True and env.success_for_stop() is True
    env.official_terminal = False
    assert env.success_for_stop() is False
    env.restrictions = Restrictions(disable_success_stop=True)
    with pytest.raises(PermissionError):
        env.success_for_stop()


def test_native_audit_requires_real_official_success_for_termination(tmp_path):
    from audit_feedback_delivery import audit
    (tmp_path/'feedback_snapshots.jsonl').write_text(json.dumps({
        'step': 2, 'feedback_action': 2, 'success': False, 'terminated': True, 'truncated': False})+'\n')
    (tmp_path/'official_evaluation.jsonl').write_text('\n'.join(json.dumps({
        'action': i, 'official_success': v}) for i,v in [(1,True),(2,False)]))
    saved_request(tmp_path, {'step':2,'state':{'object_names':['bottle']},'terminated':True,'truncated':False})
    assert audit(tmp_path)['delivery_verified']
    (tmp_path/'official_evaluation.jsonl').write_text('\n'.join(json.dumps({
        'action': i, 'official_success': False}) for i in [1,2]))
    assert not audit(tmp_path)['delivery_verified']
