"""Language routing through the native tool, without models or simulator actions."""
import json
from types import SimpleNamespace

import pytest

from research.feedback_campaign import prepare
from research.runner import validate_reference_behavior
from robots.robocasa import prompt_bundle
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.restrictions import Restrictions
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.prompt.utils import format_prompt

TASK = 'Put the bottle in the bottom drawer and close the drawer.'


def primitives(monkeypatch, mode='provided', scope='subgoal_only'):
    monkeypatch.setenv('STAGE_VLA_PROMPT_MODE', mode)
    monkeypatch.setenv('STAGE_VLA_INSTRUCTION_SCOPE', scope)
    monkeypatch.setenv('VLA_MAX_CHUNKS', '40')
    monkeypatch.setenv('VLA_ACTION_STEPS_PER_CHUNK', '8')
    monkeypatch.setenv('VLA_SETTLE_PATIENCE', '999')
    p = object.__new__(RoboCasaPrimitives)
    p.env = SimpleNamespace(current_raw_obs={'language': TASK})
    p._vla_desync = True
    p._recording = False
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {'reason': 'cap'}
    p._vla_skill = SimpleNamespace(run=run)
    return p, calls


def invoke(p, prompt):
    return p.run_vla_skill(None, 70, None, prompt, False, 8, 999, 0.012)


@pytest.mark.parametrize('prompt', [TASK, TASK.upper(),
    '  put the bottle in the bottom drawer AND close the drawer! ',
    'Please execute: ' + TASK])
def test_full_task_rejected_before_policy_or_history_change(monkeypatch, prompt):
    p, calls = primitives(monkeypatch)
    result = invoke(p, prompt)
    assert result['error_code'] == 'full_task_prompt_disallowed'
    assert not calls and p._vla_desync


def test_subgoal_reaches_policy_verbatim_with_original_action_settings(monkeypatch):
    p, calls = primitives(monkeypatch)
    prompt = 'grasp the green bottle beside the cabinet'
    result = invoke(p, prompt)
    assert calls[0][0] == (prompt, 40, 8)
    assert calls[0][1]['settle_patience'] == 999
    assert calls[0][1]['force_reset'] is True
    assert result['effective_prompt'] == prompt and not result['prompt_overridden']
    assert result['vla_instruction_scope'] == 'subgoal_only'
    invoke(p, prompt)
    assert calls[1][1]['force_reset'] is False


def test_historical_task_language_routing_is_unchanged(monkeypatch):
    p, calls = primitives(monkeypatch, 'task_language', 'unrestricted')
    result = invoke(p, 'pick up the bottle')
    assert calls[0][0][0] == TASK
    assert result['prompt_overridden'] is True


@pytest.mark.parametrize('with_feedback', [False, True])
def test_both_arms_present_subgoal_only_prompt_and_tool_schema(tmp_path, monkeypatch, with_feedback):
    p, _ = primitives(monkeypatch)
    monkeypatch.setenv('STAGE_RESTRICTED', '1')
    config = tmp_path / 'restrictions.json'
    config.write_text(json.dumps({'hide_task_feedback': not with_feedback,
                                  'disable_privileged_grasp': not with_feedback}))
    monkeypatch.setenv('STAGE_RESTRICTIONS_FILE', str(config))
    monkeypatch.setenv('STAGE_HYBRID_GUIDANCE', '0')
    monkeypatch.setenv('STAGE_DECISION_REPORTING', 'upstream')
    text = format_prompt(prompt_bundle.system_prompt(), variables={})
    assert 'Pass ONLY the next concrete' in text
    assert 'Never shorten, paraphrase' not in text
    assert 'current language subgoal or the complete task' not in text
    toolkit = object.__new__(RoboCasaToolkit)
    toolkit._primitives = p
    toolkit._restrictions = Restrictions(disable_bootstrap=True, hide_task_feedback=not with_feedback)
    toolkit._state = None
    toolkit._tools = {}
    toolkit._mode = 'evaluation'
    toolkit._register_robocasa_tools()
    for name in ('vla_skill', 'vla_arm'):
        declaration = toolkit._tools[name]
        assert 'never the complete task' in declaration.description
        assert 'Full task descriptions are forbidden' in declaration.input_schema['properties']['prompt']['description']


@pytest.mark.parametrize('mode', ['provided', 'task_language'])
def test_paired_campaign_pins_language_contract(tmp_path, mode):
    manifest = prepare(tmp_path / mode, policy='rldx', plan='smoke', task='DeliverStraw',
                       seed=907, lanes=1, localization_assumption='ideal_robot_localization',
                       vla_prompt_mode=mode)
    assert len(manifest['episodes']) == 2
    for spec in manifest['episodes']:
        validate_reference_behavior(spec)
        assert spec['vla_prompt_mode'] == mode
        assert spec['vla_instruction_scope'] == ('subgoal_only' if mode == 'provided' else 'unrestricted')
        assert spec['hybrid_guidance'] is False


def test_subgoal_scope_cannot_silently_use_full_task_mode():
    with pytest.raises(ValueError, match='subgoal_only requires'):
        validate_reference_behavior({'configuration_role': 'ablation',
                                     'vla_instruction_scope': 'subgoal_only',
                                     'vla_prompt_mode': 'task_language'})
