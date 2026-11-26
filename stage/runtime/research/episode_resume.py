"""Recover an infrastructure-interrupted episode from its saved action prefix."""
import base64
from collections import deque
import json
from pathlib import Path
import shutil

import numpy as np

from research.common import read_json, write_json


def planner_items(source):
    """Rehydrate actual recorded images by their tool-output artifact and step."""
    source = Path(source)
    request = sorted((source / 'private/model_inputs').glob('request_*.json'))[-1]
    payload = read_json(request)
    items = payload['input']
    first = next(i for i, item in enumerate(items)
                 if item.get('role') == 'user' and any(
                     'Intro' in p.get('text', '') for p in item.get('content', [])))
    from research.subgoal_images import LABEL
    items = [item for item in items[first:] if not (
        item.get('role') == 'user' and any(part.get('text', '').startswith(LABEL)
        for part in item.get('content', [])))]
    restored = 0
    for item in items:
        parts = item.get('output')
        if not isinstance(parts, list):
            continue
        pending = []
        for part in parts:
            if part.get('type') == 'input_text':
                text = part.get('text', '')
                try:
                    data = json.loads(text)
                except ValueError:
                    data = None
                if isinstance(data, dict) and 'images' in data and 'step' in data:
                    pending = [source / 'public' / image['artifact'] / f"{data['step']:02d}.png"
                               for image in data['images']]
                elif text == '[Earlier image omitted from this request; recent images retained.]' and pending:
                    pending.pop(0)
            elif part.get('type') == 'input_image':
                if not pending:
                    raise ValueError('Saved image has no explicit tool artifact mapping')
                path = pending.pop(0)
                url = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
                if len(url) != part['image_url']['image_payload_characters']:
                    raise ValueError(f'Saved image differs from recorded request: {path}')
                part['image_url'] = url
                restored += 1
    return items, {'source_request': str(request), 'history_items': len(items), 'restored_images': restored}


def restore_environment(facade):
    source = Path(facade.spec['resume_from'])
    prior = read_json(source / 'private/result.json')
    if prior.get('valid') is not False or prior.get('termination_reason') != 'error':
        raise ValueError('Only infrastructure-interrupted episodes may be recovered')
    original = read_json(source / 'episode.json')
    for key in ('planner_image_window', 'planner_observation_profile'):
        if original.get(key) != facade.spec.get(key):
            raise ValueError(f'Resume changes Planner visual protocol: {key}')
    for key in ('task', 'seed', 'policy', 'action_budget', 'split', 'physics'):
        if original[key] != facade.spec[key]:
            raise ValueError(f'Resume changes episode setting: {key}')
    with np.load(source / 'private/trajectory.npz') as saved:
        initial, expected = saved['states'][0].copy(), saved['states'][-1].copy()
    if not np.array_equal(initial, facade._state()):
        raise ValueError('Resume reset does not match saved initial state')
    rows = [json.loads(s) for s in (source / 'private/actions.jsonl').read_text().splitlines()]
    frames = []
    from types import SimpleNamespace
    from robots.robocasa.vla_skill import VLASkill
    adapter = SimpleNamespace(current_raw_obs=None,
        render_camera=lambda camera_name, height, width: facade.render_camera(camera_name, height, width, False)[::-1])
    history_reader = SimpleNamespace(env=adapter, VLA_OBS_RES=256)
    decisions = [json.loads(s) for s in (source / 'private/subgoal_decisions.jsonl').read_text().splitlines()]
    prompt = decisions[-1]['prompt']
    for index, row in enumerate(rows):
        facade.set_action_source(row['source'], row['tool'])
        facade.step(np.asarray(row['action']))
        if index >= len(rows) - 8:
            adapter.current_raw_obs = facade._observation
            frames.append(VLASkill._raw_frame(history_reader, prompt))
    difference = float(np.max(np.abs(facade._state() - expected)))
    receipt = {'source_episode': str(source), 'replayed_actions': len(rows),
               'initial_state_equal': True, 'final_state_max_abs_error': difference,
               'final_state_matches': bool(np.allclose(facade._state(), expected, rtol=0, atol=1e-9)),
               'replay_policy_calls': 0, 'replay_planner_calls': 0,
               'remaining_action_budget': facade.spec['action_budget'] - len(rows)}
    write_json(facade.private / 'resume_environment.json', receipt)
    if not receipt['final_state_matches']:
        raise ValueError('Saved-action replay does not reproduce the interruption state')
    np.savez_compressed(facade.private / 'resume_vla_history.npz',
        **{key: np.stack([frame[key] for frame in frames]) for key in frames[0] if key != 'annotation.human.task_description'})
    facade.set_action_source('unattributed', 'outside_motion_tool')


def restore_toolkit(toolkit, spec, private):
    source = Path(spec['resume_from'])
    receipt = read_json(private / 'resume_environment.json')
    if not receipt['final_state_matches']:
        raise ValueError('Physical replay has not been accepted')
    # Keep all prior public observation indices valid in the recovered context.
    shutil.copytree(source / 'public', Path(spec['output_dir']) / 'public', dirs_exist_ok=True)
    from rpent.session.base import StepRecord
    manifest = read_json(source / 'public/states.json')
    toolkit._state._steps = [StepRecord(**{**item, 'artifacts': set(item['artifacts'])})
                             for item in manifest['steps']]
    toolkit._state._run_artifacts = set(manifest['run_artifacts'])
    for name in ('tool_calls.jsonl', 'subgoal_decisions.jsonl', 'memory_reads.jsonl', 'feedback_delivery.jsonl'):
        shutil.copyfile(source / 'private' / name, private / name)
    decision = json.loads((private / 'subgoal_decisions.jsonl').read_text().splitlines()[-1])
    gate = toolkit.subgoal_review
    gate.state = decision
    gate.classified_step = decision['observation_step']
    from research import review_window
    window = review_window._window
    window.total = receipt['replayed_actions']
    # The restored request already contains the latest observation and token.
    windows = [json.loads(s) for s in (source / 'private/review_windows.jsonl').read_text().splitlines()]
    last = windows[-1]
    window.token = last['review_token']
    window.last_capture = window.total
    window.last_grant = max((r['actions'] for r in windows if r['event'] == 'planner_review'), default=-1)
    window.used = last['used']
    skill = toolkit._primitives._vla_skill
    skill._load()
    prompt = decision['prompt']
    with np.load(private / 'resume_vla_history.npz') as arrays:
        skill._hist = deque([{**{key: arrays[key][i].copy() for key in arrays.files},
                              'annotation.human.task_description': prompt} for i in range(8)], maxlen=8)
    skill._last_prompt = prompt
    toolkit._primitives._vla_desync = False
    items, planner_receipt = planner_items(source)
    path = private / 'resume_planner_items.json'
    write_json(path, items)
    import os
    os.environ['STAGE_RESUME_ITEMS_FILE'] = str(path)
    os.environ['STAGE_RESUME_ACTION_COUNT'] = str(window.total)
    write_json(private / 'resume_planner.json', {**planner_receipt,
        'restored_subgoal': prompt, 'restored_action_count': window.total,
        'vla_sensor_history_frames': len(skill._hist),
        'policy_rng': 'reseeded to original policy seed; original RNG checkpoint was not recorded',
        'history_delivery': 'native thread/inject_items; completed tools are not reexecuted'})
