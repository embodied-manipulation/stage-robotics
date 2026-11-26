"""Replay saved actions and optionally force reset-renderer collection; no policy."""
from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
from pathlib import Path

import numpy as np

from research.common import append_json, read_json, write_json
from research.env_server import ResearchEnvFacade

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh diagnostic directory')
    parser.add_argument('--force-gc-step', type=int, default=None,
                        help='Disable automatic GC and collect after this saved action (0 means after reset)')
    args = parser.parse_args()
    faulthandler.enable()
    source = read_json(args.episode)
    reference = Path(source['output_dir']) / 'private'
    rows = [json.loads(line) for line in (reference / 'actions.jsonl').read_text().splitlines() if line]
    if args.force_gc_step is not None and not 0 <= args.force_gc_step <= len(rows):
        parser.error('--force-gc-step must fall within the saved action sequence')
    args.output.mkdir(parents=True, exist_ok=False)
    spec = {**source, 'output_dir': str(args.output), 'diagnostic': True}
    write_json(args.output / 'episode.json', spec)
    os.environ['STAGE_RESTRICTED'] = '1' if source['mode'] == 'restricted' else '0'
    report = {'source_episode': str(args.episode), 'saved_actions': len(rows),
              'policy_calls': 0, 'force_gc_step': args.force_gc_step,
              'preserve_reset_contexts': os.environ.get('STAGE_PRESERVE_RESET_CONTEXTS') == '1',
              'render_order_per_action': [CAMERAS[0], *CAMERAS],
              'scope': 'saved-action and render replay; no policy or RPC scheduling replay'}
    write_json(args.output / 'diagnostic_started.json', report)
    gc_enabled = gc.isenabled()
    if args.force_gc_step is not None:
        gc.disable()
    facade = None
    try:
        facade = ResearchEnvFacade(spec)
        with np.load(reference / 'initial_state.npz') as original, np.load(args.output / 'private/initial_state.npz') as current:
            report['initial_state_equal'] = {key: bool(np.array_equal(original[key], current[key]))
                                             for key in ('qpos', 'qvel', 'ctrl')}
        original_scene = read_json(reference / 'scene.json')
        current_scene = read_json(args.output / 'private/scene.json')
        report['scene_equal'] = {key: original_scene[key] == current_scene[key]
                                 for key in ('task', 'seed', 'split', 'task_language', 'layout_id', 'style_id', 'noslip_iterations')}
        write_json(args.output / 'initial_pair.json', report)
        if not all(report['initial_state_equal'].values()) or not all(report['scene_equal'].values()):
            raise RuntimeError('Diagnostic reset does not match the recorded initial state/scene')

        def views():
            return [facade.render_camera(camera, 256, 256, False).copy() for camera in CAMERAS]

        def collect_at(step):
            if args.force_gc_step != step:
                return
            before_state = facade._state()
            before = views()
            from OpenGL import GL
            before_error = int(GL.glGetError())
            append_json(args.output / 'events.jsonl', {'step': step, 'stage': 'before_forced_gc'})
            count = gc.collect()
            append_json(args.output / 'events.jsonl', {'step': step, 'stage': 'after_forced_gc', 'collected': count})
            after = views()
            after_error = int(GL.glGetError())
            report['forced_gc'] = {'step': step, 'collected': count,
                                  'before_gl_error': before_error, 'after_gl_error': after_error,
                                  'simulator_state_unchanged': bool(np.array_equal(before_state, facade._state())),
                                  'mean_absolute_rgb_difference': [float(np.abs(a.astype(float) - b).mean())
                                                                   for a, b in zip(before, after)],
                                  'after_image_std': [float(image.std()) for image in after],
                                  'retained_reset_simulators': len(facade._retired_simulators)}
            write_json(args.output / 'gc_result.json', report['forced_gc'])

        # Native initial tool observations render RGB-D twice per view: the
        # camera artifact and the world-map input. RLDX then seeds three views.
        for camera in (CAMERAS[0], CAMERAS[2], 'mobilebase0_navview'):
            for _ in range(2):
                facade.render_camera(camera, 256, 256, True)
        views()
        collect_at(0)
        facade.set_action_source('vla', 'saved_action_replay')
        for index, row in enumerate(rows, 1):
            facade.step(np.asarray(row['action'], dtype=np.float64))
            append_json(args.output / 'events.jsonl', {'step': index, 'stage': 'before_record_frame'})
            facade.render_camera(CAMERAS[0], 256, 256, False)
            views()
            collect_at(index)
        report['completed_actions'] = facade.action_count
        report['render_replay_completed'] = True
        write_json(args.output / 'diagnostic.json', report)
    finally:
        if facade is not None:
            facade.close()
        if gc_enabled:
            gc.enable()


if __name__ == '__main__':
    main()
