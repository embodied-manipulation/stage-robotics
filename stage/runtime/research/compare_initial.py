"""Compare the actual initial robot/simulator state of each defaults control pair."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from research.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pair-field', default='mode')
    parser.add_argument('--left', default='original_defaults')
    parser.add_argument('--right', default='instrumented_defaults')
    args = parser.parse_args()
    pairs = {}
    for spec in read_json(args.root / 'campaign.json')['episodes']:
        pairs.setdefault((spec['task'], spec['seed']), {})[spec[args.pair_field]] = Path(spec['output_dir']) / 'private'
    results = []
    for (task, seed), paths in pairs.items():
        original, instrumented = paths[args.left], paths[args.right]
        missing = [str(path / name) for path in (original, instrumented)
                   for name in ('initial_state.npz', 'scene.json') if not (path / name).exists()]
        if missing:
            results.append({'task': task, 'seed': seed, 'complete': False, 'missing': missing})
            continue
        fields = {}
        with np.load(original / 'initial_state.npz') as left, np.load(instrumented / 'initial_state.npz') as right:
            for key in ('qpos', 'qvel', 'ctrl'):
                fields[key] = bool(np.array_equal(left[key], right[key]))
        a, b = read_json(original / 'scene.json'), read_json(instrumented / 'scene.json')
        import imageio.v2 as imageio
        image_differences = {}
        for camera in ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'):
            left_image = imageio.imread(original / f'initial_{camera}.png')
            right_image = imageio.imread(instrumented / f'initial_{camera}.png')
            difference = np.abs(left_image.astype(float) - right_image.astype(float))
            image_differences[camera] = {'bitwise_equal': bool(np.array_equal(left_image, right_image)),
                                         'mean_absolute_difference_uint8': float(difference.mean()),
                                         'maximum_absolute_difference_uint8': float(difference.max())}
        fields['scene'] = all(a[key] == b[key] for key in ('task', 'seed', 'split', 'task_language', 'layout_id', 'style_id', 'noslip_iterations'))
        results.append({'task': task, 'seed': seed, 'complete': True,
                        'physical_state_equal': all(fields.values()), 'fields': fields,
                        'image_differences': image_differences})
    matched = bool(results) and all(r.get('physical_state_equal', False) for r in results)
    report = {'pairs': results, 'all_physical_states_equal': matched,
              'all_images_bitwise_equal': all(r.get('complete') and all(i['bitwise_equal'] for i in r['image_differences'].values()) for r in results),
              'image_note': 'RGB equality is reported separately; small renderer differences are not a physical-state mismatch. Review substantive visual differences.'}
    write_json(args.root / 'initial_comparison.json', report)
    print(f"{len(results)} pairs; physical states matched: {matched}; images bitwise equal: {report['all_images_bitwise_equal']}")
    if not matched:
        raise SystemExit('Initial state pair missing or physically different; inspect initial_comparison.json')


if __name__ == '__main__':
    main()
