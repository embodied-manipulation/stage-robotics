"""Prepare a paired feedback smoke or Target50 cohort; never submit jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from research.campaign import episodes
from research.common import REPO, read_json, write_json


def prepare(root: Path, *, policy: str, plan: str, task: str, seed: int | None,
            lanes: int, localization_assumption: str, vla_prompt_mode: str = 'provided') -> dict:
    if vla_prompt_mode not in {'provided', 'task_language'}:
        raise ValueError('vla_prompt_mode must be provided or task_language')
    if localization_assumption != 'ideal_robot_localization':
        raise ValueError('The native adapter requires explicitly declared ideal robot localization')
    if lanes < 1:
        raise ValueError('lanes must be positive')
    root = root.resolve()
    if (root / 'campaign.json').exists() or (root.exists() and any(root.iterdir())):
        raise FileExistsError(f'Use a fresh output directory: {root}')
    base = episodes('target340', policy, root)
    if plan == 'smoke':
        candidates = [item for item in base if item['task'] == task]
        if not candidates:
            raise ValueError(f'Task absent from frozen Target50: {task}')
        chosen = candidates[0] if seed is None else next((s for s in candidates if s['seed'] == seed), None)
        if chosen is None:
            raise ValueError('Choose a seed from this task in research/data/target50_tasks.json')
        base = [chosen]
    elif plan not in {'pilot250', 'target340'}:
        raise ValueError('plan must be smoke, pilot250 or target340')
    elif seed is not None:
        raise ValueError('Target50 cohorts use the frozen per-task seed lists')
    if plan == 'pilot250':
        # Keep the first five frozen trials of every task, including Atomic.
        seen = Counter()
        selected = []
        for item in base:
            seen[item['task']] += 1
            if seen[item['task']] <= 5:
                selected.append(item)
        base = selected
    configurations = {}
    for arm in ('without_success_feedback', 'with_success_feedback'):
        path = root / 'configs' / f'{arm}.yaml'
        write_json(path, read_json(REPO / 'configs' / f'{arm}.yaml'))
        configurations[arm] = path
    specs = []
    for pair_index, template in enumerate(base):
        # Alternate within-pair order to avoid always running feedback last.
        arms = list(configurations)
        if pair_index % 2:
            arms.reverse()
        for arm in arms:
            key = f"{policy}/{arm}/{template['task']}/seed_{template['seed']}"
            spec = {**template, 'key': key, 'output_dir': str(root / 'episodes' / key),
                    'lane': pair_index % lanes, 'feedback_arm': arm,
                    'feedback_profile': 'none' if arm.startswith('without') else 'official_task_feedback',
                    'feedback_sampling': 'step_cached', 'decision_reporting_profile': 'upstream',
                    'feedback_protocol': 'without-restricted-original' if arm.startswith('without') else 'native-privileged-experience-nosam3/v2',
                    'prompt_experience': 'restricted' if arm.startswith('without') else 'upstream_embedded',
                    'success_stop_policy': 'disabled' if arm.startswith('without') else 'native_vla_chunk_boundary',
                    'restrictions_file': str(configurations[arm]),
                    'robot_state_assumption': localization_assumption,
                    'diagnostic': plan == 'smoke'}
            spec.update(
                vla_prompt_mode=vla_prompt_mode,
                vla_instruction_scope='subgoal_only' if vla_prompt_mode == 'provided' else 'unrestricted',
                configuration_role='ablation' if vla_prompt_mode == 'provided' else 'baseline',
                language_protocol='provided-subgoal-only/v1' if vla_prompt_mode == 'provided' else 'upstream-task-language/v1',
            )
            write_json(Path(spec['output_dir']) / 'episode.json', spec)
            write_json(Path(spec['output_dir']) / 'runtime_profile.json',
                       {'fix_id': 'preserve_reset_contexts', 'preserve_reset_contexts': True})
            specs.append(spec)
    manifest = {'schema': 'stage.feedback-campaign/v1', 'plan': plan,
                'policy': policy, 'lanes': lanes, 'source_root': str(REPO),
                'diagnostic': plan == 'smoke',
                'launch_authorized': False, 'required_gate': 'feedback_profile_review',
                'pairs_by_set': dict(Counter(item['task_set'] for item in base)),
                'seed_selection': 'first five frozen seeds per task' if plan == 'pilot250' else 'frozen registry',
                'planner_auth': {'mode': 'chatgpt_subscription', 'source': 'explicit operator Codex home'},
                'robot_state_assumption': localization_assumption,
                'vla_prompt_mode': vla_prompt_mode,
                'language_protocol': 'provided-subgoal-only/v1' if vla_prompt_mode == 'provided' else 'upstream-task-language/v1',
                'episodes': specs}
    write_json(root / 'campaign.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', choices=('smoke', 'pilot250', 'target340'), default='smoke')
    parser.add_argument('--task', default='OpenDrawer')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--policy', choices=('rldx', 'xr1'), default='rldx')
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--vla-prompt-mode', choices=('provided', 'task_language'), default='provided',
                        help='New runs use subgoal-only provided prompts; task_language is the historical reference')
    parser.add_argument('--localization-assumption', choices=('ideal_robot_localization',), required=True,
                        help='Native world-frame robot poses are ideal simulation localization, not a deployed estimator')
    args = parser.parse_args()
    manifest = prepare(**vars(args))
    print(f"{args.root / 'campaign.json'}: {len(manifest['episodes'])} episodes; no jobs submitted")


if __name__ == '__main__':
    main()
