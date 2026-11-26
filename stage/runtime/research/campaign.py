"""Materialize frozen Target50 cohorts without submitting jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.common import REPO, read_json, write_json


def episodes(plan: str, policy: str, root: Path, *, original_source: Path | None = None) -> list[dict]:
    tasks = read_json(REPO / 'research/data/target50_tasks.json')['tasks']
    if plan == 'consistency3':
        selected = [(tasks[i], tasks[i]['seeds'][0], mode) for i in (2, 20, 38)
                    for mode in ('original_defaults', 'instrumented_defaults')]
    elif plan == 'restricted_smoke':
        selected = [(tasks[i], tasks[i]['seeds'][0], 'restricted') for i in (2, 20)]
    else:
        selected = [(task, seed, 'restricted') for task in tasks for seed in task['seeds']]
    records = []
    for task, seed, mode in selected:
        key = f"{policy}/{mode}/{task['name']}/seed_{seed}"
        records.append({
            'schema': 'stage.restricted-episode/v1', 'key': key,
            'task': task['name'], 'task_set': task['task_set'], 'seed': seed,
            'policy': policy, 'mode': mode, 'split': 'pretrain',
            'official_horizon': task['official_horizon'], 'action_budget': int(task['official_horizon'] * 1.6),
            'source_root': str(original_source if mode == 'original_defaults' and original_source else REPO),
            'output_dir': str(root / 'episodes' / key),
            'environment_reset': {'version': 'xr1-gym-reset/v1', 'constructor_seed': 7,
                                  'constructor_resets': 1, 'episode_reset': 'env.rng=default_rng(seed); env.reset()',
                                  'bootstrap_reset': 'attach matched episode; no extra physical reset'},
            'physics': {'noslip_iterations': 1}, 'planner': {'model': 'gpt-6-astra', 'reasoning_effort': 'low'},
            'native_action_steps': 8, 'policy_seed': 7, 'planner_timeout_s': 7200,
            'memory_profile': 'hf', 'max_turns': 120,
            'configuration_role': 'baseline', 'vla_prompt_mode': 'task_language',
            'hybrid_guidance': False, 'prompt_profile': 'upstream_with_restricted_edits',
            'vla_max_chunks': 40, 'vla_settle_patience': 999,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', choices=('consistency3', 'restricted_smoke', 'target340'), required=True)
    parser.add_argument('--policy', choices=('rldx', 'xr1'), default='rldx')
    parser.add_argument('--lanes', type=int, default=1)
    parser.add_argument('--original-source', type=Path)
    parser.add_argument('--memory-dir', type=Path)
    args = parser.parse_args()
    if args.plan == 'consistency3' and args.original_source is None:
        parser.error('--original-source must point to an unmodified efc392a checkout for original/default pairs')
    if args.lanes < 1:
        parser.error('--lanes must be positive')
    path = args.root / 'campaign.json'
    if path.exists():
        raise FileExistsError(path)
    records = episodes(args.plan, args.policy, args.root.resolve(), original_source=args.original_source)
    for index, spec in enumerate(records):
        spec['lane'] = index % args.lanes
        if args.memory_dir:
            spec['memory_dir'] = str(args.memory_dir.resolve())
        write_json(Path(spec['output_dir']) / 'episode.json', spec)
    write_json(path, {'schema': 'stage.restricted-campaign/v1',
                     'plan': args.plan, 'policy': args.policy,
                     'source_root': str(REPO), 'lanes': args.lanes, 'launch_authorized': False,
                     'required_gate': 'rldx340_complete_and_reviewed' if args.policy == 'xr1' and args.plan == 'target340' else 'root_smoke_review',
                     'episodes': records})
    print(f'{path}: {len(records)} episodes; launch_authorized=false')


if __name__ == '__main__':
    main()
