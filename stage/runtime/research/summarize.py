"""Report valid denominators and the distinct tool/action collaboration shares."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from research.common import read_json, write_json


def summarize(root: Path) -> dict:
    manifest = read_json(root / 'campaign.json')
    view_path = root / 'latest_attempt_view.json'
    selected = read_json(view_path)['selected'] if view_path.exists() else None
    groups = defaultdict(lambda: {'planned': 0, 'finished': 0, 'valid': 0, 'official_any': 0,
                                  'selected': 0, 'unselected': 0,
                                  'official_terminal': 0, 'actions': 0, 'vla_actions': 0,
                                  'analytic_actions': 0, 'unattributed_actions': 0,
                                  'vla_tool_calls': 0, 'analytic_tool_calls': 0,
                                  'vlm_requests': 0, 'vla_requests': 0,
                                  'cost_sums': {}, 'cost_episodes_with_values': {}})
    for spec in manifest['episodes']:
        key = '/'.join((spec['policy'], spec.get('feedback_arm', spec['mode']), spec['task_set']))
        group = groups[key]
        group['planned'] += 1
        selection = selected.get(spec['key']) if selected is not None else spec
        if selection is None:
            group['unselected'] += 1
            continue
        group['selected'] += 1
        private = Path(selection['output_dir']) / 'private'
        result_path = private / 'result.json'
        if not result_path.exists():
            continue
        result = read_json(result_path)
        group['finished'] += 1
        if not result['valid']:
            continue
        group['valid'] += 1
        group['official_any'] += int(result['official_process_any'])
        group['official_terminal'] += int(result['official_terminal'])
        group['actions'] += result['actions']
        for source in ('vla', 'analytic', 'unattributed'):
            group[f'{source}_actions'] += result['action_steps_by_source'].get(source, 0)
        metrics_path, tool_path = private / 'metrics.json', private / 'tool_metrics.json'
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        tools = read_json(tool_path) if tool_path.exists() else {}
        for name in ('vla_tool_calls', 'analytic_tool_calls'):
            group[name] += metrics.get(name, tools.get(name, 0))
        for field, metric, fallback in (
            ('vla_requests', 'vla_forwards', private / 'vla_requests.jsonl'),
            ('vlm_requests', 'vlm_request_attempts_including_retries', private / 'model_inputs/capabilities.jsonl'),
        ):
            group[field] += metrics[metric] if metrics.get(metric) is not None else len(fallback.read_text().splitlines()) if fallback.exists() else 0
        costs = {name: metrics.get(name) for name in ('planner_wall_seconds', 'episode_wall_seconds',
                 'vlm_request_wall_seconds', 'vla_inference_seconds')}
        costs.update({f'vlm_{name}': value for name, value in metrics.get('vlm_token_usage', {}).items()
                      if name in {'input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_output_tokens'}})
        for name, value in costs.items():
            if value is not None:
                group['cost_sums'][name] = group['cost_sums'].get(name, 0) + value
                group['cost_episodes_with_values'][name] = group['cost_episodes_with_values'].get(name, 0) + 1
    for group in groups.values():
        group['official_any_rate'] = group['official_any'] / group['valid'] if group['valid'] else None
        group['official_terminal_rate'] = group['official_terminal'] / group['valid'] if group['valid'] else None
        group['vla_action_fraction'] = group['vla_actions'] / group['actions'] if group['actions'] else None
        calls = group['vla_tool_calls'] + group['analytic_tool_calls']
        group['vla_tool_call_fraction'] = group['vla_tool_calls'] / calls if calls else None
    return {'schema': 'stage.restricted-summary/v1',
            'selection_source': 'latest_attempt_view' if selected is not None else 'original_manifest',
            'selection_path': str(view_path if selected is not None else root / 'campaign.json'),
            'cost_scope': 'selected valid attempts only; summed per-episode wall time is not campaign calendar duration',
            'request_count_source': 'metrics.json actual VLM attempts including retries and VLA forwards; legacy ledger line counts only when absent',
            'groups': dict(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    write_json(args.root / 'summary.json', result)
    print(result)


if __name__ == '__main__':
    main()
