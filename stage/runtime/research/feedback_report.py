"""Report feedback arms by Set on their common valid task/seed pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import median

from research.common import read_json, write_json

ARMS = ('without_success_feedback', 'with_success_feedback')
LABELS = dict(zip(ARMS, ('Without Feedback', 'Official Task Feedback')))
SETS = ('atomic_seen', 'composite_seen', 'composite_unseen')
MEASURES = {
    'episode_wall_seconds': 'episode_wall_seconds',
    'output_tokens': 'output_tokens',
    'vla_tool_calls': 'vla_tool_calls',
    'vlm_calls': 'vlm_request_attempts_including_retries',
    'vla_forwards': 'vla_forwards',
}


def _load(spec: dict, selection: dict | None) -> dict:
    row = {name: spec[name] for name in ('key', 'task', 'task_set', 'seed', 'feedback_arm')}
    row.update(status='pending', included=False, output_dir=None)
    if selection is None:
        return row
    private = Path(selection['output_dir']) / 'private'
    row['output_dir'] = str(private.parent)
    if not (private / 'result.json').exists():
        return row
    result = read_json(private / 'result.json')
    if result.get('valid') is not True:
        row['status'] = 'invalid'
        return row
    if not (private / 'metrics.json').exists():
        row['status'] = 'incomplete'
        return row
    metrics = read_json(private / 'metrics.json')
    if metrics.get('valid') is not True:
        row['status'] = 'invalid'
        return row
    success = result.get('official_process_any')
    if not isinstance(success, bool) or metrics.get('official_process_any') != success:
        raise ValueError(f"Missing or inconsistent official success: {spec['key']}")
    first = metrics.get('first_success_step', result.get('first_success_step'))
    row.update(status='valid', success=success, first_success_step=first if success else None,
               termination_reason=metrics.get('termination_reason'), actions=metrics.get('actions'),
               **{name: metrics.get(field) for name, field in MEASURES.items()})
    usage = metrics.get('vlm_token_usage', {})
    row['output_tokens'] = usage.get('output_tokens')
    row['token_usage_source'] = usage.get('source')
    row['vlm_attempts_without_reported_usage'] = usage.get('attempts_without_reported_usage')
    return row


def summarize(root: Path) -> dict:
    """Select one attempt per key; retain task failures and exclude invalid pairs."""
    manifest = read_json(root / 'campaign.json')
    view = root / 'latest_attempt_view.json'
    selected = read_json(view)['selected'] if view.exists() else None
    pairs = defaultdict(dict)
    rows = []
    for spec in manifest['episodes']:
        arm = spec.get('feedback_arm')
        if arm not in ARMS or spec['task_set'] not in SETS:
            raise ValueError('Expected declared feedback arms and RoboCasa Target50 Sets')
        pair = (spec['task_set'], spec['task'], spec['seed'])
        if arm in pairs[pair]:
            raise ValueError(f'Duplicate logical episode: {pair} / {arm}')
        row = _load(spec, selected.get(spec['key']) if selected is not None else spec)
        pairs[pair][arm] = row
        rows.append(row)
    for pair in pairs.values():
        if set(pair) != set(ARMS):
            raise ValueError('Manifest must contain both arms of every pair')
        if all(row['status'] == 'valid' for row in pair.values()):
            for row in pair.values():
                row['included'] = True
    groups = []
    for task_set in SETS:
        for arm in ARMS:
            planned = [r for r in rows if r['task_set'] == task_set and r['feedback_arm'] == arm]
            included = [r for r in planned if r['included']]
            status = Counter(r['status'] for r in planned)
            n = len(included)
            successes = sum(r['success'] for r in included)
            group = {'task_set': task_set, 'feedback_arm': arm, 'condition': LABELS[arm],
                     'planned': len(planned), 'valid_in_arm': status['valid'],
                     'valid_pairs': n, 'invalid': status['invalid'], 'pending': status['pending'],
                     'incomplete': status['incomplete'], 'unpaired_valid': status['valid'] - n,
                     'official_any_successes': successes,
                     'official_any_rate': successes / n if n else None}
            for name in MEASURES:
                values = [r[name] for r in included if r.get(name) is not None]
                # A missing value must not become a measured zero or a full total.
                group[f'{name}_coverage'] = len(values)
                group[f'{name}_known_sum'] = sum(values) if values else None
                group[f'{name}_total'] = sum(values) if n and len(values) == n else None
                group[f'{name}_mean'] = sum(values) / n if n and len(values) == n else None
            first = [r['first_success_step'] for r in included
                     if r['success'] and r.get('first_success_step') is not None]
            group.update(first_success_step_n=len(first),
                         first_success_step_mean=sum(first) / successes if successes and len(first) == successes else None,
                         first_success_step_median=median(first) if successes and len(first) == successes else None,
                         token_usage_unreported_attempts=sum(r.get('vlm_attempts_without_reported_usage') or 0 for r in included))
            groups.append(group)
    return {'schema': 'stage.feedback-report/v1', 'plan': manifest.get('plan'),
            'diagnostic': manifest.get('diagnostic', False),
            'selection_source': str(view if selected is not None else root / 'campaign.json'),
            'population': 'common valid task/seed pairs; all valid task failures retained',
            'cost_scope': 'selected paired valid episodes, successes and failures; excluded attempts not in totals',
            'time_scope': 'environment startup through final aggregation; excludes lane model load and scheduler queue',
            'first_success_scope': 'first native official true environment step, conditional on success in each arm; failures are null',
            'output_token_scope': 'provider-reported output tokens include reasoning; no extra reasoning-token addition',
            'groups': groups, 'episodes': rows}


def _format(value, digits: int = 1) -> str:
    return 'NA' if value is None else f'{value:.{digits}f}'


def write_report(report: dict, directory: Path) -> None:
    """Export six Set/condition rows plus episode-level selection evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'feedback_summary.json', report)
    for name, rows in (('feedback_summary.csv', report['groups']), ('feedback_episodes.csv', report['episodes'])):
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# RoboCasa feedback pilot', '',
             f"Plan: `{report['plan']}`; diagnostic: `{report['diagnostic']}`.", '',
             'Population: common valid task/seed pairs; task failures retained. Unpaired/invalid/pending counts are in CSV/JSON.', '',
             '| Set | Condition | Paired n / planned | Official any-step | Mean time / episode (s) | Total output tokens | VLA calls total (mean/ep) | VLM calls total (mean/ep) | First success step mean (median; n) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in report['groups']:
        rate = g['official_any_rate']
        sr = 'NA' if rate is None else f"{100 * rate:.1f}% ({g['official_any_successes']}/{g['valid_pairs']})"
        calls = [f"{_format(g[name + '_total'], 0)} ({_format(g[name + '_mean'])})" for name in ('vla_tool_calls', 'vlm_calls')]
        first = f"{_format(g['first_success_step_mean'])} ({_format(g['first_success_step_median'])}; {g['first_success_step_n']})"
        lines.append(f"| {g['task_set']} | {g['condition']} | {g['valid_pairs']}/{g['planned']} | {sr} | "
                     f"{_format(g['episode_wall_seconds_mean'])} | {_format(g['output_tokens_total'], 0)} | "
                     f"{calls[0]} | {calls[1]} | {first} |")
    lines += ['', 'Time and costs cover successful and failed valid episodes through termination, not only the prefix to success.',
              'VLA calls count native VLA tool invocations; forwards are separate CSV fields. VLM calls count transport attempts including retries.',
              'First-success steps condition on each arm\'s own successes; different success sets do not establish faster execution.',
              'Output tokens are reported usage, including reasoning once. Missing request usage is flagged in CSV/JSON; NA is not zero.', '']
    (directory / 'feedback_summary.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    directory = args.output_dir or args.root / 'reports'
    report = summarize(args.root)
    write_report(report, directory)
    print(directory / 'feedback_summary.md')


if __name__ == '__main__':
    main()
