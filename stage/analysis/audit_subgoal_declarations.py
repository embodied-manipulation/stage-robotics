"""Read indexed valid historical episodes; do not modify execution or results."""
import collections
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / 'historical_episodes.jsonl'


def main():
    indexed = [json.loads(line) for line in INDEX.read_text().splitlines()]
    unique = {}
    for row in indexed:
        if row['status'] == 'valid' and row.get('online_log'):
            unique[str(Path(row['result_file']).resolve())] = row
    groups = collections.defaultdict(collections.Counter)
    episodes, evidence = [], []
    for result_path, row in sorted(unique.items()):
        if json.loads(Path(result_path).read_text()).get('valid') is not True:
            raise ValueError(f'Indexed valid result changed: {result_path}')
        log = Path(row['online_log'])
        calls = log.with_name('tool_calls.jsonl')
        counts = collections.Counter(episodes=1)
        for line_number, line in enumerate(log.read_text().splitlines(), 1):
            event = json.loads(line)
            if event.get('type') != 'planner_intervention':
                continue
            decision, progress = event['decision'], event['progress']
            counts['accepted'] += 1
            counts['decision:' + decision] += 1
            counts[f'pair:{decision}/{progress}'] += 1
            errors = event.get('model_errors', [])
            counts['any_logged_model_error'] += bool(errors)
            for code in errors:
                counts['model_error:' + code] += 1
            mismatch = ((decision == 'advance' and progress != 'complete') or
                        (decision == 'recover' and progress not in ('blocked', 'uncertain')))
            counts['decision_progress_mismatch'] += mismatch
            logged = bool(set(errors) & {
                'advance_without_complete_progress',
                'recover_without_blocked_or_uncertain_progress'})
            if mismatch != logged:
                raise ValueError(f'Logged/recomputed rule differs: {log}:{line_number}')
            if errors or (decision == 'continue' and progress == 'complete'):
                evidence.append(dict(kind='accepted_declaration', path=str(log),
                                     line=line_number, event=event))
        for line_number, line in enumerate(calls.read_text().splitlines(), 1):
            event = json.loads(line)
            if event.get('tool') != 'subgoal_review':
                continue
            counts['review_calls'] += 1
            error = event.get('error')
            if error:
                counts['rejected_calls'] += 1
                counts['rejection_reason:' + str(error)] += 1
                evidence.append(dict(kind='rejected_review', path=str(calls),
                                     line=line_number, event=event))
            else:
                counts['nonerror_calls'] += 1
        if counts['nonerror_calls'] != counts['accepted']:
            raise ValueError(f'Accepted log/tool-call counts differ: {log}')
        policy = row['config'].get('policy')
        group = (policy, row['mode'])
        groups[group].update(counts)
        episodes.append(dict(result_file=result_path, online_log=str(log),
                             tool_log=str(calls), campaign=row['campaign'],
                             policy=policy, mode=row['mode'], counts=dict(counts)))
    total = collections.Counter()
    for counts in groups.values():
        total.update(counts)
    report = dict(
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        index=str(INDEX),
        scope='Live-read logs of unique valid results with online_log in existing historical index; not all historical runs or invalid attempts.',
        definitions={
            'decision_progress_mismatch': 'Accepted advance with progress != complete, or recover with progress outside blocked/uncertain.',
            'mismatch_denominator': 'Accepted planner_intervention events, including start and continue; finish excluded.',
            'transition_only_denominator': 'Accepted advance + recover events.',
            'rejection_denominator': 'Logged subgoal_review tool calls, including repeated attempts; other tools excluded.',
            'rejected_calls': 'Nonempty error field in tool_calls.jsonl; no claim about attempts absent from this log.',
            'continue_complete': 'Reported separately; not flagged by existing model_errors rules.',
            'comparison': 'Historical cohorts differ in models/protocols/tasks; descriptive only, not paired V/H comparison.',
        },
        totals=dict(total),
        groups=[dict(policy=k[0], mode=k[1], counts=dict(v)) for k, v in sorted(groups.items())],
        episodes=episodes)
    (ROOT / 'subgoal_declaration_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'subgoal_declaration_evidence.jsonl').write_text(
        ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in evidence))
    print(json.dumps(dict(totals=report['totals'], groups=report['groups']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
