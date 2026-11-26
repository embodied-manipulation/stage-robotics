"""Summarize measured VLA/Codes use; local completion requires separate evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(rows):
    actions = Counter({'VLA': 0, 'Codes': 0})
    calls = Counter({'VLA': 0, 'Codes': 0})
    goals, variants = {}, set()
    switches = zero_motion = 0
    for row in rows:
        if row['schema'] != 'final-execution-segment/v1':
            raise ValueError('Unsupported execution trace')
        variants.add(row['variant'])
        executor, count = row['executor'], row['physical_actions']
        if executor not in actions or type(count) is not int or count < 0:
            raise ValueError('Invalid executor/action count')
        if row['action_end'] - row['action_start'] != count:
            raise ValueError('Action interval and count disagree')
        calls[executor] += 1
        actions[executor] += count
        zero_motion += count == 0
        switches += bool(row['executor_switch'])
        goal = row['subgoal_id']
        if goal is not None and count:
            goals.setdefault(goal, Counter())[executor] += count
    if len(variants) > 1:
        raise ValueError('Summarize one episode/variant at a time')
    classified = bool(variants) and variants != {'H_unclassified'}
    composition = Counter({'VLA_only': 0, 'Codes_only': 0, 'Mixed': 0})
    goal_rows = []
    for goal, counts in goals.items():
        kind = 'Mixed' if len(counts) == 2 else next(iter(counts)) + '_only'
        composition[kind] += 1
        goal_rows.append(dict(subgoal_id=goal, composition=kind, actions=dict(counts)))
    total = sum(actions.values())
    return dict(schema='final-execution-summary/v1', variant=next(iter(variants), None),
        actions_by_executor=dict(actions), measured_motion_actions=total,
        action_fractions={k: v/total if total else None for k,v in actions.items()},
        action_fraction_denominator='physical actions inside traced executor calls',
        calls_by_executor=dict(calls), zero_motion_calls=zero_motion, executor_switches=switches,
        subgoals_with_motion=len(goals) if classified else None,
        composition_counts=dict(composition) if classified else None,
        subgoals=goal_rows if classified else None,
        local_completion='not scored; requires outcome evidence for the declared condition')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    private = args.episode / 'private'
    with (private / 'execution_segments.jsonl').open() as stream:
        report = summarize(json.loads(line) for line in stream if line.strip())
    result_path = private / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    report['native_task_result'] = dict(source=str(result_path), valid=result.get('valid'),
        success=result.get('success') if result.get('valid') is True else None,
        definition='native whole-task outcome, separate from local Subgoal judgments')
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
