"""Follow rejected continue edits through actual calls and accepted goal states."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    evidence = [json.loads(line) for line in
                (ROOT / 'subgoal_declaration_evidence.jsonl').read_text().splitlines()]
    targets = [e for e in evidence if e['kind'] == 'rejected_review' and
               str(e['event'].get('error', '')).startswith('continue cannot change')]
    cases = []
    for path in sorted({e['path'] for e in targets}):
        calls = [json.loads(line) for line in Path(path).read_text().splitlines()]
        states = [json.loads(line) for line in Path(path).with_name('subgoal_decisions.jsonl').read_text().splitlines()
                  if json.loads(line).get('type') == 'planner_intervention']
        accepted = {}
        state_index = 0
        for i, call in enumerate(calls):
            if call.get('tool') == 'subgoal_review' and not call.get('error'):
                state = states[state_index]
                for key in ('decision', 'progress', 'observation_step'):
                    assert call['arguments'][key] == state[key], (path, i, key)
                accepted[i] = state
                state_index += 1
        assert state_index == len(states)
        for target in (e for e in targets if e['path'] == path):
            i = target['line'] - 1
            before = max(k for k in accepted if k < i)
            next_call = next(k for k in range(i + 1, len(calls)) if calls[k].get('tool') == 'subgoal_review')
            after = min(k for k in accepted if k > i)
            until = next((k for k in range(after + 1, len(calls)) if calls[k].get('tool') == 'subgoal_review'), len(calls))
            physical = lambda start, stop: [dict(line=k + 1, **calls[k]) for k in range(start, stop)
                                            if (calls[k].get('physical_actions') or 0) > 0]
            cases.append(dict(
                case_id=len(cases) + 1, tool_log=path, rejected_line=i + 1,
                prior_state=accepted[before], rejected=calls[i],
                next_call_line=next_call + 1, next_call=calls[next_call],
                accepted_line=after + 1, accepted_call=calls[after], accepted_state=accepted[after],
                same_stage=accepted[before]['subgoal_id'] == accepted[after]['subgoal_id'],
                motion_before_acceptance=physical(i + 1, after),
                motion_after_acceptance_until_next_review=physical(after + 1, until)))
    summary = dict(
        rejected_calls=len(cases), episodes=len({c['tool_log'] for c in cases}),
        next_call_decisions=dict(collections.Counter(c['next_call']['arguments']['decision'] for c in cases)),
        next_call_rejected=sum(bool(c['next_call'].get('error')) for c in cases),
        next_accepted_decisions=dict(collections.Counter(c['accepted_state']['decision'] for c in cases)),
        unique_retry_chains=len({(c['tool_log'], c['accepted_line']) for c in cases}),
        cases_with_motion_before_acceptance=sum(bool(c['motion_before_acceptance']) for c in cases))
    (ROOT / 'rejected_continue_followups.json').write_text(json.dumps(
        dict(summary=summary, cases=cases), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for c in cases:
        p = Path(c['tool_log'])
        print(json.dumps(dict(id=c['case_id'], task=p.parts[-5], seed=p.parts[-4], line=c['rejected_line'],
            old=c['prior_state']['completion_condition'],
            attempted={k:c['rejected']['arguments'].get(k) for k in ('prompt','completion_condition')},
            next_decision=c['accepted_state']['decision'], new=c['accepted_state']['completion_condition'],
            same_stage=c['same_stage'], next_error=c['next_call'].get('error'),
            motion_before=[(m['tool'],m['physical_actions']) for m in c['motion_before_acceptance']],
            motion_after=[(m['tool'],m.get('arguments'),m['physical_actions']) for m in c['motion_after_acceptance_until_next_review']]),ensure_ascii=False))


if __name__ == '__main__':
    main()
