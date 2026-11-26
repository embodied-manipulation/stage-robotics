"""Offline Stage diagnostics. Declaration, lexical flags and outcomes stay separate."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

STOP=set('a an the is are was were be been being to of on in at into onto and or for with from by as it its this that then'.split())
VLA={'vla_skill','vla_arm','pi0_pick','pi0_doubled'}
CODES={'move_to','move_delta','move_pose','rotate_wrist','rotate_pitch','set_gripper','release','scripted_grasp','navigate_to','move_base','execute_action'}


def read_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def words(text):return set(re.findall(r'[a-z0-9]+',text.lower()))-STOP


def similarity(a,b):
    a,b=words(a),words(b)
    return len(a&b)/len(a|b) if a|b else 0.0


def alignment(current,previous):
    condition=current.get('completion_condition','')
    old=similarity(condition,previous.get('prompt','')+' '+previous.get('completion_condition',''))
    new=similarity(condition,current.get('prompt',''))
    return dict(stage_id=current['subgoal_id'],decision=current['decision'],
        new_prompt=current.get('prompt'),new_condition=condition,
        previous_prompt=previous.get('prompt'),previous_condition=previous.get('completion_condition'),
        previous_similarity=old,new_similarity=new,condition_describes_previous_stage=old>new and old>=.3,
        transition_evidence=current.get('transition_evidence'),decision_reason=current.get('decision_reason'),
        flag_meaning='lexical screening candidate; not verified semantic misalignment')


def audit_episode(private,identity):
    decisions=[r for r in read_lines(private/'subgoal_decisions.jsonl') if r.get('type')=='planner_intervention']
    calls=read_lines(private/'tool_calls.jsonl');segments=read_lines(private/'execution_segments.jsonl')
    counts=Counter();stages={};alignments=[];previous={};accepted={};di=0
    for state in decisions:
        decision,progress=state['decision'],state['progress'];counts['accepted']+=1
        if decision in ('advance','recover'):
            counts['advance_recover_denominator']+=1
            counts['advance_recover_inconsistent']+=int(
                (decision=='advance' and progress!='complete') or
                (decision=='recover' and progress not in ('blocked','uncertain')))
        counts[f'decision_progress:{decision}/{progress}']+=1
        for flag,value in dict(advance_without_complete=decision=='advance' and progress!='complete',
            recover_while_progressing=decision=='recover' and progress=='progressing',
            continue_with_complete=decision=='continue' and progress=='complete',
            continue_with_blocked=decision=='continue' and progress=='blocked').items():counts[flag]+=int(value)
        sid=state['subgoal_id']
        if decision!='continue':
            if previous:
                stages[previous['subgoal_id']]['ending']=('advance_complete' if decision=='advance' and progress=='complete'
                    else 'advance_or_recover_noncomplete' if progress!='complete' else 'recover_complete')
                stages[previous['subgoal_id']]['ending_progress_claim']=progress
                if decision in ('advance','recover'):alignments.append(dict(identity,**alignment(state,previous)))
            stages[sid]=dict(stage_id=sid,prompt=state.get('prompt'),condition=state.get('completion_condition'),
                decision=decision,continue_count=0,actions=Counter(),tools=Counter(),history_modes=Counter(),
                zero_motion_calls=0,unknown_motion_calls=0,ending='episode_end_without_accepted_completion_transition',
                outcome_by_prompt='unknown',outcome_by_condition='unknown',outcome_source='no_local_outcome_review_supplied')
        else:stages[sid]['continue_count']+=1
        previous=state
    rejection_counts=Counter();cases=[];active=None
    for i,call in enumerate(calls):
        name=call.get('tool');error=call.get('error')
        counts['tool_calls_total']+=1
        counts['tool_rejected']+=int(bool(error))
        if name=='subgoal_review':
            counts['review_calls_total']+=1
            counts['review_rejected']+=int(bool(error))
        if identity['variant']=='C' and name in VLA:counts['c_vla_tool_call_violation']+=1
        if error:rejection_counts[f'{name}: {error}']+=1
        if name=='subgoal_review' and not error:
            if di>=len(decisions):raise ValueError(f'Accepted call has no state: {private}:{i+1}')
            state=decisions[di];di+=1
            for key in ('decision','progress','observation_step'):
                if call.get('arguments',{}).get(key)!=state.get(key):raise ValueError(f'Call/state mismatch: {private}:{i+1}')
            active=state['subgoal_id'];accepted[i]=state
        if not segments and active and name in VLA|CODES:
            stage=stages[active];n=call.get('physical_actions')
            if type(n) is not int:stage['unknown_motion_calls']+=1
            elif n>0:
                stage['actions']['VLA' if name in VLA else 'Codes']+=n;stage['tools'][name]+=n
            else:stage['zero_motion_calls']+=1
    if di!=len(decisions):raise ValueError(f'Saved accepted states not aligned: {private}')
    for row in segments:
        if identity['variant']=='C' and row.get('executor')=='VLA':counts['c_vla_violation']+=1
        stage=stages.get(row.get('subgoal_id'))
        if stage is None:continue
        n=row.get('physical_actions')
        if type(n) is not int:stage['unknown_motion_calls']+=1
        elif n>0:stage['actions'][row['executor']]+=n;stage['tools'][row['tool']]+=n
        else:stage['zero_motion_calls']+=1
        history=row.get('vla_history') or {}
        if history.get('history_mode'):stage['history_modes'][history['history_mode']]+=1
    for i,call in enumerate(calls):
        if call.get('tool')!='subgoal_review' or not str(call.get('error','')).startswith('continue cannot change'):continue
        before=max((k for k in accepted if k<i),default=None)
        after=min((k for k in accepted if k>i),default=None)
        following=next((k for k in range(i+1,len(calls)) if calls[k].get('tool')=='subgoal_review'),None)
        state=accepted.get(after);flag=next((a['condition_describes_previous_stage'] for a in alignments
            if state and a['stage_id']==state['subgoal_id']),None)
        cases.append(dict(tool_log=str(private/'tool_calls.jsonl'),rejected_line=i+1,prior_state=accepted.get(before),
            rejected=call,next_call_line=following+1 if following is not None else None,
            next_call=calls[following] if following is not None else None,accepted_line=after+1 if after is not None else None,
            accepted_call=calls[after] if after is not None else None,accepted_state=state,
            chain_id=f'{private}:{after}' if after is not None else f'{private}:unresolved',
            condition_describes_previous_stage=flag if state and state['decision']=='advance' else None,
            motion_before_acceptance=[dict(line=k+1,**calls[k]) for k in range(i+1,after if after is not None else len(calls))
                                      if (calls[k].get('physical_actions') or 0)>0]))
    rtc=[];last={}
    for row in read_lines(private/'rtc_requests.jsonl'):
        key=row.get('episode_key');goal=row.get('subgoal_id')
        changed=key in last and last[key]!=goal
        rtc.append(dict(subgoal_id=goal,request=row.get('request'),logged_reason=row.get('reset_reason'),
            effective_reason='new_subgoal' if changed else row.get('reset_reason')));last[key]=goal
    output=[]
    for stage in stages.values():
        executors=set(stage['actions']);stage['executor']=('Mixed' if len(executors)>1 else next(iter(executors)) if executors else 'Unknown_or_zero_motion')
        output.append(dict(identity,**stage))
    return dict(identity,counts=dict(counts),rejections=dict(rejection_counts),stages=output,alignments=alignments,
        rejected_continue_cases=cases,rtc_reset_interpretation=rtc)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path);p.add_argument('--episode',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inputs={}
    if a.index:
        for row in read_lines(a.index):
            if row.get('status')=='valid' and row.get('online_log'):
                private=Path(row['online_log']).parent
                inputs[str(private.resolve())]=dict(episode=str(private.parent),variant='historical_'+row['mode'],
                    policy=row['config'].get('policy'),
                    environment=('libero' if str(row.get('suite','')).startswith('libero_') else
                                 'robocasa' if row.get('suite') in ('atomic_seen','composite_seen','composite_unseen') else
                                 'unknown'))
    elif a.episode:
        private=a.episode/'private';spec=json.loads((a.episode/'episode.json').read_text())
        inputs[str(private)]=dict(episode=str(a.episode),variant=spec['final_variant'],policy=spec['policy'],
            environment='libero' if spec.get('simulator')=='libero_pro' else 'robocasa')
    else:p.error('Provide --index or --episode')
    reports=[audit_episode(Path(private),identity) for private,identity in inputs.items()]
    groups=defaultdict(Counter);stage_counts=defaultdict(Counter);rejections=defaultdict(Counter)
    for report in reports:
        key='|'.join(str(report[k]) for k in ('variant','policy','environment'))
        groups[key].update(report['counts']);rejections[key].update(report['rejections'])
        for stage in report['stages']:
            key2='|'.join(str(stage[k]) for k in ('variant','executor','decision'))
            stage_counts[key2]['stages']+=1;stage_counts[key2]['continue_count']+=stage['continue_count']
    cases=[c for r in reports for c in r['rejected_continue_cases']]
    chains={c['chain_id']:c for c in cases};advance=[c for c in chains.values() if (c.get('accepted_state') or {}).get('decision')=='advance']
    summary=dict(episodes=len(reports),groups=groups,stage_counts=stage_counts,rejections=rejections,
        rejected_continue_calls=len(cases),retry_chains=len(chains),
        next_accepted_decisions=dict(Counter((c.get('accepted_state') or {}).get('decision','unresolved') for c in chains.values())),
        retry_chain_advances=len(advance),retry_chain_advances_with_alignment_candidate=sum(c['condition_describes_previous_stage'] is True for c in advance),
        historical_99_7_basis='Accepted advance+recover declarations; excludes continue_with_complete; not visual accuracy',
        outcomes='Unreviewed prompt/condition outcomes are unknown, never inferred from progress or task success.',
        model_requests=0,simulation_rollouts=0,jobs_submitted=0)
    denominator=sum(r['counts'].get('advance_recover_denominator',0) for r in reports)
    mismatches=sum(r['counts'].get('advance_recover_inconsistent',0) for r in reports)
    summary['label_consistency']=dict(numerator=denominator-mismatches,denominator=denominator,
        fraction=(denominator-mismatches)/denominator if denominator else None)
    summary['rejection_rates']={key:dict(
        review_rejected=c['review_rejected'],review_calls=c['review_calls_total'],
        review_rate=c['review_rejected']/c['review_calls_total'] if c['review_calls_total'] else None,
        tool_rejected=c['tool_rejected'],tool_calls=c['tool_calls_total'],
        tool_rate=c['tool_rejected']/c['tool_calls_total'] if c['tool_calls_total'] else None)
        for key,c in groups.items()}
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('stage_goal_alignment',[x for r in reports for x in r['alignments']]),
                      ('stages',[x for r in reports for x in r['stages']]),('episode_audits',reports)]:
        (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    (a.output/'rejected_continue_followups.json').write_text(json.dumps(dict(summary=summary,cases=cases),ensure_ascii=False,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('episodes','rejected_continue_calls','retry_chains','next_accepted_decisions','retry_chain_advances_with_alignment_candidate')},indent=2))


if __name__=='__main__':main()
