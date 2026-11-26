"""Prepare paired final manifests from recorded task/reset identities. Never launch."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime'),str(HERE/'backends/cap')]
from stage_bridge.final_config import VARIANTS, validate, validate_pairs


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare(reference, root, *, policy=None, provider=None, auth_home=None,
            variants=('V','H'), tasks=None, seeds=None, lanes=2,
            ablation_tasks=None, ablation_seeds=None):
    reference=Path(reference).resolve();root=Path(root).resolve()
    original=json.loads(reference.read_text())
    selected=[deepcopy(e.get('spec',e)) for e in original['episodes']]
    if tasks is not None:selected=[e for e in selected if str(e['task']) in set(tasks)]
    if seeds is not None:selected=[e for e in selected if e['seed'] in seeds]
    if not selected:raise ValueError('No selected task/seed identities')
    if any(e.get('execution_mode')!='vla_only' for e in selected):
        raise ValueError('Use an existing VLA Subgoal reference; classification is preserved')
    if len(set(variants))!=len(variants) or set(variants)-VARIANTS.keys():
        raise ValueError('Distinct supported variants required')
    simulator='libero' if selected[0].get('simulator')=='libero_pro' else 'robocasa'
    if policy not in (None,'xr1','rldx') or (policy and simulator!='robocasa'):
        raise ValueError('Policy override selects the RoboCasa XR-1/RLDX adapter only')
    lanes=min(lanes,len(selected))
    if not 1<=lanes<=3:raise ValueError('Use one through three lanes on one GPU')
    root.mkdir(parents=True,exist_ok=False)
    identities=[];all_groups=[]
    for variant in variants:
        vla,codes,labels=VARIANTS[variant]
        group=root/variant;group.mkdir()
        records=[]
        for i,source in enumerate(selected):
            if variant=='H_unclassified' and (
                (ablation_tasks is not None and str(source['task']) not in set(ablation_tasks)) or
                (ablation_seeds is not None and source['seed'] not in ablation_seeds)):
                continue
            s=deepcopy(source)
            task=s.get('task_label',s.get('task_set','tasks'))
            pair=f'{task}/{s["task"]}/seed_{s["seed"]}'
            key=f'{variant}/{pair}'
            output=group/'episodes'/pair/'attempt_001'
            s.update(framework='Final_Version',final_protocol='existing-subgoal-paired/v1',
                legacy_framework_revision=source.get('framework_revision'),
                framework_revision='final-existing-subgoal-20261029/v1',
                final_variant=variant,classification_enabled=labels,code_as_policy=codes,vla_enabled=vla,
                execution_mode='codes_only' if not vla else 'hybrid' if codes else 'vla_only',hybrid_guidance=False,
                pair_id=pair,key=key,lane=i%lanes,source_root=str(HERE/'runtime'),
                backend_root=str(HERE/'backends/cap'),protocol_core=str(HERE/'runtime/subgoal_core'),
                output_dir=str(output),source_reference=str(reference),
                external_memory=False,task_memory=False,global_memory=False,online_rsi=False,
                memory_access='disabled',configuration_role='ablation',
                classification_semantic_error_policy='record_only/v1',
                native_score_is_not_progress_belief=True,
                review_window_public_protocol='no-remaining-actions/v2')
            s.pop('planner_window_remaining_actions',None)
            s.pop('resume_from',None)
            s.pop('continue_declaration_protocol',None)
            if policy:s['policy']=policy
            if provider:s['planner_provider']=provider
            if s.get('planner_provider')=='chatgpt_subscription':
                s['auth_home']=str(Path(auth_home or original.get('auth_home') or os.environ.get('STAGE_CODEX_HOME', str(Path.home()/'.codex'))).resolve())
            s['model_request_authorized']=False
            s['launch_authorized']=False
            if variant!='V':s.pop('reused_result_file',None)
            if s['policy']=='rldx':s['policy_history_impl']='transition-owned/rldx-v1'
            if not labels:s['subgoal_status_protocol']=None
            if simulator=='libero':
                from subgoal_core.observations import configuration, LIBERO_STAGE_IMAGES
                from stage_bridge.libero_budget import apply_task_budget
                s.update(libero_pro_source=str(HERE/'backends/cap/libero/dependencies/libero_pro'),
                    bddl_root=str(HERE/'backends/cap/libero/task_assets/bddl_files'),
                    init_states_root=str(HERE/'backends/cap/libero/task_assets/init_files'),
                    # User correction 2026-11-02: no minimum before failure,
                    # and no physical actions after the Planner finishes.
                    minimum_vla_actions_before_stuck=0, post_agent_wait_steps=0,
                    stopping_protocol='libero-no-minimum-no-post-wait/v1')
                s['planner'] = dict(s['planner'], max_turns=240)
                s['planner_observation_profile'] = 'libero-stage-policy-plus-calibrated/v1'
                s['planner_image_window'] = configuration(
                    source['planner_image_window']['window_size'], cameras=LIBERO_STAGE_IMAGES)
                apply_task_budget(s)
                if (source.get('minimum_vla_actions_before_stuck', 0) != 0
                        or source.get('post_agent_wait_steps', 0) != 0
                        or source['planner'].get('max_turns') != 240
                        or any(source.get(k) != s[k] for k in ('physical_action_budget', 'env_horizon',
                            'planner_observation_profile', 'planner_image_window'))):
                    s.pop('reused_result_file', None)
                if s.get('language_profile_path'):
                    s['language_profile_path']=str(HERE/'profiles/libero_benchmark_native.md')
                if s.get('policy_rtc'):
                    s['rtc_audit_dir']=str(group/'control/policy')
                (output/'private').mkdir(parents=True,exist_ok=True)
                (output/'public').mkdir(exist_ok=True)
            else:
                restrictions=json.loads(Path(source['restrictions_file']).read_text())
                write(group/'restrictions.json',restrictions)
                s['restrictions_file']=str(group/'restrictions.json')
                s['memory_dir']=str(output/'disabled_memory')
            validate(s)
            write(output/'episode.json',s)
            records.append(s)
            if variant==variants[0]:identities.append({'pair_id':pair,'task':s['task'],'seed':s['seed']})
        if not records:raise ValueError('Empty ablation selection')
        providers={s['planner_provider'] for s in records}
        if len(providers)!=1:raise ValueError('Select one provider explicitly for the paired campaign')
        selected_provider=providers.pop()
        manifest=dict(schema='final-subgoal-campaign/v1',framework='Final_Version',
            required_gate=original.get('required_gate', 'user_authorized_final_campaign'),
            simulator=simulator,final_variant=variant,source_root=str(HERE/'runtime'),
            expected_episodes=len(records),lanes=lanes,policy=records[0]['policy'],
            planner_provider=selected_provider,auth_home=records[0].get('auth_home'),
            launch_authorized=False,model_request_authorized=False,prepared_from=str(reference),diagnostic=False,
            online_rsi=False,external_memory=False,episodes=records,
            classification_enabled=labels,selection_policy='one_valid_attempt; retain_valid_failures',
            automatic_provider_fallback=False,automatic_episode_retries=False)
        write(group/'campaign.json',manifest)
        deployment=dict(scheduler='local',result_tag='subscription' if selected_provider=='chatgpt_subscription' else 'api',
            planner_provider=selected_provider,auth_home=manifest['auth_home'],gpu_count=1,
            concurrency=lanes,transport_root=str(HERE/'transport'),uid=os.getuid(),gid=os.getgid(),
            port_base=28341,port_stride=10,cpu=20,memory_mib=98304,
            wall_timeout_seconds=10800*((len(records)+lanes-1)//lanes))
        write(group/'deployment.json',deployment)
        if simulator=='libero':
            for s in records:
                control=group/'control/episodes'/s['key']
                write(control/'campaign.json',dict(manifest,expected_episodes=1,episodes=[s]))
            old=reference.parent/'control/feedback_cpu_simulation.json'
            if old.is_file():
                receipt=json.loads(old.read_text())
                receipt['reused_evidence_from']=str(old)
                receipt['scope']='Unchanged native feedback implementation only; new Hybrid path checked separately.'
                write(group/'control/feedback_cpu_simulation.json',receipt)
        else:
            (group/'control').mkdir(exist_ok=True)
        all_groups.append(str(group/'campaign.json'))
    combined=root/'paired';combined.mkdir()
    by_pair={}
    for file in all_groups:
        for s in json.loads(Path(file).read_text())['episodes']:
            by_pair[(s['pair_id'],s['final_variant'])]=s
    rng=random.Random(20261029);schedule=[]
    for item in identities:
        order=[arm for arm in variants if (item['pair_id'],arm) in by_pair]
        rng.shuffle(order)
        schedule.append(dict(item,arm_order=order))
    ordered=[by_pair[(block['pair_id'],arm)] for block in schedule for arm in block['arm_order']]
    validate_pairs(ordered)
    paired_manifest=dict(manifest,final_variant='paired',episodes=ordered,
        expected_episodes=len(ordered),scheduling='blocked-task-seed-arm-order/v1',
        pair_schedule=schedule,arm_manifests=all_groups,classification_enabled='per_episode',
        launch_unit='paired; do not also launch arm roots')
    write(combined/'campaign.json',paired_manifest)
    deployment['wall_timeout_seconds']*=len(variants)
    write(combined/'deployment.json',deployment)
    (combined/'control').mkdir(exist_ok=True)
    if simulator=='libero':
        for s in ordered:
            write(combined/'control/episodes'/s['key']/'campaign.json',
                  dict(paired_manifest,expected_episodes=1,episodes=[s]))
        old=reference.parent/'control/feedback_cpu_simulation.json'
        if old.is_file():
            receipt=json.loads(old.read_text());receipt['reused_evidence_from']=str(old)
            receipt['scope']='Unchanged native feedback implementation only; Hybrid checks are separate.'
            write(combined/'control/feedback_cpu_simulation.json',receipt)
    write(root/'pairing.json',dict(protocol='existing-subgoal-paired/v1',reference=str(reference),
        codes_only_arm='C' if 'C' in variants else None,groups=all_groups,paired_identities=len(identities),randomization_seed=20261029,
        schedule=schedule,launch_manifest=str(combined/'campaign.json'),
        execution='Each task/seed block stays on one lane; arms execute in the recorded randomized order.',
        jobs_submitted=0,model_requests=0))
    return all_groups


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--policy',choices=['xr1','rldx'])
    p.add_argument('--provider',choices=['openai_official_budgeted','openai_official_budgeted_v2','chatgpt_subscription'])
    p.add_argument('--auth-home',type=Path)
    p.add_argument('--variants',nargs='+',default=['V','H'])
    p.add_argument('--tasks',nargs='+');p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--ablation-tasks',nargs='+')
    p.add_argument('--ablation-seeds',type=int,nargs='+')
    p.add_argument('--lanes',type=int,default=2)
    a=p.parse_args()
    print(json.dumps(prepare(a.reference,a.root,policy=a.policy,provider=a.provider,
        auth_home=a.auth_home,variants=a.variants,tasks=a.tasks,seeds=a.seeds,lanes=a.lanes,
        ablation_tasks=a.ablation_tasks,ablation_seeds=a.ablation_seeds),indent=2))
