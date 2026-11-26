"""Run the continuous Subgoal protocol on this framework's native LIBERO tools."""
import argparse
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from research.common import append_json, write_json
from research.libero_bootstrap import prepare


def instrument_handler_time(toolkit):
    """Time actual tool dispatch separately from outer observation/log handling."""
    current = threading.local()
    for name, declaration in list(toolkit._tools.items()):
        def timed(_declaration=declaration, **inputs):
            started = time.perf_counter()
            current.handler_executed = True
            try:
                return _declaration(**inputs)
            finally:
                current.handler_seconds = time.perf_counter() - started
        toolkit._tools[name] = declaration.with_handler(timed)
    return current


def configure_planner(spec,public,private):
    if spec['planner_provider'] == 'chatgpt_subscription':
        os.environ['AGENTIC_DESKTOP_CODEX_HOME'] = spec['auth_home']
    restrictions=private/'planner_restrictions.json'
    write_json(restrictions,dict(restrict_file_access=True,disable_task_memory=True,
        disable_global_memory=True,hide_task_feedback=False,disable_success_stop=False))
    os.environ.update(STAGE_CODE_AS_POLICY=str(int(spec['code_as_policy'])),STAGE_PLANNER_PROVIDER=spec['planner_provider'],
        STAGE_PUBLIC_OUTPUT_DIR=str(public),STAGE_PRIVATE_OUTPUT_DIR=str(private),
        STAGE_TRANSPORT_RUNTIME=str(Path(spec['source_root']).parent/'transport'),
        STAGE_RESTRICTED='0',STAGE_RESTRICTIONS_FILE=str(restrictions))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--env-endpoint',required=True)
    parser.add_argument('--vla-endpoint',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    spec=prepare(args.episode)
    from stage_bridge.final_config import capabilities
    vla, codes, labels = capabilities(spec)
    if (spec['planner_provider'] not in {'openai_official_budgeted','openai_official_budgeted_v2','chatgpt_subscription'}
            or spec['external_memory'] is not False or spec['review_window_actions']!=40):
        raise ValueError('Use the selected provider and paired no-Memory manifest')
    output=Path(spec['output_dir']); public=output/'public'; private=output/'private'
    public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
    configure_planner(spec,public,private)
    os.environ['STAGE_VLA_ENABLED'] = str(int(vla))
    os.environ['STAGE_VARIANT'] = spec['final_variant']
    from rpent.utils.logging import init_output_dir
    init_output_dir(private)
    from rpent.utils.rpc import make_rpc_client
    from rpent.dashboard.events import NullDashboardEventSink
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.toolkit import LiberoToolkit
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from stage_bridge.libero import EpisodeFiles, install, prompt_pair
    from subgoal_core.observations import configure
    from subgoal_core.prompts import (
        LIBERO_PROMPT_PROTOCOL, LEGACY_LIBERO_PROMPT_PROTOCOL, SHARED_PROMPT_PROFILE,
    )
    from research.planner_transport import set_allowed_tools, planner_transport
    from rpent.planner.base import build_planner

    env_rpc=make_rpc_client(args.env_endpoint); vla_rpc=make_rpc_client(args.vla_endpoint)
    toolkit=None
    error=None; started=time.time()
    try:
        meta=env_rpc.call('env.get_env_meta')
        env=LiberoEnvClient(env_rpc,expected_meta=meta,return_all_frames=True)
        feedback_protocol = spec.get('task_feedback_protocol')
        if feedback_protocol:
            from stage_bridge.libero_feedback import PROTOCOL
            if feedback_protocol != PROTOCOL:
                raise ValueError(f'Unsupported task feedback: {feedback_protocol}')
            env.get_task_feedback = lambda: env_rpc.call('env.get_task_feedback')
        model=Pi05VLAClient(vla_rpc,embodiment='libero')
        predict=model.predict
        def observed_predict(obs,options=None):
            begin=time.perf_counter()
            before=env_rpc.call('research.action_count')['actions']
            try:
                if spec.get('policy_rtc'):
                    from stage_bridge.libero_rtc import request_options
                    options = request_options(options, episode_key=spec['key'],
                        state=toolkit.subgoal_review.state, action_count=before,
                        prompt=obs['task_descriptions'])
                    if getattr(toolkit, '_analytic_since_vla', False):
                        options['rtc']['reset_after_analytic'] = True
                actions=predict(obs,options)
                toolkit._analytic_since_vla = False
            except Exception as exc:
                write_json(private/'policy_error.json',dict(error=f'{type(exc).__name__}: {exc}',
                    action=before,at=time.time()))
                raise
            import numpy as np
            finite=bool(np.isfinite(actions).all())
            append_json(private/'vla_requests.jsonl',dict(prompt=obs['task_descriptions'],
                options=options,actions=len(actions),action_start=before,finite=finite,elapsed_seconds=time.perf_counter()-begin))
            if np.asarray(actions).shape!=(5,7) or not finite:
                raise ValueError('Native Pi0.5 must return five finite 7D actions')
            return actions
        model.predict=observed_predict
        events=NullDashboardEventSink()
        toolkit=LiberoToolkit(runtime_kwargs=dict(env=env,model=model,sam3_client=None),
            dashboard_events=events,memory=EpisodeFiles(public),state_output_dir=public)
        configure(spec,private,public)
        prompt_protocol=spec.get('prompt_protocol',LEGACY_LIBERO_PROMPT_PROTOCOL)
        minimum = spec.get('minimum_vla_actions_before_stuck', 0)
        if type(minimum) is not int or not 0 <= minimum <= spec['physical_action_budget']:
            raise ValueError('Minimum VLA effort must fit the physical action budget')
        from stage_bridge.libero_effort import install as install_effort, guidance as effort_guidance
        install_effort(toolkit, private, minimum=minimum,
                       action_count=lambda: env_rpc.call('research.action_count')['actions'],
                       code_as_policy=codes, vla_enabled=vla)
        install(toolkit,private,prompt_protocol=prompt_protocol,
                status_protocol=spec.get('subgoal_status_protocol'),
                code_as_policy=codes, classification_enabled=labels, vla_enabled=vla)
        from stage_bridge.libero_budget import install_final_prefix
        from research import review_window
        install_final_prefix(env, budget=spec['physical_action_budget'],
                             action_count=lambda: review_window._window.total)
        from stage_bridge.execution_trace import ExecutionTrace
        trace = ExecutionTrace(private, variant=spec['final_variant'], classification_enabled=labels)
        timing = instrument_handler_time(toolkit)
        execute=toolkit.execute_tool
        def measured(name,inputs):
            timing.handler_executed = False
            timing.handler_seconds = None
            before=env_rpc.call('research.action_count')['actions']; begin=time.perf_counter()
            from stage_bridge.review import ANALYTIC
            source='vla' if name=='pi0_doubled' else 'analytic' if name in ANALYTIC else None
            if source:env_rpc.call('env.set_action_source',args=(source,name))
            result=None
            tool_error=None
            try:
                result=execute(name,inputs)
                if feedback_protocol:
                    from research.feedback_delivery import record_delivery
                    try:
                        record_delivery(result.data, with_feedback=True, benchmark='libero',
                            private=private, tool=name, protocol=feedback_protocol,
                            allow_historical=name=='view_env_state' and inputs.get('step', -1)>=0)
                    except Exception as exc:
                        write_json(private/'feedback_error.json',dict(error=str(exc),tool=name,at=time.time()))
                        raise
                return result
            except Exception as exc:
                tool_error=f'{type(exc).__name__}: {exc}'
                raise
            finally:
                after=env_rpc.call('research.action_count')['actions']
                if source == 'analytic' and after > before:
                    toolkit._analytic_since_vla = True
                if source:env_rpc.call('env.set_action_source',args=('unattributed','outside_motion_tool'))
                append_json(private/'tool_calls.jsonl',dict(tool=name,arguments=inputs,source=source,
                    action_start=before,action_end=after,physical_actions=after-before,
                    handler_executed=timing.handler_executed,handler_seconds=timing.handler_seconds,
                    error=result.data.get('error') if result else None,elapsed_seconds=time.perf_counter()-begin))
                trace.record(toolkit, tool=name, before=before, after=after,
                             result=result, error=tool_error)
        toolkit.execute_tool=measured
        set_allowed_tools(toolkit.list_tools())
        write_json(private/'tool_schemas.json',[dict(name=t.name,description=t.description,parameters=t.input_schema) for t in toolkit.list_tools()])
        write_json(private/'runtime_identity.json',{**spec,'actual_native_env_meta':meta,'native_tools':True,
            'task_memory_loaded':False,'global_memory_loaded':False,'embedded_task_recipes':False})
        system,user=prompt_pair(prompt_protocol,task_feedback_protocol=feedback_protocol,
            language_profile=spec.get('language_profile'),
            language_profile_path=spec.get('language_profile_path'),
            subgoal_status_protocol=spec.get('subgoal_status_protocol'),
            code_as_policy=codes, classification_enabled=labels, vla_enabled=vla,
            planner_observation_profile=spec['planner_observation_profile'])
        if minimum:
            system += '\n\n' + effort_guidance(minimum, code_as_policy=codes, vla_enabled=vla)
        if prompt_protocol == LIBERO_PROMPT_PROTOCOL:
            write_json(private/'protocol_alignment.json',dict(
                shared_prompt_profile=SHARED_PROMPT_PROFILE,
                classification_enabled=labels,
                shared_sections=(['VLA_Rules','Workflow','Subgoal_Review',
                    'Scene_Evidence_And_Recovery','VLA_Observation_Checkpoints','Output']
                    if labels else ['Execution_Review','Output']),
                model_adapter=dict(policy='RLinf-Pi05-LIBERO-130-fullshot-SFT',
                    native_action_steps=5,chunks_per_review=8,review_window_actions=40,
                    policy_observations='native current external/wrist images and proprioception',
                    cross_call_policy_history=bool(spec.get('policy_rtc')),
                    learned_memory=False,rtc=bool(spec.get('policy_rtc')),
                    policy_rtc=spec.get('policy_rtc')),
                feedback_binding=feedback_protocol or 'native top-level terminated',
                instruction_routing='provided; exact accepted prompt' if labels else 'provided; instruction-change context',
                instruction_scope='subgoal_only',
                additional_language_profile=spec.get('language_profile'),
                subgoal_status_protocol=spec.get('subgoal_status_protocol')))
        if spec.get('language_profile') not in (None, 'none'):
            from stage_bridge.libero_language import PROFILE_PATH
            write_json(private/'language_profile.json',dict(profile=spec['language_profile'],
                source=str(spec.get('language_profile_path') or PROFILE_PATH),
                training_corpus_membership_verified=False,
                backend_instruction_rewriting=False,task_recipes_loaded=False))
        (private/'system_prompt.md').write_text(system)
        (private/'user_prompt.md').write_text(user)
        if args.smoke:
            # Real policy input/forward/simulation, without consuming a Planner turn.
            primitives=toolkit._primitives
            from research import review_window
            with review_window._window.motion():
                env_rpc.call('env.set_action_source',args=('smoke','native_policy_probe'))
                primitives._vlm_chunk(spec['task_language'])
            write_json(private/'smoke.json',dict(passed=True,physical_actions=5,model_request=False))
        else:
            with planner_transport(private,public_dir=public,enforce_boundary=True,
                    direct_hosts=[urlparse(args.env_endpoint).hostname],reasoning_effort=spec['planner']['reasoning_effort']):
                planner=build_planner('codex',output_dir=private,recipe_tag=spec['key'].replace('/','_'),
                    robot_name='libero',model=spec['planner']['model'],reasoning_effort=spec['planner']['reasoning_effort'],
                    planner_timeout_s=spec['planner']['timeout_s'],dashboard_events=events)
                result=planner.solve(system_prompt=system,user_message=user,toolkit=toolkit,
                    max_turns=spec['planner']['max_turns'])
                error=result.error
                write_json(private/'planner_result.json',dict(finish=result.finish_result,stats=result.stats,
                    error=error,wall_seconds=time.time()-started))
    finally:
        if toolkit is not None:toolkit.close()
        try:env_rpc.call('research.finalize',timeout_s=180)
        finally:env_rpc.close();vla_rpc.close()
    return int(bool(error))


if __name__=='__main__':
    raise SystemExit(main())
