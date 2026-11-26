"""Run the STAGE RoboCasa365 evaluation harness with private, observational tool metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

from research.common import append_json, read_json, rpc, write_json

VLA_TOOLS = {'vla_skill', 'vla_arm'}
ANALYTIC_TOOLS = {'move_to', 'move_delta', 'rotate_pitch', 'set_gripper', 'release',
                  'scripted_grasp', 'navigate_to', 'move_base', 'execute_action'}


def validate_reference_behavior(spec: dict) -> None:
    """Keep language-routing and planner-guidance experiments out of the baseline."""
    if spec.get('vla_instruction_scope', 'unrestricted') == 'subgoal_only' and spec.get('vla_prompt_mode') != 'provided':
        raise ValueError('subgoal_only requires vla_prompt_mode=provided')
    if spec.get('configuration_role', 'baseline') == 'ablation':
        return
    if spec.get('decision_reporting_profile', 'upstream') != 'upstream':
        raise ValueError('Modified decision reporting requires configuration_role=ablation')
    if spec.get('vla_prompt_mode', 'task_language') != 'task_language' or spec.get('hybrid_guidance', False):
        raise ValueError('Baseline requires upstream task_language routing and no added hybrid guidance; label an intentional variant configuration_role=ablation')
    if spec.get('mode') == 'restricted' and spec.get('prompt_profile') != 'upstream_with_restricted_edits':
        raise ValueError('Use a fresh upstream_with_restricted_edits baseline manifest; older simplified-prompt episodes are historical diagnostics')


def instrument_toolkit(toolkit, private: Path, endpoint: str, *, feedback_protocol=None,
                       execution_trace=None):
    """Retain native scheduling while recording every invocation and actual steps."""
    original_execute = toolkit.execute_tool
    counts = Counter()
    lock = threading.Lock()
    current = threading.local()

    def measured(tool):
        def execute(**inputs):
            source = 'vla' if tool.name in VLA_TOOLS else 'analytic' if tool.name in ANALYTIC_TOOLS else None
            current.executed = True
            current.before = rpc(endpoint, 'research.action_count')['actions']
            if source:
                rpc(endpoint, 'env.set_action_source', (source, tool.name))
            started = time.perf_counter()
            try:
                return tool(**inputs)
            finally:
                try:
                    current.after = rpc(endpoint, 'research.action_count')['actions']
                    current.execution_seconds = time.perf_counter() - started
                finally:
                    if source:
                        rpc(endpoint, 'env.set_action_source', ('unattributed', 'outside_motion_tool'))

        return tool.with_handler(execute)

    for name, tool in list(toolkit._tools.items()):
        toolkit._tools[name] = measured(tool)

    def invoke(name, inputs):
        current.executed = False
        current.before = current.after = None
        current.execution_seconds = None
        started = time.perf_counter()
        result = None
        error = None
        try:
            result = original_execute(name, inputs)
            from robots.robocasa.restrictions import get_restrictions
            from research.feedback_delivery import record_delivery
            flags = get_restrictions()
            if flags.enabled:
                record_delivery(result.data, with_feedback=not flags.hide_task_feedback,
                                benchmark='robocasa', private=private, tool=name,
                                protocol=feedback_protocol)
            return result
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            data = result.to_dict() if result is not None else {}
            source = 'vla' if name in VLA_TOOLS else 'analytic' if name in ANALYTIC_TOOLS else None
            with lock:
                counts[name] += 1
                append_json(private / 'tool_calls.jsonl', {
                    'tool': name, 'arguments': inputs, 'source': source,
                    'handler_executed': current.executed,
                    'action_start': current.before, 'action_end': current.after,
                    'physical_actions': current.after - current.before if current.after is not None and current.before is not None else 0,
                    'elapsed_seconds': time.perf_counter() - started,
                    'handler_seconds': current.execution_seconds,
                    'error': error or data.get('error'), 'result_fields': sorted(data),
                })
                if execution_trace is not None and current.before is not None and current.after is not None:
                    execution_trace.record(toolkit, tool=name, before=current.before,
                        after=current.after, result=result, error=error)
                physical = sum(counts[key] for key in VLA_TOOLS | ANALYTIC_TOOLS)
                vla_calls = sum(counts[key] for key in VLA_TOOLS)
                write_json(private / 'tool_metrics.json', {
                    'calls_by_tool': dict(counts), 'vla_tool_calls': vla_calls,
                    'analytic_tool_calls': physical - vla_calls,
                    'vla_tool_call_fraction': vla_calls / physical if physical else None,
                    'fraction_denominator': 'all motion-tool invocations, including validation errors; separate from environment action steps',
                })

    toolkit.execute_tool = invoke
    return toolkit


def native_cli_argv(spec: dict, env_endpoint: str, vla_endpoint: str) -> list[str]:
    """Build the native CLI invocation shared by execution and argument regression tests."""
    output = Path(spec['output_dir'])
    memory_profile = spec.get('memory_profile', 'hf')
    argv = ['rpent', '--robot', 'robocasa', '--planner', 'codex', '--model', 'gpt-6-astra',
            '--reasoning-effort', spec['planner']['reasoning_effort'], '--task-name', spec['task'], '--split', spec['split'],
            '--seed', str(spec['seed']), '--output-dir', str(output),
            '--env-endpoint', env_endpoint, '--vla-endpoint', vla_endpoint,
            '--planner-timeout-s', str(spec['planner_timeout_s']),
            '--max-turns', str(spec.get('max_turns', 120)),
            '--memory-profile', memory_profile]
    if memory_profile == 'local':
        argv += ['--memory-dir', spec.get('memory_dir', str(output / 'memory'))]
    else:
        argv += ['--memory-version', spec.get('memory_version', 'GPT_6_astra_high')]
        if spec.get('memory_repo'):
            argv += ['--memory-repo', spec['memory_repo']]
    argv += ['--no-auto-merge-memory']
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--env-endpoint', required=True)
    parser.add_argument('--vla-endpoint', required=True)
    args = parser.parse_args()
    spec = read_json(args.episode)
    from stage_bridge.final_config import capabilities
    vla, codes, labels = capabilities(spec)
    validate_reference_behavior(spec)
    if (type(spec.get('code_as_policy')) is not bool
            or spec.get('review_window_actions') != 40
            or spec.get('review_window_chunks') != 5
            or spec.get('native_action_steps') != 8
            or spec.get('vla_max_chunks') != 5
            or spec.get('vla_default_chunks') != 5
            or spec.get('vla_chunk_budget_mode') != 'framework_observation_cadence/v1'
            or spec.get('policy_history_protocol') != 'explicit-subgoal-transitions/v1'):
        raise ValueError('Use the c5 manifest: 5 x 8 actions and explicit Subgoal history transitions')
    os.environ['STAGE_CODE_AS_POLICY'] = str(int(spec['code_as_policy']))
    os.environ['STAGE_VARIANT'] = spec['final_variant']
    from stage_bridge.codes_guidance import validate_profile
    os.environ['STAGE_CODES_PROMPT_PROFILE'] = validate_profile(spec)
    os.environ['STAGE_VLA_ENABLED'] = str(int(vla))
    os.environ['STAGE_POLICY_HISTORY_PROTOCOL'] = spec['policy_history_protocol']
    output = Path(spec['output_dir'])
    public, private = output / 'public', output / 'private'
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    from research.subgoal_images import configure as configure_images
    configure_images(spec, private, public)
    os.environ.update(STAGE_PUBLIC_OUTPUT_DIR=str(public), STAGE_PRIVATE_OUTPUT_DIR=str(private),
                      STAGE_ATTACH_ONLY='1', STAGE_ACTION_PROVENANCE='1',
                      STAGE_VLA_POLICY=spec['policy'],
                      STAGE_RESTRICTED='1' if spec['mode'] == 'restricted' else '0')
    os.environ['STAGE_VLA_PROMPT_MODE'] = spec.get('vla_prompt_mode', 'task_language')
    os.environ['STAGE_VLA_INSTRUCTION_SCOPE'] = spec.get('vla_instruction_scope', 'unrestricted')
    os.environ['STAGE_HYBRID_GUIDANCE'] = '1' if spec.get('hybrid_guidance', False) else '0'
    os.environ['STAGE_DECISION_REPORTING'] = spec.get('decision_reporting_profile', 'upstream')
    os.environ['STAGE_PLANNER_OBSERVATION_PROFILE'] = spec.get(
        'planner_observation_profile', 'native-three-view/v1')
    from robots.robocasa.tools import planner_observation_profile
    planner_observation_profile()
    # Framework-owned observation cadence; public VLA calls have no chunk-count argument.
    os.environ['VLA_MAX_CHUNKS'] = str(spec['vla_max_chunks'])
    os.environ['VLA_ACTION_STEPS_PER_CHUNK'] = str(spec.get('native_action_steps', 8))
    os.environ['VLA_SETTLE_PATIENCE'] = str(spec.get('vla_settle_patience', 999))
    os.environ['STAGE_VIDEO_RESOLUTION'] = str(spec.get('video_export_resolution', 512))
    from robots.robocasa.action_interface import vla_instruction_scope, vla_prompt_mode, hybrid_guidance_enabled, decision_reporting_profile
    vla_prompt_mode()
    vla_instruction_scope()
    hybrid_guidance_enabled()
    decision_reporting_profile()
    if spec.get('restrictions_file'):
        os.environ['STAGE_RESTRICTIONS_FILE'] = spec['restrictions_file']
    else:
        os.environ.pop('STAGE_RESTRICTIONS_FILE', None)
    os.environ['STAGE_RECORD_PROMPT_DIFF'] = '1' if spec['mode'] == 'restricted' or decision_reporting_profile() != 'upstream' else '0'
    # This independent revision changes only the requested cadence and
    # explicit Subgoal history ownership plus the corresponding prompt rules.
    cli = importlib.import_module('rpent.cli.main')
    get_toolkit = cli.get_toolkit
    def build_toolkit(*a, **kw):
        from research.subgoal_review import install as install_subgoals
        from research.review_window import install as install_window
        toolkit = get_toolkit(*a, **kw)
        if labels:
            install_subgoals(toolkit, private, code_as_policy=codes,
                            status_protocol=spec.get('subgoal_status_protocol'), vla_enabled=vla)
        else:
            from stage_bridge.unclassified import install as install_unclassified
            install_unclassified(toolkit, private)
        install_window(toolkit, spec['review_window_actions'], private)
        if spec.get('resume_from'):
            from research.episode_resume import restore_toolkit
            restore_toolkit(toolkit, spec, private)
            from research.subgoal_images import _window as images
            if images is not None:
                images.restore(spec['resume_from'])
        from stage_bridge.execution_trace import ExecutionTrace
        toolkit = instrument_toolkit(toolkit, private, args.env_endpoint,
            feedback_protocol=spec.get('feedback_protocol'),
            execution_trace=ExecutionTrace(private, variant=spec['final_variant'],
                                           classification_enabled=labels))
        from research.planner_transport import set_allowed_tools
        set_allowed_tools(toolkit.list_tools())
        return toolkit

    cli.get_toolkit = build_toolkit
    argv = native_cli_argv(spec, args.env_endpoint, args.vla_endpoint)
    sys.argv = argv
    write_json(private / 'runtime_identity.json', {
        **spec, 'cli': argv, 'public_output': str(public), 'private_output': str(private),
        'decision_reporting_profile': decision_reporting_profile(),
        'native_tools': True, 'native_action_steps': 8,
        'xr1_state_compatibility': spec['policy'] == 'xr1',
    })
    from research.planner_transport import planner_transport
    try:
        with planner_transport(private, public_dir=public, enforce_boundary=spec['mode'] == 'restricted',
                               reasoning_effort=spec['planner']['reasoning_effort'],
                               direct_hosts=[urlparse(args.env_endpoint).hostname, urlparse(args.vla_endpoint).hostname]):
            return cli.main()
    finally:
        from research.metrics import preserve_native_results
        preserve_native_results(spec)


if __name__ == '__main__':
    raise SystemExit(main())
