"""Private episode aggregation using native artifacts and actual transport attempts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import time

from research.common import read_json, write_json


def json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import json
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed process can leave one incomplete final line. Its
                # missing evidence is reported, never converted to success.
                continue
    return records


def preserve_native_results(spec: dict) -> dict:
    output = Path(spec['output_dir'])
    private = output / 'private'
    transcript_name = f"transcript_{spec['task']}_{spec['split']}_s{spec['seed']}.json"
    transcript = next((path for path in (private / transcript_name, output / transcript_name) if path.exists()), None)
    try:
        record = read_json(transcript) if transcript else {}
    except ValueError:
        record = {}
    native_result = next((path for path in (private / 'result.json', output / 'result.json') if path.exists()), None)
    if native_result and read_json(native_result).get('schema') != 'stage.restricted-result/v1':
        write_json(private / 'native_cli_result.json', read_json(native_result))
        if native_result.parent == private:
            native_result.unlink()
    raw_paths = {}
    for field in ('output_path', 'raw_stream_path', 'last_message_path'):
        source = Path(record.get('stats', {}).get(field, ''))
        if source.is_file():
            target = private / 'raw_planner_outputs' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            raw_paths[field] = str(target)
    if transcript:
        target = private / 'raw_planner_outputs' / transcript.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if transcript.resolve() != target.resolve():
            shutil.copyfile(transcript, target)
        raw_paths['transcript'] = str(target)
    if not raw_paths:
        tag = f"codex_{spec['task']}_{spec['split']}_s{spec['seed']}.txt"
        for directory in (private, output):
            for suffix in ('', '.stream.jsonl', '.last'):
                candidate = directory / (tag + suffix)
                if candidate.is_file():
                    target = private / 'raw_planner_outputs' / candidate.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(candidate, target)
                    raw_paths[candidate.name] = str(target)
    write_json(private / 'planner_metrics.json', {
        'elapsed_seconds': record.get('elapsed_s'), 'stats': record.get('stats', {}),
        'error': record.get('error'), 'finish': record.get('finish'), 'raw_outputs': raw_paths,
        'transcript_available': transcript is not None,
    })
    return record


def finalize_metrics(spec: dict, *, result: dict | None = None, error: str | None = None) -> dict:
    """Write metrics for completed or interrupted attempts; keep missing values explicit."""
    output = Path(spec['output_dir'])
    private = output / 'private'
    environment = read_json(private / 'environment_metrics.json') if (private / 'environment_metrics.json').exists() else {}
    if not (private / 'planner_metrics.json').exists():
        preserve_native_results(spec)
    planner = read_json(private / 'planner_metrics.json')
    done = read_json(private / 'planner_done.json') if (private / 'planner_done.json').exists() else {}
    tools = json_lines(private / 'tool_calls.jsonl')
    vla = json_lines(private / 'vla_requests.jsonl')
    transport = json_lines(private / 'transport/request_records.jsonl')
    rejected = [r for r in transport if r.get('outcome') == 'request_transform_rejected']
    attempts = [r for r in transport if r.get('outcome') != 'request_transform_rejected']
    measured_usage = [r['usage'] for r in attempts if r.get('usage')]
    stats = planner.get('stats', {})
    wall_seconds = planner.get('elapsed_seconds')
    if wall_seconds is None and (private / 'planner_started.json').exists() and done.get('finished_at'):
        wall_seconds = done['finished_at'] - read_json(private / 'planner_started.json')['started_at']
    if measured_usage:
        token_usage = {
            'input_tokens': sum(u.get('input_tokens', 0) for u in measured_usage),
            'output_tokens': sum(u.get('output_tokens', 0) for u in measured_usage),
            'cached_input_tokens': sum(u.get('input_tokens_details', {}).get('cached_tokens', 0) for u in measured_usage),
            'reasoning_output_tokens': sum(u.get('output_tokens_details', {}).get('reasoning_tokens', 0) for u in measured_usage),
            'source': 'sum of per-attempt provider usage, including retried or incomplete attempts with reported usage',
            'attempts_with_usage': len(measured_usage), 'attempts_without_reported_usage': len(attempts) - len(measured_usage),
        }
    else:
        token_usage = {'input_tokens': stats.get('total_input_tokens'), 'output_tokens': stats.get('total_output_tokens'),
                       'cached_input_tokens': stats.get('total_cached_input_tokens'),
                       'reasoning_output_tokens': stats.get('total_reasoning_output_tokens'),
                       'source': 'native cumulative SDK usage fallback; no per-attempt provider usage available',
                       'attempts_with_usage': 0, 'attempts_without_reported_usage': len(attempts)}
    budget = environment.get('budget_exhausted') or done.get('reason') == 'action_budget'
    finish = planner.get('finish')
    agent_finished = bool(finish) or any(t['tool'] == 'finish' and not t.get('error') for t in tools)
    failure = error or planner.get('error') or ((result or {}).get('valid') is False)
    if budget:
        termination, detail = 'budget_exhausted', 'physical_action_budget'
    elif planner.get('error') == f"Codex SDK timed out after {spec.get('planner_timeout_s')}s":
        termination, detail = 'budget_exhausted', 'planner_wall_budget'
    elif stats.get('turns_used', 0) >= spec.get('max_turns', 120) and not agent_finished:
        termination, detail = 'budget_exhausted', 'planner_turn_budget'
    elif failure or done.get('exit_code', 0) != 0:
        termination, detail = 'error', error or planner.get('error') or done.get('reason') or 'planner_process_failed'
    elif agent_finished:
        termination, detail = 'agent_done', 'native_finish_tool'
    else:
        termination, detail = 'error', 'planner_exited_without_finish_or_budget_evidence'
    sources = environment.get('action_steps_by_source', {})
    motion_calls = Counter(t.get('source') for t in tools if t.get('source') in {'vla', 'analytic'})
    motion_total = sum(motion_calls.values())
    record = {
        'schema': 'stage.restricted-metrics/v1', 'key': spec['key'],
        'diagnostic': bool(spec.get('diagnostic', False)),
        'benchmark': not bool(spec.get('diagnostic', False)),
        'task': spec['task'], 'task_set': spec.get('task_set'), 'seed': spec['seed'],
        'policy': spec['policy'], 'mode': spec['mode'], 'source_revision': spec.get('source_revision'),
        'valid': result is not None and termination in {'agent_done', 'budget_exhausted'},
        'success': environment.get('official_process_any'),
        'success_definition': 'official predicate true at any executed environment step; evaluator only',
        'official_process_any': environment.get('official_process_any'),
        'official_terminal': environment.get('official_terminal'),
        'first_success_step': environment.get('first_success_step'),
        'final_predicate_states': environment.get('final_predicate_states', {}),
        'termination_reason': termination, 'termination_detail': detail,
        'actions': environment.get('actions', 0), 'action_budget': spec['action_budget'],
        'action_steps_by_source': sources, 'vla_action_fraction': environment.get('vla_action_fraction'),
        'zero_command_steps': environment.get('zero_command_steps', 0),
        'vla_tool_calls': motion_calls['vla'], 'analytic_tool_calls': motion_calls['analytic'],
        'vla_tool_call_fraction': motion_calls['vla'] / motion_total if motion_total else None,
        'tool_calls': tools,
        'vla_forwards': len(vla), 'vla_inference_seconds': sum(r.get('inference_seconds', 0) for r in vla),
        'vlm_request_attempts_including_retries': len(attempts),
        'vlm_completed_requests': sum(r.get('outcome') == 'completed' for r in attempts),
        'vlm_boundary_rejected_requests': len(rejected), 'vlm_token_usage': token_usage,
        'planner_wall_seconds': wall_seconds,
        'episode_wall_seconds': (time.time() - read_json(private / 'episode_started.json')['time'])
        if (private / 'episode_started.json').exists() else None,
        'episode_wall_definition': 'environment startup through final aggregation; shared model load and scheduler queue recorded per lane',
        'vlm_request_wall_seconds': sum(r.get('timing_seconds', {}).get('request_seconds', 0) for r in attempts),
        'vlm_post_upload_response_seconds': sum(r.get('timing_seconds', {}).get('post_upload_response_seconds', 0) for r in attempts),
        'vlm_timing_note': 'client request/response time includes transfer, queue, prefill and generation; pure server inference is unavailable',
        'planner_turns': stats.get('turns_used'), 'planner_finish': finish,
        'raw_planner_outputs': planner.get('raw_outputs', {}),
        'raw_model_input_directory': str(private / 'model_inputs'),
        'transport_request_ledger': str(private / 'transport/request_records.jsonl'),
        'dense_actions': str(private / 'actions.jsonl'),
        'physical_success': None, 'physical_metric_status': 'not_evaluated',
    }
    write_json(private / 'metrics.json', record)
    return record
