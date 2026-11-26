"""One persistent policy per GPU lane, fresh simulator and Planner per episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from research.common import (INSTALL, PYTHON, REPO, RLDX_CHECKPOINT, RLDX_PYTHON,
                             XR1_CHECKPOINT, read_json, rpc, runtime_environment,
                             stop, wait_ready, write_json)


def lane_specs(root: Path, manifest: dict, lane: int) -> list:
    """Shared CPU startup check, run before allocating a GPU or starting services."""
    required_gate = manifest.get('required_gate')
    if not isinstance(required_gate, str) or not required_gate:
        raise ValueError('campaign.json requires an explicit nonempty required_gate; regenerate the manifest')
    if required_gate == 'rldx340_complete_and_reviewed':
        gate = read_json(root / 'gates/rldx340_reviewed.json')
        if not gate.get('accepted'):
            raise RuntimeError('XR1 full cohort is gated on reviewed RLDX340 results')
    if manifest.get('policy') not in {'xr1', 'rldx'}:
        raise ValueError('RoboCasa worker requires policy xr1 or rldx')
    specs = [s for s in manifest['episodes'] if s['lane'] == lane]
    if not specs:
        raise ValueError('No episodes assigned to this lane')
    if any(s['policy'] != manifest['policy'] for s in specs):
        raise ValueError('Episode policy differs from worker policy')
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--port-base', type=int, default=25801)
    parser.add_argument('--approved', action='store_true', help='Root reviewed this isolated implementation and cohort')
    parser.add_argument('--check-only', action='store_true', help='Validate startup on CPU without starting services')
    args = parser.parse_args()
    if not args.approved:
        parser.error('Launch requires the root review gate: --approved')
    manifest = read_json(args.root / 'campaign.json')
    specs = lane_specs(args.root, manifest, args.lane)
    if args.check_only:
        print(json.dumps({'lane': args.lane, 'episodes': len(specs),
                          'required_gate': manifest['required_gate'], 'services_started': 0}))
        return 0
    control = args.root / 'control' / f'lane_{args.lane}'
    control.mkdir(parents=True, exist_ok=True)
    claim = control / 'worker_started.json'
    with claim.open('x') as stream:
        stream.write('{}')
    processes, streams, service_metadata = [], [], []
    model = env_process = adapter = None
    failures, completed = [], []
    env_port, vla_port, model_port = args.port_base, args.port_base + 1, args.port_base + 2
    ip = socket.gethostbyname(socket.gethostname())
    env_url, vla_url = f'http://127.0.0.1:{env_port}', f'http://127.0.0.1:{vla_port}'
    environment = runtime_environment()
    for key in ('CODEX_API_KEY', 'OPENAI_API_KEY', 'PLANNER_ACCESS_TOKEN', 'STAGE_CODEX_PROVIDER_KEY'):
        environment.pop(key, None)
    environment.update(HF_HOME=str(INSTALL.parents[2] / 'hf_runtime'),
                       HF_MODULES_CACHE=str(INSTALL.parents[2] / 'hf_runtime/modules'))

    def launch(command, path, env, cwd=REPO):
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('w')
        streams.append(stream)
        started_at = time.time()
        child = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env)
        processes.append(child)
        service_metadata.append((child, {'service': path.stem, 'pid': child.pid,
                                        'log': str(path), 'started_at': started_at}))
        return child

    def service_snapshot():
        records = []
        for child, metadata in service_metadata:
            returncode = child.poll()
            records.append({**metadata, 'returncode': returncode,
                            'signal': -returncode if returncode is not None and returncode < 0 else None})
        return records

    def interrupted(_signum, _frame):
        raise InterruptedError('worker interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    error = None
    active = None
    services_at_error = None
    try:
        write_json(claim, {'pid': os.getpid(), 'host': socket.gethostname(), 'ip': ip, 'lane': args.lane,
                          'policy': manifest['policy'], 'episodes': len(specs), 'started_at': time.time()})
        if manifest['policy'] == 'rldx':
            shards = sorted(set(read_json(RLDX_CHECKPOINT / 'model.safetensors.index.json')['weight_map'].values()))
            for name in shards:
                with (RLDX_CHECKPOINT / name).open('rb') as stream:
                    if not stream.read(1):
                        raise RuntimeError('RLDX checkpoint shard is empty')
            write_json(control / 'checkpoint_readable.json', {
                'checkpoint': str(RLDX_CHECKPOINT), 'readable_shards': shards,
                'scope': 'one-byte readability in actual GPU Worker; no hash'})
            model_env = runtime_environment(model=True)
            from research.common import DATA
            model_env.update(HF_HOME=str(DATA / 'checkpoints/rldx-support-hf'),
                             HF_HUB_CACHE=str(DATA / 'checkpoints/rldx-support-hf/hub'))
            model = launch([RLDX_PYTHON, '-m', 'research.vla_policy_server', '--model-path', str(RLDX_CHECKPOINT),
                            '--port', str(vla_port)], control / 'model.log', model_env)
            wait_ready(vla_url, [model])
        else:
            model = launch([PYTHON, '-m', 'research.xr1_model_server', '--model', str(XR1_CHECKPOINT),
                            '--port', str(model_port), '--sessions', str(len(specs)), '--load-device', 'cuda'],
                           control / 'model.log', environment)
            deadline = time.monotonic() + 1200
            while 'READY ' not in (control / 'model.log').read_text():
                if model.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('XR1 model startup failed')
                time.sleep(2)
        write_json(control / 'model_loaded.json', {'pid': model.pid, 'checkpoint_loads': 1, 'time': time.time()})
        for index, spec in enumerate(specs):
            active = Path(spec['output_dir'])
            private = active / 'private'
            private.mkdir(parents=True, exist_ok=True)
            if (private / 'result.json').exists():
                raise FileExistsError(f'Preserve completed attempt; create a separate retry manifest: {active}')
            write_json(private / 'episode_started.json', {'time': time.time()})
            spec_path = active / 'episode.json'
            episode_env = runtime_environment(source=Path(spec['source_root']))
            episode_env.update(STAGE_RESTRICTED='1' if spec['mode'] == 'restricted' else '0',
                               STAGE_ATTACH_ONLY='1', STAGE_ACTION_PROVENANCE='1', STAGE_VLA_POLICY=spec['policy'],
                               PYTHONFAULTHANDLER='1')
            if spec.get('restrictions_file'):
                episode_env['STAGE_RESTRICTIONS_FILE'] = spec['restrictions_file']
            else:
                episode_env.pop('STAGE_RESTRICTIONS_FILE', None)
            env_process = launch([PYTHON, '-m', 'research.env_server', '--episode', str(spec_path),
                                  '--port', str(env_port)], private / 'env.log', episode_env, cwd=Path(spec['source_root']))
            wait_ready(env_url, [env_process, model])
            if manifest['policy'] == 'rldx':
                rpc(vla_url, 'research.begin_episode', (str(private),), sessions=True)
            else:
                adapter = launch([PYTHON, '-m', 'research.xr1_server', '--model-port', str(model_port),
                                  '--port', str(vla_port), '--private-dir', str(private)],
                                 private / 'vla_adapter.log', environment)
                wait_ready(vla_url, [adapter, model])
            write_json(control / 'request.json', {'episode': str(spec_path), 'index': index,
                       'env_endpoint': f'http://{ip}:{env_port}', 'vla_endpoint': f'http://{ip}:{vla_port}'})
            deadline = time.monotonic() + spec['planner_timeout_s'] + 180
            budget_reached_at = None
            while not (private / 'planner_done.json').exists():
                if any(p.poll() is not None for p in (env_process, model, adapter) if p):
                    raise RuntimeError('Owned service exited while Planner was active')
                if time.monotonic() > deadline:
                    write_json(private / 'planner_cancel.json', {'reason': 'worker_deadline'})
                    raise TimeoutError('Planner did not finish within its deadline')
                metrics_path = private / 'environment_metrics.json'
                if metrics_path.exists() and read_json(metrics_path).get('budget_exhausted'):
                    budget_reached_at = budget_reached_at or time.monotonic()
                    if time.monotonic() - budget_reached_at > 60:
                        write_json(private / 'planner_cancel.json', {'reason': 'action_budget'})
                time.sleep(2)
            done = read_json(private / 'planner_done.json')
            rpc(env_url, 'research.finalize', timeout_s=120)
            metrics = read_json(private / 'environment_metrics.json')
            valid = done['exit_code'] == 0 or metrics.get('budget_exhausted', False) or done.get('reason') == 'action_budget'
            record = {'schema': 'stage.restricted-result/v1', 'key': spec['key'],
                      'diagnostic': bool(spec.get('diagnostic', False)), 'benchmark': not bool(spec.get('diagnostic', False)),
                      'task': spec['task'], 'seed': spec['seed'], 'policy': spec['policy'], 'mode': spec['mode'],
                      'task_set': spec['task_set'], 'valid': valid,
                      'termination_reason': done.get('reason') or ('planner_finished' if valid else 'infrastructure_error'),
                      'planner_exit_code': done['exit_code'], **metrics}
            from research.metrics import finalize_metrics
            aggregated = finalize_metrics(spec, result=record)
            record['termination_reason'] = aggregated['termination_reason']
            record['termination_detail'] = aggregated['termination_detail']
            valid = record['valid'] = aggregated['valid']
            write_json(private / 'result.json', record)
            (completed if valid else failures).append(spec['key'])
            rpc(env_url, 'shutdown')
            env_process.wait(timeout=45)
            stop(adapter)
            adapter = None
            env_process = None
            write_json(control / 'progress.json', {'completed': completed, 'failures': failures, 'expected': len(specs)})
            if not valid and not manifest.get('continue_on_planner_error', False):
                raise RuntimeError('Planner failed; preserve artifacts and inspect before continuing')
            active = None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        # Capture native crashes before cancellation/cleanup sends its own signals.
        services_at_error = service_snapshot()
        if active:
            write_json(active / 'private/planner_cancel.json', {'reason': error})
            write_json(active / 'private/worker_error.json', {'error': error, 'time': time.time(),
                       'services_before_cleanup': services_at_error})
            if (active / 'private/planner_started.json').exists():
                cancel_deadline = time.monotonic() + 45
                while not (active / 'private/planner_done.json').exists() and time.monotonic() < cancel_deadline:
                    time.sleep(1)
            from research.metrics import finalize_metrics
            finalize_metrics(spec, error=error)
    finally:
        services_before_cleanup = services_at_error if services_at_error is not None else service_snapshot()
        for child in reversed(processes):
            stop(child)
        for stream in streams:
            stream.close()
        write_json(control / 'worker_exit.json', {'completed': completed, 'failures': failures, 'error': error,
                   'services_before_cleanup': services_before_cleanup,
                   'services_after_cleanup': service_snapshot(),
                   'all_children_stopped': all(p.poll() is not None for p in processes), 'finished_at': time.time()})
    return 0 if error is None and (not failures or manifest.get('continue_on_planner_error', False)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
