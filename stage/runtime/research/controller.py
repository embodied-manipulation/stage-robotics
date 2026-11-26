"""Network-enabled native Planner controller; GPU workers contain no login data."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from research.common import OVERLAY, PYTHON, REPO, read_json, runtime_environment, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lane', type=int, default=0)
    args = parser.parse_args()
    control = args.root / 'control' / f'lane_{args.lane}'
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = stream = private = None
    handled = set()
    reason = None

    def interrupted(_signum, _frame):
        raise InterruptedError('controller interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_json(control / 'controller_started.json', {'pid': os.getpid(), 'time': time.time()})
    try:
        while True:
            write_json(control / 'controller_heartbeat.json', {'time': time.time(), 'pid': os.getpid()})
            if child is not None:
                cancel = private / 'planner_cancel.json'
                if cancel.exists() and child.poll() is None:
                    reason = read_json(cancel)['reason']
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                if child.poll() is not None:
                    write_json(private / 'planner_done.json', {'exit_code': child.returncode, 'reason': reason, 'finished_at': time.time()})
                    stream.close()
                    child = stream = None
            if child is None and (control / 'worker_exit.json').exists():
                return 0 if read_json(control / 'worker_exit.json')['error'] is None else 1
            request_path = control / 'request.json'
            if child is None and request_path.exists():
                request = read_json(request_path)
                episode = request['episode']
                if episode not in handled:
                    spec = read_json(episode)
                    runner_module = spec.get('runner_module', 'research.runner')
                    if runner_module not in {'research.runner', 'research.action_preflight'}:
                        raise ValueError(f'Unsupported runner module: {runner_module}')
                    private = Path(spec['output_dir']) / 'private'
                    if (private / 'planner_done.json').exists():
                        handled.add(episode)
                        continue
                    env = runtime_environment(source=Path(spec['source_root']))
                    extra_deps = os.environ.get('STAGE_EXTRA_DEPS')
                    if extra_deps:
                        env['PYTHONPATH'] += f':{extra_deps}'
                    env.update(HF_HUB_CACHE=os.environ.get('HF_HUB_CACHE', str(OVERLAY / 'hf-cache')))
                    stream = (private / 'planner.log').open('w')
                    child = subprocess.Popen([PYTHON, '-m', runner_module, '--episode', episode,
                                              '--env-endpoint', request['env_endpoint'], '--vla-endpoint', request['vla_endpoint']],
                                             cwd=Path(spec['source_root']), env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    write_json(private / 'planner_started.json', {'pid': child.pid, 'started_at': time.time()})
                    reason = None
                    handled.add(episode)
            time.sleep(2)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            write_json(private / 'planner_done.json', {'exit_code': child.returncode, 'reason': 'controller_interrupted', 'finished_at': time.time()})
        if stream is not None:
            stream.close()
        write_json(control / 'controller_exit.json', {'time': time.time(), 'planner_stopped': child is None or child.poll() is not None})


if __name__ == '__main__':
    raise SystemExit(main())
