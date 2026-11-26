"""Run OFF/ON renderer-lifetime diagnostics in separate GPU child processes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

from research.common import PYTHON, REPO, read_json, runtime_environment, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force-gc-step', type=int, default=400)
    args = parser.parse_args()
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(int(os.environ['AGENTIC_RUN_GID']))
        os.setuid(int(os.environ['AGENTIC_RUN_UID']))
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    gpu = torch.cuda.get_device_properties(0)
    write_json(args.output / 'gpu.json', {'name': gpu.name, 'device_count': torch.cuda.device_count(),
               'cuda_result': float((torch.ones(1, device='cuda') * 2).item())})
    results = []
    for label, enabled in [('guard_off', False), ('guard_on', True)]:
        environment = runtime_environment()
        environment.update(STAGE_PRESERVE_RESET_CONTEXTS='1' if enabled else '0', PYTHONFAULTHANDLER='1')
        output = args.output / label
        command = [PYTHON, '-m', 'research.render_replay_diagnostic', '--episode', str(args.episode),
                   '--output', str(output), '--force-gc-step', str(args.force_gc_step)]
        started = time.time()
        with (args.output / f'{label}.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            write_json(args.output / f'{label}_started.json', {'command': command, 'pid': child.pid,
                       'started_at': started, 'preserve_reset_contexts': enabled})
            try:
                returncode = child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                returncode = child.returncode
        result = {'guard_enabled': enabled, 'returncode': returncode,
                  'signal': -returncode if returncode < 0 else None,
                  'elapsed_seconds': time.time() - started, 'child_stopped': child.poll() is not None,
                  'diagnostic': read_json(output / 'diagnostic.json') if (output / 'diagnostic.json').exists() else None}
        results.append(result)
        write_json(args.output / f'{label}_exit.json', result)
    write_json(args.output / 'pair_result.json', {'diagnostic_only': True, 'benchmark': False,
               'policy_calls': 0, 'results': results, 'all_children_stopped': all(r['child_stopped'] for r in results)})


if __name__ == '__main__':
    main()
