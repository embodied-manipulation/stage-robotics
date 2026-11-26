"""Drop the container root identity, record the allocated GPU, and run one lane."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--port-base', type=int, default=25801)
    parser.add_argument('--approved', action='store_true')
    args = parser.parse_args()
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(int(os.environ['AGENTIC_RUN_GID']))
        os.setuid(int(os.environ['AGENTIC_RUN_UID']))
    import torch
    from research.common import write_json
    gpu = torch.cuda.get_device_properties(0)
    evidence = {'gpu_name': gpu.name, 'gpu_count': torch.cuda.device_count(),
                'memory_bytes': gpu.total_memory, 'uid': os.getuid(),
                'cuda_result': float((torch.ones(1, device='cuda') * 2).item())}
    write_json(args.root / 'control' / f'lane_{args.lane}' / 'gpu.json', evidence)
    if evidence['gpu_count'] != 1 or evidence['cuda_result'] != 2.0:
        raise RuntimeError(f'Expected exactly one usable allocated GPU: {evidence}')
    command = [sys.executable, '-m', 'research.worker', '--root', str(args.root),
               '--lane', str(args.lane), '--port-base', str(args.port_base)]
    if args.approved:
        command.append('--approved')
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
