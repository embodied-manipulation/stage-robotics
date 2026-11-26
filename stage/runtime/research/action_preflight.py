"""Non-benchmark physical diagnostic using native Code-as-Policy tools without a VLM request."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from research.common import read_json, rpc, write_json


class DiagnosticPlanner:
    """Exercise observation, analytic control, and one native eight-action VLA chunk."""

    def __init__(self, private: Path, endpoint: str, receipt: dict):
        self.private = private
        self.endpoint = endpoint
        self.receipt = receipt

    def solve(self, *, toolkit, **_kwargs):
        from rpent.planner.base import PlannerResult

        def execute(name, arguments, expected_actions=0):
            self.receipt['phase'] = name
            write_json(self.private / 'diagnostic.json', self.receipt)
            before = rpc(self.endpoint, 'research.action_count')['actions']
            result = toolkit.execute_tool(name, arguments)
            after = rpc(self.endpoint, 'research.action_count')['actions']
            record = {'tool': name, 'arguments': arguments, 'result': result.to_dict(),
                      'image_count': len(result.images), 'action_start': before,
                      'action_end': after, 'physical_actions': after - before}
            self.receipt['calls'].append(record)
            write_json(self.private / 'diagnostic.json', self.receipt)
            if result.is_error:
                raise RuntimeError(f"Diagnostic {name} failed: {result.data['error']}")
            if after - before != expected_actions:
                raise RuntimeError(f'Diagnostic {name} executed {after - before} actions; expected {expected_actions}')
            return result

        try:
            initial = execute('view_env_state', {})
            cameras = {image['camera'] for image in initial.data.get('images', [])}
            if not {'agentview', 'navview', 'wrist'} <= cameras or len(initial.images) < 3:
                raise RuntimeError(f'Diagnostic native camera observations missing: {sorted(cameras)}')
            language = initial.data.get('task_language')
            if not language:
                raise RuntimeError('Diagnostic requires the live task_language from view_env_state')
            execute('set_gripper', {'gripper': 1.0, 'steps': 1}, expected_actions=1)
            execute('vla_skill', {'prompt': language, 'max_chunks': 1}, expected_actions=8)
            finished = execute('finish', {
                'status': 'stuck',
                'summary': 'Non-benchmark diagnostic completed: native cameras, one analytic action, and one eight-action VLA chunk. No VLM was called.',
            })
            self.receipt.update(phase='complete', passed=True,
                                native_last_state_step=toolkit._state.latest_record().step_idx)
            write_json(self.private / 'diagnostic.json', self.receipt)
            return PlannerResult(
                finish_result=finished.to_dict(), messages=self.receipt['calls'],
                stats={'diagnostic': True, 'benchmark': False, 'vlm_requests': 0,
                       'total_input_tokens': 0, 'total_output_tokens': 0,
                       'turns_used': 0, 'tool_calls': len(self.receipt['calls'])},
            )
        except Exception as exc:
            self.receipt.update(passed=False, error=f'{type(exc).__name__}: {exc}')
            write_json(self.private / 'diagnostic.json', self.receipt)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--env-endpoint', required=True)
    parser.add_argument('--vla-endpoint', required=True)
    args = parser.parse_args()
    spec = read_json(args.episode)
    private = Path(spec['output_dir']) / 'private'
    receipt = {'diagnostic': True, 'benchmark': False, 'vlm_requests': 0,
               'phase': 'native_initialization', 'passed': False, 'calls': []}
    write_json(private / 'diagnostic.json', receipt)
    cli = importlib.import_module('rpent.cli.main')
    original_builder = cli.build_planner
    cli.build_planner = lambda *_args, **_kwargs: DiagnosticPlanner(private, args.env_endpoint, receipt)
    from research import runner
    try:
        status = runner.main()
        if status:
            raise RuntimeError(f'Native diagnostic runner exited with status {status}; inspect planner.log')
        return status
    except Exception as exc:
        receipt.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        write_json(private / 'diagnostic.json', receipt)
        raise
    finally:
        cli.build_planner = original_builder


if __name__ == '__main__':
    raise SystemExit(main())
