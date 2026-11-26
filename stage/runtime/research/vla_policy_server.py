"""Keep the supported upstream RLDX policy loaded; record private request timing."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from robots.robocasa.vla_server import RoboCasaVLAFacade, POLICY_MEMORY_PROTOCOL
from research.common import append_json


class MeasuredVLAFacade(RoboCasaVLAFacade):
    def __init__(self, model_path):
        self.private = None
        self.request_count = 0
        super().__init__(model_path)

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc['research.begin_episode'] = self.begin_episode

    def begin_episode(self, private_dir: str, *, session_id=None):
        self.private = Path(private_dir)
        self.request_count = 0
        # Every episode also gets a fresh native RPC client session. Reset RNG
        # once here; there is no learned policy state or prefix cache to reset.
        import random
        import numpy as np
        import torch
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        from research.common import write_json
        write_json(self.private / 'policy_contract.json', self.get_modality_config())
        return {'ok': True, 'policy_memory_protocol': POLICY_MEMORY_PROTOCOL}

    def predict(self, obs_dict, options, *, session_id):
        started = time.perf_counter()
        actions = super().predict(obs_dict, options, session_id=session_id)
        self.request_count += 1
        if self.private:
            append_json(self.private / 'vla_requests.jsonl', {
                'request': self.request_count, 'prompt': obs_dict.get('annotation.human.task_description'),
                'policy_memory_protocol': POLICY_MEMORY_PROTOCOL,
                'memory_enabled': False, 'rtc_enabled': False,
                'inference_seconds': time.perf_counter() - started,
                'returned_actions': int(next(iter(actions.values())).shape[1]),
                'session_id': session_id,
            })
        return actions


def main():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        import os
        os.environ.setdefault('RLDX_ATTN_IMPL', 'sdpa')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    MeasuredVLAFacade(args.model_path).serve(transport='http', host='0.0.0.0', port=args.port, session_sweep_s=60)


if __name__ == '__main__':
    main()
