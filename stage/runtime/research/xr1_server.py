"""Official XR1 EvalClient with the native VLA RPC interface."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from research.common import INSTALL, XR1_CHECKPOINT, append_json

CAMERAS = ('video.robot0_agentview_left', 'video.robot0_agentview_right', 'video.robot0_eye_in_hand')
STATE_KEYS = ('state.end_effector_position_relative', 'state.end_effector_rotation_relative',
              'state.base_position', 'state.base_rotation', 'state.gripper_qpos')


def build_inputs(observation, converter):
    images = {key: np.asarray(observation[key])[0].astype(np.uint8) for key in CAMERAS}
    histories = {key: np.asarray(observation[key])[0] for key in STATE_KEYS}
    if any(len(value) != 4 for value in [*images.values(), *histories.values()]):
        raise ValueError('XR1 requires exactly four sensor/state frames')
    state = np.stack([converter({key: value[index] for key, value in histories.items()}) for index in range(4)]).astype(np.float32)
    if state.shape != (4, 14):
        raise ValueError(f'XR1 state must be (4,14), got {state.shape}')
    prompt = observation['annotation.human.task_description'][0]
    return state, images, prompt


def action_fields(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 12 or raw.shape[0] < 16:
        raise ValueError(f'XR1 expected at least 16x12 predicted actions, got {raw.shape}')
    if not np.isfinite(raw).all():
        raise ValueError('XR1 returned nonfinite actions')
    # Return the native 16 predictions. The VLA skill executes 8 action steps per chunk.
    raw = raw[None, :16]
    return {'action.end_effector_position': raw[:, :, :3],
            'action.end_effector_rotation': raw[:, :, 3:6],
            'action.gripper_close': raw[:, :, 6:7],
            'action.base_motion': raw[:, :, 7:11],
            'action.control_mode': raw[:, :, 11:12]}


class XR1Facade(BaseVLAFacade):
    def __init__(self, model_port: int, private_dir: Path):
        sys.path.insert(0, str(INSTALL / 'Xiaomi-Robotics-1/eval_robocasa365'))
        from entry import EvalClient, observation_to_state
        self.converter = observation_to_state
        self.client = EvalClient(model_path=str(XR1_CHECKPOINT), host='127.0.0.1', port=model_port,
                                 robot_type='robocasa365', crop_ratio=0.95)
        self.private = private_dir
        self.requests = 0
        super().__init__(enable_sessions=True)

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc['vla.get_modality_config'] = self.get_modality_config
        self._rpc['vla.reset_session'] = self.reset_session
        self._readonly_methods.add('vla.get_modality_config')

    def get_modality_config(self, *, session_id=None):
        return {'video_delta_indices': [-6, -4, -2, 0], 'hist_maxlen': 8,
                'policy_memory_protocol': 'stateless-subgoal/v1',
                'memory_enabled': False, 'rtc_enabled': False}

    def reset_session(self, *, session_id):
        return {'ok': True, 'model_memory': 'none'}

    def predict(self, obs_dict, options=None, *, session_id):
        state, images, prompt = build_inputs(obs_dict, self.converter)
        started = time.perf_counter()
        result = action_fields(self.client.infer(state, images, prompt))
        self.requests += 1
        append_json(self.private / 'vla_requests.jsonl', {
            'request': self.requests, 'prompt': prompt,
            'reset_memory': (options or {}).get('reset_memory'),
            'inference_seconds': time.perf_counter() - started,
            'returned_actions': 16, 'session_id': session_id,
            'state_shape': list(state.shape),
            'image_shapes': {key: list(value.shape) for key, value in images.items()},
            'finite_actions': True, 'policy_memory_protocol': 'stateless-subgoal/v1',
            'session_scope': 'RPC transport only; not forwarded to learned model',
        })
        return result

    def close(self):
        self.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-port', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    args = parser.parse_args()
    XR1Facade(args.model_port, args.private_dir).serve(transport='http', host='0.0.0.0', port=args.port, session_sweep_s=60)


if __name__ == '__main__':
    main()
