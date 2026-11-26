"""Experiment identities, JSON artifacts, and owned service lifecycle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO.parent
BACKEND = REPO.parent / 'backends/cap'
DATA = Path(os.environ.get('STAGE_DATA_ROOT', str(REPO.parent / 'data')))
INSTALL = Path(os.environ.get('STAGE_INSTALL_ROOT', str(DATA / 'runtime')))
PYTHON = os.environ.get('STAGE_PYTHON', 'python')
RLDX_PYTHON = os.environ.get('STAGE_VLA_PYTHON', PYTHON)
OVERLAY = Path(os.environ.get('STAGE_OVERLAY', str(DATA / 'env_overlay')))
ROBOSUITE_SOURCE = Path(os.environ.get('STAGE_ROBOSUITE_SOURCE', str(OVERLAY / 'sources/robosuite')))
RLDX_CHECKPOINT = Path(os.environ.get('STAGE_RLDX_CHECKPOINT', str(DATA / 'checkpoints/rldx-1')))
XR1_CHECKPOINT = Path(os.environ.get('STAGE_XR1_CHECKPOINT', str(DATA / 'checkpoints/xr-1')))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def append_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, default=str) + '\n')


def runtime_environment(*, model: bool = False, source: Path = REPO) -> dict[str, str]:
    env = dict(os.environ)
    for key in ('VLA_ACTION_STEPS_PER_CHUNK', 'VLA_MAX_CHUNKS', 'VLA_SETTLE_PATIENCE', 'VLA_VIDEO_DIR'):
        env.pop(key, None)
    paths = [str(source), str(REPO), str(BRIDGE), str(BACKEND), str(OVERLAY / ('deps310' if model else 'deps311'))]
    if not model:
        paths += [str(INSTALL / 'RoboCasa365/robocasa'), str(ROBOSUITE_SOURCE), str(INSTALL / 'RoboCasa365/robosuite'), str(INSTALL / 'Xiaomi-Robotics-1')]
    env.update(PYTHONPATH=':'.join(paths), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1', MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', LP_NUM_THREADS='1', VLA_RESET_SEED='')
    return env


def rpc(endpoint: str, method: str, args: tuple = (), *, timeout_s: float = 60, sessions: bool = False) -> Any:
    from rpent.utils.rpc import make_rpc_client, wait_for_ready
    client = make_rpc_client(endpoint, enable_sessions=sessions)
    try:
        if sessions:
            wait_for_ready(client, timeout_s=timeout_s)
        return client.call(method, args=args, timeout_s=timeout_s)
    finally:
        client.close()


def wait_ready(endpoint: str, processes: list[subprocess.Popen], timeout: float = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('Owned service exited during startup; inspect its private log')
        try:
            rpc(endpoint, 'healthz', timeout_s=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f'Service readiness timed out: {endpoint}')


def stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
