"""Resolve portable template paths from an explicitly configured installation."""
import os
from pathlib import Path


def resolve_paths(value):
    if isinstance(value, dict):
        return {key: resolve_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_paths(item) for item in value]
    if isinstance(value, str):
        defaults = {
            '/data/robotics/RoboCasa365': os.environ.get('STAGE_DATA_ROOT', '/data/robotics/RoboCasa365'),
            '/opt/robotics': os.environ.get('STAGE_INSTALL_ROOT', '/opt/robotics'),
        }
        for prefix, target in defaults.items():
            if value == prefix or value.startswith(prefix + '/'):
                return str(Path(target).expanduser()) + value[len(prefix):]
    return value


def worker_environment_args():
    """Forward installation settings to isolated workers without credentials."""
    args = []
    for name in ('STAGE_DATA_ROOT', 'STAGE_INSTALL_ROOT', 'STAGE_RUN_UID', 'STAGE_RUN_GID'):
        if value := os.environ.get(name):
            args.extend(['-e', f'{name}={value}'])
    return args


def python_environment(env=None):
    """Make source-checkout modules importable in child processes from any cwd."""
    env = dict(os.environ if env is None else env)
    source_root = str(Path(__file__).resolve().parents[3])
    paths = [source_root, *env.get('PYTHONPATH', '').split(os.pathsep)]
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return env
