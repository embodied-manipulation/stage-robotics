"""Coordinate API request timing across local Planner processes."""
from contextlib import ExitStack, contextmanager
import fcntl
import json
from pathlib import Path
import random
import time


def _effective_config(config):
    config = dict(config)
    if path := config.get('live_config_file'):
        overrides = json.loads(Path(path).read_text())
        allowed = {'max_concurrent_requests', 'minimum_interval_seconds'}
        if set(overrides) - allowed:
            raise ValueError('Live pacing configuration may only change admission limits')
        config.update(overrides)
    interval = config.get('minimum_interval_seconds', 0)
    if not isinstance(interval, (int, float)) or interval < 0:
        raise ValueError('minimum_interval_seconds must be nonnegative')
    return config


@contextmanager
def _state(path):
    # Keep the inode stable: every process locks this same file.
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        state = json.loads(stream.read() or '{}')
        yield state
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(state))
        stream.flush()


@contextmanager
def _concurrency_slot(config):
    limit = config.get('max_concurrent_requests')
    if limit is None:
        yield {}
        return
    if not isinstance(limit, int) or limit < 1:
        raise ValueError('max_concurrent_requests must be a positive integer')
    directory = Path(config['concurrency_state_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    queued = time.monotonic()
    # File locks are shared across Planner processes and released on exit.
    with ExitStack() as stack:
        slots = [stack.enter_context((directory / f'{index}.lock').open('a+'))
                 for index in range(limit)]
        selected = None
        while selected is None:
            for index, stream in enumerate(slots):
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                selected = index
                break
            if selected is None:
                time.sleep(0.05)
        try:
            yield {'concurrency_slot': selected,
                   'concurrency_queue_seconds': time.monotonic() - queued}
        finally:
            fcntl.flock(slots[selected], fcntl.LOCK_UN)


@contextmanager
def request_slot(config):
    if not config:
        yield {}
        return
    config = _effective_config(config)
    path = Path(config['state_file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    queued = time.monotonic()
    # The slot covers the request; the timing lock covers only admission and
    # completion bookkeeping. Holding that lock across yield serializes all
    # requests even when several concurrency slots are configured.
    with _concurrency_slot(config) as admission:
        while True:
            with _state(path) as state:
                started = time.time()
                delay = max(0, state.get('next_request_at', 0) - started)
                if not delay:
                    generation = state.get('rate_limit_generation', 0)
                    state['next_request_at'] = started + config.get('minimum_interval_seconds', 0)
            if not delay:
                break
            # Another in-flight request may extend the cooldown while asleep.
            time.sleep(delay)
        result = {'queue_seconds': time.monotonic() - queued, **admission,
                  'max_concurrent_requests': config.get('max_concurrent_requests'),
                  'minimum_interval_seconds': config.get('minimum_interval_seconds', 0),
                  'scheduler_version': 'concurrent-admission/v2'}
        try:
            yield result
        except BaseException:
            result['request_succeeded'] = False
            raise
        finally:
            with _state(path) as state:
                limited = result.get('api_error_code') == 'rate_limit_exceeded'
                failures = state.get('consecutive_rate_limits', 0)
                cooldown = 0
                if limited:
                    failures += 1
                    state['rate_limit_generation'] = state.get('rate_limit_generation', 0) + 1
                    base = config.get('rate_limit_cooldown_seconds', 30)
                    maximum = config.get('maximum_rate_limit_cooldown_seconds', base)
                    factor = config.get('rate_limit_backoff_factor', 1)
                    cooldown = min(maximum, base * factor ** min(failures - 1, 20))
                    cooldown = max(cooldown, result.get('retry_after_seconds') or 0)
                    cooldown += random.SystemRandom().uniform(
                        0, config.get('rate_limit_jitter_seconds', 0))
                    state['next_request_at'] = max(state.get('next_request_at', 0),
                                                   time.time() + cooldown)
                elif (result.get('request_succeeded', True)
                      and generation == state.get('rate_limit_generation', 0)):
                    # An older success must not cancel a newer request's 429.
                    failures = 0
                state['consecutive_rate_limits'] = failures
                result.update(applied_cooldown_seconds=cooldown,
                              consecutive_rate_limits=failures)
