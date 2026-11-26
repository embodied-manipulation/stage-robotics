"""Native Long evaluation horizons and exact final action-prefix execution."""
OFFICIAL_HORIZONS = {'libero_10_task': 520, 'libero_10_swap': 520}
BUDGET_SOURCE = 'evaluations/libero/libero_10_openpi_pi05_eval.yaml'


def apply_task_budget(spec):
    # LongTask and LongSwap inherit the Long suite's published evaluation cap.
    # Preserve a task-specific official horizon when one is explicitly supplied.
    horizon = spec.get('official_horizon', OFFICIAL_HORIZONS.get(spec['suite']))
    if type(horizon) is not int or horizon <= 0:
        raise ValueError('An official task/suite horizon is required for the 1.6x budget')
    budget = int(horizon * 1.6)
    spec.update(official_horizon=horizon, budget_multiplier=1.6,
        budget_source=BUDGET_SOURCE, budget_protocol='official-horizon-times-1.6/v1',
        physical_action_budget=budget,
        env_horizon=budget + spec['native_settling_steps'])


def install_final_prefix(env, *, budget, action_count):
    """Clip before cadence accounting so a partial final chunk counts exactly."""
    original = env.chunk_step

    def chunk_step(actions, **kwargs):
        remaining = budget - action_count()
        if remaining <= 0:
            raise RuntimeError('Physical action budget exhausted')
        return original(actions[:remaining], **kwargs)

    env.chunk_step = chunk_step
