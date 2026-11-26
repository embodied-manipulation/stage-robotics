"""The only V/H intervention is capability; H0 ablates classification separately."""
VARIANTS = {'V': (True, False, True), 'H': (True, True, True),
            'H_unclassified': (True, True, False), 'C': (False, True, True)}

# Fields that determine the scene, information, policy, cadence or budget.
# Output paths and the declared capability/classification intervention differ.
PAIR_FIELDS = (
    'policy', 'checkpoint', 'task', 'task_set', 'suite', 'seed', 'task_language',
    'split', 'physics', 'environment_reset', 'resolved_reset_id', 'initial_state_index',
    'planner_provider', 'planner', 'policy_seed', 'planner_timeout_s', 'max_turns',
    'review_window_actions', 'review_window_public_protocol',
    'native_action_steps', 'vla_max_chunks', 'vla_default_chunks',
    'action_budget', 'physical_action_budget', 'env_horizon', 'native_settling_steps',
    'post_agent_wait_steps', 'policy_history_protocol', 'policy_memory_protocol', 'policy_rtc',
    'planner_observation_profile', 'planner_image_window', 'task_feedback_protocol',
    'feedback_profile', 'feedback_sampling', 'feedback_protocol', 'success_stop_policy',
    'vla_prompt_mode', 'vla_instruction_scope', 'language_profile', 'enable_sam3',
    'minimum_vla_actions_before_stuck', 'external_memory', 'online_rsi',
)


def validate(spec):
    variant = spec.get('final_variant')
    if variant not in VARIANTS:
        raise ValueError('Choose final_variant V, H, H_unclassified or C')
    vla, codes, labels = VARIANTS[variant]
    if spec.get('vla_enabled', True) is not vla:
        raise ValueError('VLA capability disagrees with final_variant')
    if spec.get('code_as_policy') is not codes:
        raise ValueError('Executor capability disagrees with final_variant')
    if spec.get('classification_enabled') is not labels:
        raise ValueError('Classification mode disagrees with final_variant')
    if spec.get('review_window_actions') != 40 or spec.get('external_memory') is not False:
        raise ValueError('Final paired preparation retains c40 and no external memory')
    return codes, labels


def capabilities(spec):
    codes, labels = validate(spec)
    return VARIANTS[spec['final_variant']][0], codes, labels


def validate_pairs(episodes):
    pairs = {}
    variants = {s['final_variant'] for s in episodes}
    for spec in episodes:
        group = pairs.setdefault(spec['pair_id'], {})
        if spec['final_variant'] in group:
            raise ValueError('Duplicate pair/variant: ' + spec['key'])
        group[spec['final_variant']] = spec
    for pair, group in pairs.items():
        if 'C' in variants and 'C' not in group:
            raise ValueError('Incomplete Codes-only pair: ' + pair)
        if {'V', 'H'} <= variants and not {'V', 'H'} <= group.keys():
            raise ValueError('Incomplete V/H pair: ' + pair)
        reference = next(iter(group.values()))
        for spec in group.values():
            changed = [field for field in PAIR_FIELDS if spec.get(field) != reference.get(field)]
            if changed:
                raise ValueError(f'Unmatched controls for {pair}: {", ".join(changed)}')
    return len(pairs)
