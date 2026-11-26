"""Record upstream and effective RoboCasa prompts without exposing priors online."""
import difflib
from pathlib import Path

from research.common import write_json


def record_robocasa_prompts(private: Path, variables: dict) -> None:
    from robots.robocasa import prompt_bundle
    from robots.robocasa.action_interface import decision_reporting_profile
    from robots.robocasa.prompts import evaluate
    from rpent.prompt.manipulation_guidance import GUIDANCE_PROFILE
    from rpent.prompt.utils import format_prompt

    upstream = evaluate.system_prompt()
    effective = prompt_bundle.system_prompt(variables)
    before = format_prompt(upstream, variables=variables)
    after = format_prompt(effective, variables=variables)
    private.mkdir(parents=True, exist_ok=True)
    (private/'upstream_system_prompt.md').write_text(before)
    (private/'system_prompt.md').write_text(after)
    (private/'user_prompt.md').write_text(format_prompt(prompt_bundle.user_prompt(variables), variables=variables))
    (private/'system_prompt.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile='upstream/system', tofile='effective/system')))
    reasons = {
        'Scene_Evidence_And_Recovery': 'Ground object identity and recovery in current observations; use unmet feedback to direct remaining work; reserve stuck for an evidenced terminal blocker.',
        'Subgoal_Review': 'User-selected 5 x 8 = 40-action shared window; explicit classification owns history resets; optional analytic Code as Policy.',
        'Intro': 'Remove simulator success signal description.',
        'Localization': 'Describe the additional Planner right RGB view and retain calibrated-camera-only metric localization.',
        'Goal': 'Remove evaluator predicate and progress feedback.',
        'Rules': 'Apply the selected feedback boundary and require a recovery assessment before terminal stuck.',
        'Memory': 'Expose the selected frozen upstream task/global experience with current tool schemas; online RSI and memory writes are disabled.',
        'Primitives': 'Remove disabled test-time reset.',
        'VLA_Rules': 'Replace private contact/task-counter feedback with visible progress; preserve upstream instruction routing, cadence and recovery policy unless explicitly running an ablation.',
        'Workflow': 'Preserve a suitable observable goal across motions and executor changes; evidence determines advance or recovery.',
        'Action_Selection': 'Extra research guidance: ablation only.',
        'Policy_State': 'No recurrent policy memory; continue retains sensor history including analytic motion; start/advance/recover explicitly reset it.',
        'VLA_Observation_Checkpoints': 'Framework-managed observations are independent of Subgoal duration; no Planner chunk choice or continuation-count rule.',
        'Codes_Observation_Checkpoints': 'Codes-only ablation retains the shared 40-action observation window; interrupted code is not automatically resumed.',
        'Codes_Execution': 'Explicitly selected ASPIRE-inspired Codes execution and diagnosis guidance; native tools, classification and action window remain unchanged.',
        'Task_Independent_Guidance': f'Fixed task-independent manipulation guidance ({GUIDANCE_PROFILE}); no task recipes, benchmark parameters or evaluator feedback.',
        'Output': f'Optional Task Progress reporting ({decision_reporting_profile()}): append an estimated task_progress JSON request; retain the upstream reasoning and action-selection instructions.',
    }
    if "**FOUR cameras**" in effective.get("Intro", ""):
        reasons['Intro'] = 'Describe four Planner cameras while preserving the selected feedback boundary.'
    changed = [key for key in dict.fromkeys([*upstream, *effective])
               if upstream.get(key) != effective.get(key)]
    write_json(private/'prompt_changes.json', [
        {'section': key, 'reason': reasons[key]} for key in changed])
    enabled = 'Task_Independent_Guidance' in effective
    write_json(private/'guidance_profile.json', {
        'enabled': enabled, 'profile': GUIDANCE_PROFILE if enabled else None,
        'text': effective.get('Task_Independent_Guidance', ''),
    })
