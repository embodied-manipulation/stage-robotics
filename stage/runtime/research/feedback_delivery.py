"""Validate native benchmark fields without adding model-facing audit metadata."""
from research.common import append_json


def record_delivery(data, *, with_feedback, benchmark, private, tool, protocol=None,
                    allow_historical=False):
    if not isinstance(data, dict) or 'state' not in data or 'step' not in data:
        return
    fields = ({'terminated', 'truncated'} if benchmark == 'libero'
              else {'success', 'task_progress', 'robocasa_terminated'})
    full_libero = benchmark == 'libero' and protocol == 'libero-native-task-feedback/v1'
    if full_libero:
        fields |= {'success_criteria', 'task_progress', 'grasp_feedback'}
    legacy = {'feedback_source', 'feedback_available', 'feedback_action',
              'success_stop_triggered', 'libero_terminated'}
    missing = sorted(fields - data.keys()) if with_feedback else []
    leaked = sorted((fields | legacy | {'success', 'task_progress'}) & data.keys()) if not with_feedback else []
    malformed = []
    if with_feedback:
        if legacy & data.keys():
            malformed.append('non-native feedback metadata in public observation')
        if benchmark == 'libero':
            if ({'success'} if full_libero else {'success', 'task_progress'}) & data.keys():
                malformed.append('non-native whole-task aliases in LIBERO observation')
            for key in {'terminated', 'truncated'} & data.keys():
                if type(data[key]) is not bool:
                    malformed.append(f'{key} must be native bool')
            if full_libero and not missing:
                from stage_bridge.libero_feedback import validate_observation
                try:
                    validate_observation(data, allow_historical=allow_historical)
                except (ValueError, KeyError, TypeError) as exc:
                    malformed.append(str(exc))
        elif not missing:
            if type(data['success']) is not bool or type(data['robocasa_terminated']) is not bool:
                malformed.append('official success and native termination must be bool')
            if not isinstance(data['task_progress'], dict):
                malformed.append('task_progress must be a dictionary')
    row = {'tool': tool, 'step': data['step'], 'with_feedback': with_feedback,
           'benchmark': benchmark, 'protocol': protocol or 'unspecified',
           'boundary': 'final_structured_tool_result_before_model_serialization',
           'feedback': {key: data.get(key) for key in fields} if with_feedback else {},
           'missing': missing, 'leaked': leaked, 'malformed': malformed,
           'ok': not (missing or leaked or malformed)}
    append_json(private / 'feedback_delivery.jsonl', row)
    if not row['ok']:
        raise RuntimeError(f'Official feedback delivery contract failed: {row}')
