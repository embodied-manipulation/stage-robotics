"""Limit outbound Planner images without altering text or VLA observation history."""
from copy import deepcopy


def trim_old_images(payload, *, limit=50, keep=48):
    if not 0 < keep <= limit:
        raise ValueError('Image retention must be positive and within the limit')

    def count(value):
        if isinstance(value, dict):
            return int(value.get('type') == 'input_image') + sum(count(v) for v in value.values())
        if isinstance(value, list):
            return sum(count(v) for v in value)
        return 0

    before = count(payload.get('input', []))
    remove = before - keep if before > limit else 0
    metadata = {'image_history_policy': 'over-50-keep-latest-48/v1',
                'image_count_before_trim': before, 'image_count_after_trim': before - remove,
                'oldest_images_removed': remove}
    if not remove:
        return payload, metadata
    result = deepcopy(payload)
    remaining = remove

    def visit(value):
        nonlocal remaining
        if isinstance(value, dict):
            if value.get('type') == 'input_image' and remaining:
                remaining -= 1
                return {'type': 'input_text', 'text': '[Earlier image omitted from this request; recent images retained.]'}
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value

    result['input'] = visit(result.get('input', []))
    return result, metadata
