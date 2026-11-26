"""Read-only provenance for valid historical V outcomes; never relabel logs."""
import json
from pathlib import Path


def reused_result(spec):
    reference=spec.get('reused_result_file')
    if not reference:return None
    if spec.get('final_variant')!='V':raise ValueError('Only V may reuse historical V results')
    result=json.loads(Path(reference).read_text())
    if result.get('valid') is not True:raise ValueError('Historical reuse requires a valid result')
    return dict(result, historical_result_file=reference, reused_for_pair_id=spec['pair_id'])
