"""Official-only transport with a shared, fail-closed campaign dollar ledger.

Rates verified 2026-11-05: Standard Astra, long context above 272K.
The guard charges uncached input at the maximum 12h cache-write rate,
and reserves a full model-window upper bound before each paid request.
Unknown post-upload usage retains its reservation and blocks new requests.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid
from .reliable_response import request_reliably

RESERVATION_USD = 65.0  # 1.05M * $50/M + 128K * $75/M = $62.10, rounded up.

@contextmanager
def transaction(path):
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        yield state
        tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)

def reserve(path, episode):
    with transaction(path) as state:
        committed = state['guard_spent_usd'] + sum(v['usd'] for v in state['reservations'].values())
        if state.get('blocked') or committed + RESERVATION_USD > state['limit_usd']:
            state['blocked'] = state.get('blocked') or 'insufficient_unreserved_budget'
            return None
        ticket = uuid.uuid4().hex
        state['reservations'][ticket] = dict(usd=RESERVATION_USD, episode=episode, at=time.time())
        return ticket

def settle(path, ticket, result):
    usage = result.get('usage')
    with transaction(path) as state:
        held = state['reservations'].pop(ticket)
        known = isinstance(usage, dict) and 'input_tokens' in usage and 'output_tokens' in usage
        if known:
            i, o = int(usage['input_tokens']), int(usage['output_tokens'])
            c = int((usage.get('input_tokens_details') or {}).get('cached_tokens') or 0)
            if min(i,o,c) < 0 or c > i or i > 1050000 or o > 128000:
                state['blocked'] = 'unexpected_usage_bounds'
                estimate = guard = held['usd']
            else:
                long = i > 272000
                estimate = ((i-c)*(20 if long else 10)+c*(2 if long else 1)+o*(75 if long else 50))/1e6
                guard = ((i-c)*(50 if long else 25)+c*(2 if long else 1)+o*(75 if long else 50))/1e6
                for name, count in [('input_tokens',i),('cached_input_tokens',c),('output_tokens',o)]:
                    state[name] += count
        elif (result.get('safe_to_retry_before_api_upload') is True
              or result.get('http_status') in {400,401,403,404,405,413,422,429}):
            estimate = guard = 0.0
        else:
            estimate = 0.0
            guard = held['usd']
            state['unknown_usage_requests'] += 1
            state['blocked'] = 'unknown_post_upload_usage'
        state['estimated_spent_usd'] += estimate
        state['guard_spent_usd'] += guard
        state['requests'] += 1
        state['updated_at'] = time.time()
        entry = dict(at=time.time(), ticket=ticket, episode=held['episode'], usage=usage,
                     estimated_usd=estimate, guard_usd=guard,
                     http_status=result.get('http_status'), response_event=result.get('response_event'),
                     api_error_code=result.get('api_error_code'), request_id=result.get('request_id'))
        with Path(path).with_name('api_usage.jsonl').open('a') as out:
            out.write(json.dumps(entry)+'\n')

def routed_request(request, body, _local_key, proxy, event_path, *, attempt_sink, **options):
    payload = json.loads(body)
    if payload.get('model') != 'gpt-6-astra' or payload.get('service_tier') not in (None,'auto','default','standard'):
        raise ValueError('Budget rates require standard gpt-6-astra')
    if payload.get('background') or payload.get('previous_response_id') or payload.get('conversation'):
        raise ValueError('Budget guard requires existing stateless request protocol')
    path = os.environ['STAGE_API_BUDGET_FILE']
    ticket = reserve(path, str(Path(event_path).parent.parent))
    if ticket is None:
        result = dict(outcome='budget_guard_blocked', api_request_may_have_been_sent=False,
                      safe_to_retry_before_api_upload=True, provider='openai-official')
        attempt_sink(result)
        return result
    key = Path(os.environ['STAGE_OFFICIAL_KEY_FILE']).read_text().strip()
    options.update(origin='api.openai.com', max_attempts=1, max_preupload_attempts=1)
    settled = False
    def record(result):
        nonlocal settled
        settle(path,ticket,result)
        settled = True
        result.update(provider='openai-official', budget_ticket=ticket)
        attempt_sink(result)
    try:
        return request_reliably(request,body,key,proxy,event_path,attempt_sink=record,**options)
    finally:
        if not settled:
            settle(path,ticket,{'outcome':'unreported_transport_exception'})
