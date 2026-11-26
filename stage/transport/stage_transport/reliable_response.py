"""Retry a stateless generation before exposing any model output to Codex.

Local tools execute only after delivery. No RPC/action is retried here. A lost
generation can still consume unreported provider tokens; log every attempt.
"""
import json
import random
import time
import uuid

REVISION = 'buffered-stateless-generation/v1'
NETWORK_ERRORS = {'ConnectionError', 'ConnectionResetError', 'ConnectionRefusedError',
                  'BrokenPipeError', 'SSLEOFError', 'SSLError', 'TimeoutError',
                  'RemoteDisconnected', 'IncompleteRead', 'gaierror', 'OSError'}


def replayable(body, options):
    supported = (options.get('origin')=='chatgpt.com' and options.get('path')=='/backend-api/codex/responses') or (
        options.get('origin')=='api.openai.com' and options.get('path')=='/v1/responses')
    if not supported:
        return False
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return False
    # The formal client advertises local MCP/code tools in input. Hosted tools
    # in top-level tools could execute remotely, so do not replay those requests.
    return (isinstance(payload, dict) and payload.get('store') is False
            and not payload.get('background') and not payload.get('previous_response_id')
            and not payload.get('conversation') and payload.get('tools', []) == []
            and isinstance(payload.get('input'), list))


def transient(record):
    status = record.get('http_status')
    if status is not None and status not in {200, 408, 429, 500, 502, 503, 504}:
        return False
    if record.get('response_event') in {'response.completed', 'response.incomplete', 'response.failed', 'error'}:
        return False
    return (record.get('outcome') in NETWORK_ERRORS or status in {408, 429, 500, 502, 503, 504}
            or (status == 200 and record.get('outcome') == 'http_or_response_not_completed'))


def request_reliably(request, body, key, proxy, event_path, *, attempt_sink,
                     max_attempts=4, max_preupload_attempts=None,
                     proxy_candidates=None,
                     sleeper=time.sleep, clock=time.monotonic,
                     jitter=random.uniform, **options):
    """One overall deadline, one delivered response, one ledger row per attempt."""
    allowed = replayable(body, options)
    deadline = clock() + float(options.get('overall_timeout', 240))
    logical_id = uuid.uuid4().hex
    downstream_headers = options.pop('on_headers', None)
    downstream_chunk = options.pop('on_chunk', None)
    preupload_limit = max_attempts if max_preupload_attempts is None else max_preupload_attempts
    uploaded_attempts = 0
    preupload_attempts = 0
    for attempt in range(1, max_attempts + preupload_limit + 1):
        headers = []; chunks = bytearray()
        def collect_headers(status, values):
            headers[:] = [status, values]
        def collect_chunk(chunk):
            if len(chunks) + len(chunk) > 8 * 1048576:
                raise ValueError('Buffered response size limit')
            chunks.extend(chunk)
        remaining = max(.01, deadline - clock())
        current = dict(options, request_id=uuid.uuid4().hex, overall_timeout=remaining,
                       on_headers=collect_headers, on_chunk=collect_chunk)
        # Also bound stage timeouts so retry never multiplies the total deadline.
        for name, default in [('connect_timeout', 5), ('proxy_connect_timeout', 30),
                              ('tls_timeout', 12), ('upload_timeout', 90), ('read_timeout', 120)]:
            current[name] = min(float(options.get(name, default)), remaining)
        selected_proxy = proxy_candidates[(attempt - 1) % len(proxy_candidates)] if proxy_candidates else proxy
        result = request(body, key, selected_proxy, event_path, **current)
        complete = result.get('outcome') == 'completed'
        preupload = (result.get('safe_to_retry_before_api_upload') is True
                     and not result.get('api_request_may_have_been_sent'))
        if preupload:
            preupload_attempts += 1
        else:
            uploaded_attempts += 1
        retryable = transient(result) and (allowed or preupload)
        delay = min(30, 2 ** attempt) + jitter(0, 1)
        retry_after = result.get('retry_after_seconds')
        if isinstance(retry_after, (int, float)):
            delay = max(delay, retry_after)
        within_attempt_limit = (preupload_attempts < preupload_limit if preupload
                                else uploaded_attempts < max_attempts)
        retry = not complete and retryable and within_attempt_limit and clock()+delay < deadline
        result.update(transport_revision=REVISION, logical_request_id=logical_id,
                      connection_attempt=attempt, retry_scheduled=retry,
                      uploaded_attempts=uploaded_attempts, uploaded_attempt_limit=max_attempts,
                      preupload_failures=preupload_attempts, preupload_attempt_limit=preupload_limit,
                      buffered_until_terminal=True,
                      possible_unreported_provider_usage=bool(
                          result.get('api_request_may_have_been_sent') and not result.get('usage')))
        # The same attempt never appears in two ledgers or gets added twice.
        attempt_sink(dict(result))
        if retry:
            sleeper(delay)
            continue
        # An incomplete stream must never expose half a generated tool call.
        # Complete provider errors (400/auth/etc.) are forwarded unchanged.
        terminal_error = (headers and (headers[0] != 200 or result.get('response_event') in
                                       {'response.incomplete', 'response.failed', 'error'}))
        if headers and (complete or terminal_error):
            if downstream_headers:
                downstream_headers(*headers)
            if downstream_chunk and chunks:
                downstream_chunk(bytes(chunks))
        return result
