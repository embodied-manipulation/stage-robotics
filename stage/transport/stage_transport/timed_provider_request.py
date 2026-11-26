"""Diagnostic Responses client with per-attempt phase timing; never logs bodies.

This uses Python/OpenSSL, not Codex's native TLS stack. Keep that distinction in
comparisons. Upload completion means local SSL writes completed, not server ACK.
No automatic retries occur after an API request may have been sent.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import os
import json
import math
import re
import statistics
from pathlib import Path
import socket
import ssl
import time
import uuid
from urllib.parse import urlsplit

ORIGIN = urlsplit(os.environ.get('PLANNER_BASE_URL', 'https://api.example.com/v1')).hostname


def rate_limit_metadata(response):
    names = ('retry-after', 'x-ratelimit-limit-requests', 'x-ratelimit-limit-tokens',
             'x-ratelimit-remaining-requests', 'x-ratelimit-remaining-tokens',
             'x-ratelimit-reset-requests', 'x-ratelimit-reset-tokens')
    headers = {name: value[:256] for name in names
               if (value := response.getheader(name)) is not None}
    retry_after = headers.get('retry-after')
    seconds = None
    if retry_after is not None:
        try:
            seconds = max(0.0, float(retry_after))
        except ValueError:
            try:
                seconds = max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
        if seconds is not None and not math.isfinite(seconds):
            seconds = None
    return {'rate_limit_headers': headers, 'retry_after_seconds': seconds}


def phase_durations(record):
    events = record.get('events', {})
    pairs = {
        'request_seconds': ('request_start', 'request_end'),
        'connection_seconds': ('tcp_connect_start', 'upload_start'),
        'upload_seconds': ('upload_start', 'upload_complete'),
        'post_upload_response_seconds': ('upload_complete', 'response_terminal_event'),
        'time_to_first_model_delta_seconds': ('upload_complete', 'first_model_delta'),
        'model_stream_seconds': ('first_model_delta', 'response_terminal_event'),
    }
    return {name: events[end] - events[start] for name, (start, end) in pairs.items()
            if start in events and end in events}


def summarize_timings(records):
    metrics = {}
    for record in records:
        for name, value in phase_durations(record).items():
            metrics.setdefault(name, []).append(value)
    def stats(values):
        values = sorted(values)
        return {'count': len(values), 'total': sum(values), 'mean': statistics.mean(values),
                'p50': statistics.median(values), 'p95': values[min(len(values)-1, int(.95*len(values)))],
                'max': max(values)}
    return {'requests': len(records), 'completed': sum(r.get('outcome') == 'completed' for r in records),
            'failed_or_incomplete': sum(r.get('outcome') != 'completed' for r in records),
            'metrics_seconds': {name: stats(values) for name, values in metrics.items()},
            'request_body_bytes': sum(r.get('body_bytes', 0) for r in records),
            'pure_server_inference_seconds': None,
            'measurement_notes': {
                'upload_seconds': 'Client SSL writes completed, not server receipt acknowledgement.',
                'post_upload_response_seconds': 'Client-observed Planner response latency: upstream transfer, queue, prefill, reasoning/generation and downstream transfer; pure server inference is unavailable.',
                'failed_requests': 'Known phase durations are included; missing phases are not treated as zero.',
                'compact': 'Compaction requests are included in request/upload totals; no SSE inference duration is invented.'}}


class PhaseLog:
    def __init__(self, path, request_id=None, **metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.record = {'request_id': request_id or uuid.uuid4().hex,
                       'started_at': datetime.now(timezone.utc).isoformat(),
                       'transport': 'python_openssl_diagnostic', 'openssl': ssl.OPENSSL_VERSION,
                       'events': {}, **metadata}

    def mark(self, name, **values):
        elapsed = time.monotonic()-self.started
        self.record['events'].setdefault(name, elapsed)
        self.record.update(values)
        with self.path.open('a') as stream:
            stream.write(json.dumps({'request_id': self.record['request_id'], 'event': name,
                                    'elapsed_seconds': elapsed, **values})+'\n')


def _connect_response(sock):
    # Do not read TLS bytes past the CONNECT headers.
    data = bytearray()
    while not data.endswith(b'\r\n\r\n'):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError('EOF in CONNECT headers')
        data.extend(chunk)
        if len(data) > 16384:
            raise ValueError('Oversized CONNECT response')
    return int(bytes(data).split(b'\r\n', 1)[0].split()[1])


class BufferedSocket:
    def __init__(self, reader):
        self.reader = reader

    def makefile(self, *args, **kwargs):
        return self.reader


def open_tunnel(trace, proxy, *, origin=ORIGIN, origin_port=443, context=None, connect_timeout=5, proxy_connect_timeout=30, tls_timeout=12, deadline=None):
    def timeout(limit):
        if deadline is None:
            return limit
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Overall deadline')
        return min(limit, remaining)
    parsed = urlsplit(proxy)
    if parsed.scheme != 'http' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Explicit credential-free HTTP proxy required')
    context = context or ssl.create_default_context()
    trace.mark('tcp_connect_start', proxy=f'{parsed.hostname}:{parsed.port or 80}')
    sock = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=timeout(connect_timeout))
    try:
        trace.mark('tcp_connect_complete', proxy_peer_address=sock.getpeername()[0])
        trace.mark('connect_request_start', proxy_connect_timeout_seconds=proxy_connect_timeout)
        sock.settimeout(timeout(proxy_connect_timeout))
        sock.sendall(f'CONNECT {origin}:{origin_port} HTTP/1.1\r\nHost: {origin}:{origin_port}\r\n\r\n'.encode())
        status = _connect_response(sock)
        trace.mark('connect_response', connect_status=status)
        if status != 200:
            raise ConnectionRefusedError('Proxy CONNECT rejected')
        trace.mark('tls_start')
        sock.settimeout(timeout(tls_timeout))
        sock = context.wrap_socket(sock, server_hostname=origin, do_handshake_on_connect=False)
        sock.do_handshake()
        trace.mark('tls_complete', tls_version=sock.version(), alpn=sock.selected_alpn_protocol())
        return sock
    except BaseException:
        sock.close()
        raise


def _error_metadata(error, key):
    message = str(error.get('message') or error.get('detail') or '').replace(key, '[REDACTED]')
    message = re.sub(r'(?i)Bearer\s+[^\s"<>]+|data:image/[^\s"<>]+|sk-[A-Za-z0-9_-]+', '[REDACTED]', message)
    return {'api_error_type': error.get('type'), 'api_error_code': error.get('code'),
            'api_error_param': error.get('param'), 'api_error_message': message[:800]}


def _sse_event(data, trace, key):
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return
    kind = obj.get('type')
    if (obj.get('response') or {}).get('model'):
        trace.record['response_model'] = obj['response']['model']
    if 'first_sse_event' not in trace.record['events']:
        trace.mark('first_sse_event', first_sse_type=kind)
    if isinstance(kind, str) and kind.endswith('.delta') and 'first_model_delta' not in trace.record['events']:
        trace.mark('first_model_delta', first_delta_type=kind)
    if kind in {'response.completed', 'response.incomplete', 'response.failed', 'error'}:
        response = obj.get('response') or {}
        trace.mark('response_terminal_event', response_event=kind, response_status=response.get('status'))
        trace.mark(kind.replace('.', '_'))
        usage = response.get('usage') or {}
        trace.record['usage'] = {k: usage[k] for k in ['input_tokens', 'output_tokens', 'total_tokens',
                                  'input_tokens_details', 'output_tokens_details'] if k in usage}
        error = response.get('error') or obj.get('error') or (obj if kind == 'error' else {})
        if error:
            details = _error_metadata(error, key)
            trace.record.update(details)
            trace.record.setdefault('api_errors', []).append({'event_type': kind,
                'elapsed_seconds': time.monotonic()-trace.started, **details})


def timed_request(body, key, proxy, event_path, *, request_id=None, label='', context=None,
                  connect_timeout=5, proxy_connect_timeout=30, tls_timeout=12, upload_timeout=90, read_timeout=120,
                  overall_timeout=240, method='POST', path='/v1/responses',
                  origin=ORIGIN, origin_port=443, on_headers=None, on_chunk=None, forward_headers=None,
                  stop_at_sse_terminal=False, tls_maximum_version=None):
    if origin not in {ORIGIN, 'chatgpt.com', 'localhost', 'api.openai.com'} or origin_port not in {443, 8443}:
        raise ValueError('Unexpected API origin')
    prefix = '/backend-api/codex' if origin == 'chatgpt.com' else '/v1'
    if path not in {prefix + '/responses', prefix + '/responses/compact'}:
        raise ValueError('Unexpected API path')
    if not key or '\r' in key or '\n' in key:
        raise ValueError('Invalid credential')
    if tls_maximum_version is not None:
        if tls_maximum_version != 'TLSv1_2' or origin != ORIGIN:
            raise ValueError('TLS compatibility override is restricted to TLS 1.2 on configured origin')
        context = context or ssl.create_default_context()
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    trace = PhaseLog(event_path, request_id, label=label, body_bytes=len(body), api_path=path)
    trace.mark('request_start')
    sock = reader = response = None
    stage = 'connect'
    try:
        sock = open_tunnel(trace, proxy, origin=origin, origin_port=origin_port, context=context,
                           connect_timeout=connect_timeout, proxy_connect_timeout=proxy_connect_timeout,
                           tls_timeout=tls_timeout, deadline=trace.started+overall_timeout)
        stage = 'upload'
        headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
        excluded = {'host', 'authorization', 'connection', 'content-length', 'transfer-encoding',
                    'accept-encoding', 'proxy-authorization', 'proxy-connection'}
        for name, value in (forward_headers or {}).items():
            if name.lower() not in excluded and not any(c in name+value for c in '\r\n'):
                # HTTP header names are case-insensitive. Native Codex sends
                # lowercase names; two Content-Type fields are rejected by
                # the subscription endpoint even when their values match.
                for existing in list(headers):
                    if existing.lower() == name.lower():
                        del headers[existing]
                headers[name] = value
        headers.update({'Host': origin if origin_port==443 else f'{origin}:{origin_port}', 'Authorization': f'Bearer {key}', 'Accept-Encoding': 'identity',
                        'Content-Length': str(len(body)), 'Connection': 'close'})
        header = (f'{method} {path} HTTP/1.1\r\n'+''.join(f'{name}: {value}\r\n' for name,value in headers.items())+'\r\n').encode()
        trace.mark('upload_start', api_request_may_have_been_sent=True, body_bytes_written=0)
        sock.settimeout(min(upload_timeout, max(.01, trace.started+overall_timeout-time.monotonic())))
        sock.sendall(header)
        upload_deadline = min(time.monotonic()+upload_timeout, trace.started+overall_timeout)
        view = memoryview(body)
        written = 0
        while written < len(body):
            remaining = upload_deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Upload deadline')
            sock.settimeout(remaining)
            count = sock.send(view[written:written+65536])
            if not count:
                raise ConnectionError('EOF during upload')
            written += count
            trace.record['body_bytes_written'] = written
        trace.mark('upload_complete', body_bytes_written=written,
                   upload_completion_semantics='client SSL writes completed; not server receipt acknowledgement')
        stage = 'first_response'
        sock.settimeout(min(read_timeout, max(.01, trace.started+overall_timeout-time.monotonic())))
        reader = sock.makefile('rb')
        if not reader.peek(1):
            raise ConnectionError('EOF before HTTP response')
        trace.mark('first_http_byte')
        response = http.client.HTTPResponse(BufferedSocket(reader))
        response.begin()
        trace.mark('http_headers_complete', http_status=response.status,
                   response_content_type=response.getheader('Content-Type'),
                   provider_request_id=response.getheader('x-request-id'),
                   **rate_limit_metadata(response))
        if on_headers:
            on_headers(response.status, response.getheaders())
        stage = 'response_body'
        pending = b''
        error_body = bytearray()
        received = 0
        terminal_event_delimited = False
        while True:
            remaining = trace.started+overall_timeout-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Overall deadline')
            sock.settimeout(min(read_timeout, remaining))
            chunk = response.read1(65536)
            if not chunk:
                break
            if received == 0:
                trace.mark('first_response_body_byte')
            received += len(chunk)
            trace.record['response_body_bytes'] = received
            if on_chunk:
                on_chunk(chunk)
            if response.status != 200 and len(error_body) < 65536:
                error_body.extend(chunk[:65536-len(error_body)])
            pending += chunk
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                if line.startswith(b'data:'):
                    _sse_event(line[5:].strip(), trace, key)
                elif not line.strip() and trace.record.get('response_event'):
                    terminal_event_delimited = True
            # Some proxies leave HTTP chunk framing open after a complete Responses
            # terminal event. The delimited SSE event is the model response's
            # completion boundary; no additional generation or tool replay is needed.
            if stop_at_sse_terminal and response.status == 200 and terminal_event_delimited:
                trace.mark('http_body_stopped_at_sse_terminal')
                break
            if len(pending) > 1048576:
                # Do not retain unlimited non-SSE content in diagnostics.
                pending = b''
            if received > 8*1048576:
                raise ValueError('Diagnostic response size cap')
        trace.mark('response_complete', response_body_bytes=received)
        if error_body:
            try:
                obj = json.loads(error_body)
                trace.record.update(_error_metadata(obj.get('error') or obj, key))
            except (ValueError, AttributeError):
                trace.record['api_error_body_not_json'] = True
        trace.record['outcome'] = ('completed' if response.status == 200 and
                                   (trace.record.get('response_event') == 'response.completed'
                                    or path.endswith('/responses/compact'))
                                   else 'http_or_response_not_completed')
    except Exception as error:
        if stage == 'connect' and 'tls_start' in trace.record['events']:
            stage = 'tls_handshake'
        elif stage == 'connect' and 'connect_request_start' in trace.record['events']:
            stage = 'proxy_connect'
        trace.mark('request_error', outcome=type(error).__name__, failed_stage=stage)
        trace.record['safe_to_retry_before_api_upload'] = not trace.record.get('api_request_may_have_been_sent', False)
    finally:
        for value in (response, reader, sock):
            if value:
                value.close()
        trace.mark('request_end')
        trace.record['timing_seconds'] = phase_durations(trace.record)
    return trace.record
