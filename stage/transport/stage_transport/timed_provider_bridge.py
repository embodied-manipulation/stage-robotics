"""Loopback bridge for native Codex requests and optional declared input filtering.

TLS is Python/OpenSSL on the outbound leg. Request body bytes and end-to-end
request headers are preserved; HTTP framing and Accept-Encoding are normalized.
Without request_transform this preserves the existing diagnostic transport.
An observation-window variant explicitly changes only the request input.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
import re
import ssl
from pathlib import Path
import threading
import uuid

from stage_transport.timed_provider_request import ORIGIN, timed_request, summarize_timings
from stage_transport.reliable_response import request_reliably
from stage_transport.planner_auth import API_AUTH, SUBSCRIPTION_AUTH, SUBSCRIPTION_BASE, subscription_auth_file


def input_shape(value):
    counts = {'input_image_count': 0, 'image_data_url_characters': 0,
              'input_text_characters': 0, 'base64_like_characters_in_text': 0}

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'input_image':
                counts['input_image_count'] += 1
                url = obj.get('image_url')
                if isinstance(url, str) and url.startswith('data:'):
                    counts['image_data_url_characters'] += len(url)
            for name, item in obj.items():
                if name in {'text', 'output', 'content'} and isinstance(item, str):
                    counts['input_text_characters'] += len(item)
                    counts['base64_like_characters_in_text'] += sum(len(m[0]) for m in re.finditer(r'[A-Za-z0-9+/]{1000,}={0,2}', item))
                elif isinstance(item, (dict, list)):
                    walk(item)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
    walk(value)
    return counts


class TimedProviderBridge:
    def __init__(self, upstream, *, output, proxy=None,
                 request_transform=None, request_options=None, request_pacing=None,
                 auth_mode=API_AUTH, auth_file=None):
        proxy = proxy or os.environ.get('PLANNER_HTTP_PROXY', '')
        expected = {API_AUTH: f'https://{ORIGIN}/v1', SUBSCRIPTION_AUTH: SUBSCRIPTION_BASE}
        if auth_mode not in expected or upstream.rstrip('/') != expected[auth_mode]:
            raise ValueError('Unexpected origin')
        self.auth_mode = auth_mode
        self.auth_file = (subscription_auth_file(auth_file or Path(os.environ['CODEX_HOME']) / 'auth.json')
                          if auth_mode == SUBSCRIPTION_AUTH else None)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.request_transform = request_transform
        self.request_options = dict(request_options or {})
        self.request_pacing = request_pacing
        if set(self.request_options) - {'read_timeout', 'overall_timeout', 'tls_timeout', 'upload_timeout',
                                        'stop_at_sse_terminal', 'max_attempts', 'max_preupload_attempts',
                                        'tls_maximum_version', 'proxy_candidates'}:
            raise ValueError('unsupported transport timeout option')
        self.lock = threading.Lock()
        api_key = os.environ['PLANNER_ACCESS_TOKEN'] if auth_mode == API_AUTH else None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                # Re-read the private login file after native Codex refreshes it.
                key = (json.loads(outer.auth_file.read_text())['tokens']['access_token']
                       if outer.auth_file is not None else api_key)
                allowed_paths = {'/v1/responses', '/v1/responses/compact'}
                if self.path not in allowed_paths or not hmac.compare_digest(
                        self.headers.get('Authorization', ''), f'Bearer {key}'):
                    self.send_error(403)
                    return
                self.connection.settimeout(120)
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 150*1024*1024:
                        raise ValueError('Request size')
                    body = self.rfile.read(size)
                    if len(body) != size:
                        raise ValueError('Incomplete request')
                except (ValueError, OSError):
                    self.send_error(400)
                    return
                metadata = {'request_body_unchanged': True, 'path': self.path, 'auth_mode': auth_mode,
                            'original_body_bytes': len(body), 'request_header_names': sorted(
                            name for name in self.headers if name.lower() not in {'authorization', 'proxy-authorization'})}
                try:
                    payload = json.loads(body)
                    metadata.update(model=payload.get('model'), reasoning=payload.get('reasoning'),
                                    service_tier=payload.get('service_tier'),
                                    previous_response_id_present=bool(payload.get('previous_response_id')),
                                    input_items=len(payload.get('input', [])),
                                    tool_types=[v.get('type') for v in payload.get('tools', [])])
                    metadata.update(input_shape(payload.get('input', [])))
                except (ValueError, TypeError, AttributeError):
                    pass
                if outer.request_transform is not None:
                    try:
                        payload, changes = outer.request_transform(json.loads(body))
                        body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
                        metadata.update(changes, request_body_unchanged=False,
                                        outgoing_body_bytes=len(body),
                                        outgoing_input_shape=input_shape(payload.get('input', [])))
                    except (ValueError, TypeError, KeyError) as error:
                        # No request body, image bytes, provider key or exception payload is logged.
                        failure = {'outcome': 'request_transform_rejected', 'path': self.path,
                                   'error_type': type(error).__name__}
                        with outer.lock:
                            outer.records.append(failure)
                            with (outer.output/'request_records.jsonl').open('a') as stream:
                                stream.write(json.dumps(failure)+'\n')
                        self.send_error(400, 'Observation window rejected this request')
                        return
                headers_sent = False

                def on_headers(status, headers):
                    nonlocal headers_sent
                    self.send_response(status)
                    for name, value in headers:
                        if name.lower() not in {'transfer-encoding', 'connection', 'content-length'}:
                            self.send_header(name, value)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    headers_sent = True

                def on_chunk(chunk):
                    self.wfile.write(chunk)
                    self.wfile.flush()

                def record_attempt(result):
                    result.update(metadata)
                    with outer.lock:
                        outer.records.append(result)
                        with (outer.output/'request_records.jsonl').open('a') as stream:
                            stream.write(json.dumps(result)+'\n')
                        temporary = outer.output / 'timing_summary.tmp'
                        temporary.write_text(json.dumps(summarize_timings(outer.records), indent=2)+'\n')
                        temporary.replace(outer.output / 'timing_summary.json')

                from stage_transport.planner_request_pacing import request_slot
                with request_slot(outer.request_pacing) as pacing:
                    result = request_reliably(timed_request, body, key, proxy, outer.output/'phase_events.jsonl',
                                           attempt_sink=record_attempt,
                                           label='observation_window' if outer.request_transform else 'native_codex_body',
                                           forward_headers=dict(self.headers), on_headers=on_headers, on_chunk=on_chunk,
                                           origin='chatgpt.com' if auth_mode == SUBSCRIPTION_AUTH else ORIGIN,
                                           origin_port=443,
                                           context=None,
                                           path=self.path.replace('/v1/', '/backend-api/codex/', 1)
                                                if auth_mode == SUBSCRIPTION_AUTH else self.path,
                                           **outer.request_options)
                    pacing['api_error_code'] = result.get('api_error_code')
                    pacing['retry_after_seconds'] = result.get('retry_after_seconds')
                    pacing['request_succeeded'] = result.get('outcome') == 'completed'
                if outer.request_pacing:
                    result['request_pacing'] = {**outer.request_pacing, **pacing}
                if not headers_sent:
                    self.send_response(502)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"Model response interrupted after bounded retries; see transport ledger"}}')
                self.close_connection = True

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
