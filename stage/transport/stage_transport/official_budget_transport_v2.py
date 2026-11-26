"""Official API only, with a shared campaign budget guard."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import secrets

@contextmanager
def official_budget_transport_v2(private, public, *, direct_hosts, boundary, allowed_names):
    import rpent.planner.codex as codex_module
    previous = dict(os.environ)
    key = secrets.token_urlsafe(32)  # local bridge authentication only
    base = 'https://api.openai.com/v1'
    os.environ['PLANNER_BASE_URL'] = base
    os.environ['PLANNER_ACCESS_TOKEN'] = key
    # The existing transport binds its upstream origin when imported.
    from stage_transport import timed_provider_bridge as bridge_module
    from stage_transport.official_budget_route_v2 import routed_request
    original_request = bridge_module.request_reliably
    bridge_module.request_reliably = routed_request
    TimedProviderBridge = bridge_module.TimedProviderBridge
    from stage_transport.planner_auth import API_AUTH
    original_config = codex_module._codex_mcp_config_overrides
    with tempfile.TemporaryDirectory(prefix='stage-api-home-') as home:
        for name in ('CODEX_THREAD_ID', 'CODEX_REMOTE_PAYLOAD', 'CODEX_PERMISSION_PROFILE',
                     'CODEX_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL',
                     'STAGE_CODEX_PROVIDER_KEY', 'PLANNER_ENV_FILE', 'STAGE_CODEX_HOME'):
            os.environ.pop(name, None)
        direct = ','.join(dict.fromkeys(['127.0.0.1', 'localhost', '::1', *direct_hosts]))
        applied = {'CODEX_HOME': home, 'CODEX_BIN': previous.get('CODEX_BIN', 'codex'),
                   'STAGE_CODEX_HOME_ISOLATED': '1', 'STAGE_CODEX_HOME_AUTH_MODE': API_AUTH,
                   'STAGE_PRIVATE_OUTPUT_DIR': str(private), 'NO_PROXY': direct, 'no_proxy': direct}
        if public is not None:
            applied['STAGE_PUBLIC_OUTPUT_DIR'] = str(Path(public).resolve())
        os.environ.update(applied)
        bridge = TimedProviderBridge(base, output=Path(private)/'transport',
            proxy=previous.get('STAGE_PLANNER_HTTP_PROXY', ''),
            request_transform=boundary, auth_mode=API_AUTH,
            request_options={'read_timeout': 300, 'overall_timeout': 900, 'tls_timeout': 12,
                             'upload_timeout': 90, 'stop_at_sse_terminal': True,
                             'max_attempts': 1, 'max_preupload_attempts': 1})
        applied['CODEX_BASE_URL'] = bridge.start()
        os.environ.update(applied)

        from rpent.planner.utils.http_mcp_server import SERVER_NAME as _server
        _prefix = f'mcp__{_server}__'
        _provider_key = f'{_server}_proxy'

        def api_config(*, mcp_url, base_url):
            values = original_config(mcp_url=mcp_url, base_url=base_url)
            values = [v for v in values if not v.startswith(f'model_providers.{_provider_key}.env_key=')]
            if mcp_url:
                names = sorted(n.removeprefix(_prefix) for n in allowed_names())
                if not names:
                    raise RuntimeError('MCP declarations unavailable')
                values.append(f'mcp_servers.{_server}.enabled_tools={json.dumps(names)}')
                values.append(f'mcp_servers.{_server}.required=true')
                values.extend(f'mcp_servers.{_server}.tools.{n}.approval_mode="approve"' for n in names)
            return values + [f'model_providers.{_provider_key}.env_key="PLANNER_ACCESS_TOKEN"',
                             f'model_providers.{_provider_key}.requires_openai_auth=false',
                             f'model_providers.{_provider_key}.request_max_retries=0',
                             f'model_providers.{_provider_key}.stream_max_retries=0']

        codex_module._codex_mcp_config_overrides = api_config
        from research.subgoal_images import _window as image_window
        (Path(private)/'planner_provider.json').write_text(json.dumps({
            'provider': 'openai_official_budgeted_v2', 'budget_file': os.environ['STAGE_API_BUDGET_FILE'], 'result_tag': 'api', 'base_url': base,
            'model': 'gpt-6-astra', 'reasoning_effort': boundary.expected_effort, 'subscription_used': False,
            'credential_source': 'container-local protected bundle; actual provider recorded for every request',
            'image_history_policy': image_window.config if image_window else 'over-50-keep-latest-48/v1'}, indent=2)+'\n')
        try:
            yield applied
        finally:
            codex_module._codex_mcp_config_overrides = original_config
            bridge_module.request_reliably = original_request
            bridge.stop()
            os.environ.clear()
            os.environ.update(previous)
