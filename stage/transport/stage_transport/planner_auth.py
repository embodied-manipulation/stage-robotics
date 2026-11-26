"""Explicit Planner authentication routes, without copying account secrets."""
from pathlib import Path
import json
import os

from .deployment_config import python_environment

API_AUTH = 'api_environment_token'
SUBSCRIPTION_AUTH = 'chatgpt_subscription'
SUBSCRIPTION_BASE = 'https://chatgpt.com/backend-api/codex'
API_ENV_KEYS = ('PLANNER_ACCESS_TOKEN', 'PLANNER_BASE_URL', 'PLANNER_ENV_FILE', 'CODEX_API_KEY',
                'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'STAGE_CODEX_PROVIDER_KEY')


def clear_api_credentials(env):
    for name in API_ENV_KEYS:
        env.pop(name, None)


def planner_environment(config):
    env = python_environment()
    mode = config['planner'].get('auth_mode', API_AUTH)
    if mode == SUBSCRIPTION_AUTH:
        clear_api_credentials(env)
    elif mode == API_AUTH:
        env['PLANNER_ENV_FILE'] = config['credential_file']
        env['PLANNER_BASE_URL'] = config['planner']['base_url']
    else:
        raise ValueError('Unknown Planner authentication mode')
    env['STAGE_PLANNER_AUTH_MODE'] = mode
    return env


def experiment_planner(planner):
    """Only authentication and endpoint may differ across this continuation."""
    return {k: v for k, v in planner.items() if k not in {'auth_mode', 'base_url', 'desktop_auth_used'}}


def provider_matches(planner, manifest):
    """Validate the declared route for historical API and subscription episodes."""
    if manifest.get('planner_api_base_url') != planner.get('base_url'):
        return False
    mode = planner.get('auth_mode', API_AUTH)
    if mode == SUBSCRIPTION_AUTH:
        return (planner.get('base_url') == SUBSCRIPTION_BASE
                and manifest.get('codex_home_auth_mode') == SUBSCRIPTION_AUTH
                and manifest.get('planner_api_key_source') is None
                and manifest.get('desktop_auth_used') is True)
    return mode == API_AUTH and manifest.get('planner_api_key_source') == 'PLANNER_ACCESS_TOKEN'


def subscription_auth_file(path):
    path = Path(path)
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o777 != 0o600:
        raise ValueError('Subscription auth must be an owned mode-600 file')
    data = json.loads(path.read_text())
    if data.get('auth_mode') != 'chatgpt' or not data.get('tokens', {}).get('access_token'):
        raise ValueError('Current account is not signed in with ChatGPT')
    return path


def link_subscription_auth(home, source):
    """Share only login state; no desktop config, messages or session DBs."""
    source = subscription_auth_file(source).resolve()
    target = Path(home) / 'auth.json'
    target.symlink_to(source)
    return target


def subscription_provider_config(base_url, provider='rpent_proxy'):
    # A custom Responses provider retains the experiment's HTTP/SSE transport.
    # requires_openai_auth lets native Codex supply and refresh ChatGPT tokens.
    return {'model_provider': provider,
            f'model_providers.{provider}.name': provider,
            f'model_providers.{provider}.base_url': base_url.rstrip('/'),
            f'model_providers.{provider}.wire_api': 'responses',
            f'model_providers.{provider}.requires_openai_auth': True,
            'cli_auth_credentials_store': 'file', 'forced_login_method': 'chatgpt'}
