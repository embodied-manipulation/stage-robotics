"""Experiment-only subscription transport and observed capability ledger.

Reuses the already deployed Python/OpenSSL transport for both control arms.
Credentials and desktop session data are never copied into rollout artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading


TRANSPORT_RUNTIME = Path(os.environ.get(
    'STAGE_TRANSPORT_RUNTIME',
    str(Path(__file__).resolve().parents[2] / 'transport'),
))

_allowed_names: set[str] = set()


def _mcp_server_name() -> str:
    try:
        from rpent.planner.utils.http_mcp_server import SERVER_NAME
        return SERVER_NAME
    except Exception:
        return "rpent"


def set_allowed_tools(declarations) -> None:
    """Bind the current episode's native MCP declarations before first request."""
    global _allowed_names
    prefix = f"mcp__{_mcp_server_name()}__"
    _allowed_names = {f"{prefix}{tool.name}" for tool in declarations}


def _tool_inventory(tools: list) -> list[dict]:
    inventory = []
    for tool in tools:
        kind, name = tool.get("type"), tool.get("name", "")
        if kind == "namespace":
            for inner in tool.get("tools", []):
                inventory.append({"type": inner.get("type"), "name": inner.get("name", ""), "namespace": name})
                if inner.get("name") == "exec":
                    inventory.extend(
                        {"type": "nested", "name": nested, "namespace": "tools"}
                        for nested in re.findall(r"declare const tools: \{ (\w+)\(", inner.get("description", ""))
                    )
        else:
            inventory.append({"type": kind, "name": name, "namespace": ""})
    return inventory


def _recordable(value):
    """Keep all model-visible text without duplicating image pixels each turn."""
    if isinstance(value, dict):
        return {key: _recordable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_recordable(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_payload_characters": len(value), "pixels": "see public observation artifacts"}
    return value


class RequestBoundary:
    """Record actual SDK model inputs; reject capabilities outside the MCP set."""

    def __init__(self, directory: Path, *, enforce: bool, expected_effort="low"):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.enforce = enforce
        self.expected_effort = expected_effort
        self.counter = 0
        self.lock = threading.Lock()

    def __call__(self, payload):
        image_metadata = {}
        with self.lock:
            self.counter += 1
            sequence = self.counter
        declarations = list(payload.get("tools", []))
        for item in payload.get("input", []):
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                declarations.extend(item.get("tools", []))
        inventory = _tool_inventory(declarations)
        # Native Codex compaction retains conversation text but deliberately
        # removes executable tool declarations. This adds no capabilities.
        last = payload.get('input', [])[-1:] or [{}]
        last = last[0]
        compaction = (not inventory and last.get('type') == 'message'
                      and last.get('role') == 'user'
                      and any(part.get('type') == 'input_text' and part.get('text', '').startswith(
                          'You are performing a CONTEXT CHECKPOINT COMPACTION.')
                          for part in last.get('content', []) if isinstance(part, dict)))
        from research.subgoal_images import _window as images, without_images
        if images is not None:
            if compaction:
                payload = {**payload, 'input': without_images(payload.get('input', []))}
            else:
                payload, image_metadata = images.rewrite(payload)
        # Record all text/capabilities privately; images already exist in public
        # artifacts. This ledger contains no HTTP authentication headers.
        target = self.directory / f"request_{sequence:04d}.json"
        target.write_text(json.dumps(_recordable(payload), ensure_ascii=False))
        wrappers = {("custom", "exec"), ("function", "wait"), ("function", "request_user_input")}
        # Astra uses a V8 dispatcher. Deferred MCP tools are constrained by
        # enabled_tools in the isolated, single-server SDK configuration below.
        # The resource helpers can only see this server (no resources offered).
        resource_helpers = {"list_mcp_resource_templates", "list_mcp_resources", "read_mcp_resource"}
        invalid = [item for item in inventory if not (
            (item["type"] in {"function", "nested"} and item["name"] in _allowed_names)
            or (item["namespace"] == "functions" and (item["type"], item["name"]) in wrappers)
            or (item["type"] == "nested" and item["name"] in resource_helpers)
        )]
        if payload.get("model") != "gpt-6-astra":
            raise ValueError("Experiment requires gpt-6-astra")
        if (payload.get("reasoning") or {}).get("effort") != self.expected_effort:
            raise ValueError(f"Experiment requires reasoning effort {self.expected_effort}")
        receipt = {
            "request": sequence, "model": payload.get("model"),
            "reasoning": payload.get("reasoning"), "inventory": inventory,
            "allowed_mcp_tools": sorted(_allowed_names),
            "deferred_tools_boundary": "isolated SDK home; single MCP server; explicit enabled_tools",
            "boundary_enforced": self.enforce,
            "unrecognized_capabilities": invalid,
            "tool_free_compaction": compaction,
        }
        with (self.directory / "capabilities.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        if self.enforce and (invalid or (not inventory and not compaction)):
            raise ValueError("SDK advertised capabilities outside the episode tool set")
        from research.review_window import on_planner_request
        if not compaction:
            on_planner_request(payload)
        return payload, {"capability_boundary_passed": not invalid and (bool(inventory) or compaction),
                         "tool_free_compaction": compaction, "request_sequence": sequence,
                         **image_metadata}


@contextmanager
def planner_transport(private_dir, public_dir=None, *, enforce_boundary=True, direct_hosts=(), reasoning_effort="low"):
    """Yield applied env overrides for the declared Astra native CLI call."""
    private = Path(private_dir).resolve()
    private.mkdir(parents=True, exist_ok=True)
    if not (TRANSPORT_RUNTIME / 'stage_transport/timed_provider_bridge.py').is_file():
        raise FileNotFoundError('Planner transport source is missing. Recover it or set STAGE_TRANSPORT_RUNTIME to an explicitly validated surviving runtime before rollout.')
    sys.path.insert(0, str(TRANSPORT_RUNTIME))
    provider = os.environ.get('STAGE_PLANNER_PROVIDER', 'chatgpt_subscription')
    if provider == 'openai_official_budgeted_v2':
        from stage_transport.official_budget_transport_v2 import official_budget_transport_v2
        with official_budget_transport_v2(private, public_dir, direct_hosts=direct_hosts,
                              boundary=RequestBoundary(private / 'model_inputs', enforce=enforce_boundary, expected_effort=reasoning_effort),
                              allowed_names=lambda: _allowed_names) as applied:
            yield applied
        sys.path.remove(str(TRANSPORT_RUNTIME))
        return
    if provider == 'openai_official_budgeted':
        from stage_transport.official_budget_transport import official_budget_transport
        with official_budget_transport(private, public_dir, direct_hosts=direct_hosts,
                              boundary=RequestBoundary(private / 'model_inputs', enforce=enforce_boundary, expected_effort=reasoning_effort),
                              allowed_names=lambda: _allowed_names) as applied:
            yield applied
        sys.path.remove(str(TRANSPORT_RUNTIME))
        return
    if provider != 'chatgpt_subscription':
        raise ValueError(f'Unsupported Planner provider: {provider!r}')
    from stage_transport.planner_auth import (
        SUBSCRIPTION_AUTH, SUBSCRIPTION_BASE, link_subscription_auth,
    )
    from stage_transport.timed_provider_bridge import TimedProviderBridge
    import rpent.planner.codex as codex_module

    previous = dict(os.environ)
    original_config = codex_module._codex_mcp_config_overrides
    auth_source = Path(previous.get("STAGE_CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    no_proxy = ",".join(dict.fromkeys([
        "127.0.0.1", "localhost", "::1",
        *(host for key in ("NO_PROXY", "no_proxy") for host in previous.get(key, "").split(",") if host),
        *(host for host in direct_hosts if host),
    ]))
    with tempfile.TemporaryDirectory(prefix="stage-restricted-codex-home.") as home:
        link_subscription_auth(home, auth_source)
        for key in (
            "CODEX_THREAD_ID", "CODEX_REMOTE_PAYLOAD", "CODEX_PERMISSION_PROFILE",
            "CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "PLANNER_ACCESS_TOKEN",
            "STAGE_CODEX_PROVIDER_KEY", "PLANNER_BASE_URL", "PLANNER_ENV_FILE",
        ):
            os.environ.pop(key, None)
        env = {
            "CODEX_HOME": home,
            "CODEX_BIN": previous.get("CODEX_BIN", "codex"),
            "STAGE_CODEX_HOME_ISOLATED": "1",
            "STAGE_CODEX_HOME_AUTH_MODE": SUBSCRIPTION_AUTH,
            "STAGE_PRIVATE_OUTPUT_DIR": str(private),
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        if public_dir is not None:
            env["STAGE_PUBLIC_OUTPUT_DIR"] = str(Path(public_dir).resolve())
        os.environ.update(env)
        boundary = RequestBoundary(private / "model_inputs", enforce=enforce_boundary, expected_effort=reasoning_effort)
        request_options = {"read_timeout": 300, "overall_timeout": 900, "tls_timeout": 12, "upload_timeout": 90}
        if previous.get("STAGE_SUBSCRIPTION_SINGLE_ATTEMPT") == "1":
            request_options.update(max_attempts=1, max_preupload_attempts=1)
        bridge = TimedProviderBridge(
            SUBSCRIPTION_BASE, output=private / "transport",
            proxy=previous.get("STAGE_PLANNER_HTTP_PROXY", ""),
            request_transform=boundary,
            request_options=request_options,
            auth_mode=SUBSCRIPTION_AUTH, auth_file=Path(home) / "auth.json",
        )
        env["CODEX_BASE_URL"] = bridge.start()
        os.environ.update(env)

        server = _mcp_server_name()
        prefix = f"mcp__{server}__"
        provider_key = f"{server}_proxy"

        def subscription_config(*, mcp_url, base_url):
            values = original_config(mcp_url=mcp_url, base_url=base_url)
            values = [value for value in values if not value.startswith(f"model_providers.{provider_key}.env_key=")]
            if mcp_url:
                native_names = sorted(name.removeprefix(prefix) for name in _allowed_names)
                values.append(f"mcp_servers.{server}.enabled_tools={json.dumps(native_names)}")
                # Approval is confined to the current episode's declarations.
                values.extend(f'mcp_servers.{server}.tools.{name}.approval_mode="approve"' for name in native_names)
            return values + [
                f"model_providers.{provider_key}.requires_openai_auth=true",
                'cli_auth_credentials_store="file"', 'forced_login_method="chatgpt"',
                f"model_providers.{provider_key}.request_max_retries=0",
                f"model_providers.{provider_key}.stream_max_retries=0",
            ]

        codex_module._codex_mcp_config_overrides = subscription_config
        from research.subgoal_images import _window as image_window
        (private / 'planner_provider.json').write_text(json.dumps({
            'provider': 'chatgpt_subscription', 'result_tag': 'subscription',
            'model': 'gpt-6-astra', 'reasoning_effort': reasoning_effort,
            'subscription_used': True, 'base_url': SUBSCRIPTION_BASE,
            'credential_source': 'selected external ChatGPT auth.json; symlink in isolated episode home',
            'auth_home': str(auth_source.parent),
            'single_attempt_transport': previous.get("STAGE_SUBSCRIPTION_SINGLE_ATTEMPT") == "1",
            'image_history_policy': image_window.config if image_window else None,
        }, indent=2) + '\n')
        try:
            yield env
        finally:
            codex_module._codex_mcp_config_overrides = original_config
            bridge.stop()
            os.environ.clear()
            os.environ.update(previous)
            sys.path.remove(str(TRANSPORT_RUNTIME))
