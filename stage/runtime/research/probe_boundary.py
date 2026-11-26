"""Astra-low MCP reachability and capability probe; no simulator or GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.planner_transport import planner_transport, set_allowed_tools


def run(output: Path, *, capture_only=False, inventory_only=False):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.planner.codex import CodexPlanner
    from rpent.tools import ToolResult, tool

    @tool(readonly=True)
    def observe_probe() -> ToolResult:
        """Read the current diagnostic observation."""
        return ToolResult(data={"probe_marker": "observable-42"})

    @tool(readonly=True)
    def finish(status: str, summary: str) -> ToolResult:
        """End the diagnostic and report the observed marker."""
        return ToolResult(data={"_finish": True, "status": status, "summary": summary})

    class ProbeToolkit:
        def __init__(self):
            self.calls = []
            self.tools = {t.name: t for t in (observe_probe, finish)}

        def list_tools(self):
            return tuple(self.tools.values())

        def execute_tool(self, name, args):
            self.calls.append({"name": name, "args": args})
            return self.tools[name](**args)

        def cancel_active_and_wait(self):
            pass

        def resume_calls(self):
            pass

    public, private = output / "public", output / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    from rpent.utils.logging import init_output_dir
    init_output_dir(public)
    os.environ["STAGE_RESTRICTED"] = "1"
    toolkit = ProbeToolkit()
    set_allowed_tools(toolkit.list_tools())
    with planner_transport(private, public, enforce_boundary=not capture_only):
        planner = CodexPlanner(
            output_dir=str(public), repo_root=public,
            dashboard_events=NullDashboardEventSink(),
            output_path=private / "planner.out", timeout_s=180,
            model="gpt-6-astra", reasoning_effort="low",
        )
        if inventory_only:
            import openai_codex
            from rpent.planner.utils.http_mcp_server import HttpMcpServer
            server = HttpMcpServer(toolkit)
            try:
                url = server.start()
                config = planner._build_config(url)
                (private / "sdk_config_overrides.json").write_text(json.dumps(config.config_overrides, indent=2))
                with openai_codex.Codex(config=config) as codex:
                    thread = codex.thread_start(**planner._thread_options)
                    status = codex._client._request_raw("mcpServerStatus/list", {"threadId": thread.id})
                    (private / "mcp_status.json").write_text(json.dumps(status, indent=2))
                    print(json.dumps(status))
            finally:
                server.stop()
            return 0
        result = planner.solve(
            system_prompt="Complete this diagnostic using actual MCP tool calls. Discover the available tools from ALL_TOOLS if needed.",
            user_message="Invoke the observe_probe tool. Then invoke the finish MCP tool with status done and the exact observed marker in summary. You must actually invoke finish; printing a JSON object does not invoke a tool.",
            toolkit=toolkit, max_turns=5,
        )
    report = {"calls": toolkit.calls, "finish": result.finish_result, "stats": result.stats, "error": result.error}
    report["passed"] = bool(
        any(c["name"] == "observe_probe" for c in toolkit.calls)
        and result.finish_result and "observable-42" in str(result.finish_result)
        and not result.error
    )
    (private / "probe_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "calls": len(toolkit.calls), "error": result.error, "output": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.output, capture_only=args.capture_only, inventory_only=args.inventory_only))
