"""CPU integration checks: real Toolkit dispatch, fake physical environment."""
import json
from pathlib import Path
import tempfile
import unittest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import Toolkit
import research.review_window as review


class Environment:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)

    def chunk_step(self, actions, **kwargs):
        self.actions.extend(actions)


class Primitives:
    def __init__(self):
        self.env = Environment()
        self.forwards = 0
        self.history = []

    @tool
    def move_to(self, steps: int = 100) -> ToolResult:
        """Fake analytic servo including calibration and motion steps."""
        for index in range(steps):
            self.env.step(index)
        return ToolResult(data={"reached": True})

    @tool
    def vla_skill(self, prompt: str, max_chunks: int = 40) -> ToolResult:
        """Fake native eight-action policy with persistent history."""
        for _ in range(review.remaining_chunks(max_chunks, 8)):
            self.forwards += 1
            self.history.append(prompt)
            for action in range(8):
                self.env.step(action)
        return ToolResult(data={"prompt": prompt})

    @tool
    def pi0_doubled(self, prompt: str, max_chunks: int = 20) -> ToolResult:
        """Fake native eight-action chunk RPC."""
        for _ in range(review.remaining_chunks(max_chunks, 8)):
            self.forwards += 1
            self.env.chunk_step([0] * 8)
        return ToolResult(data={"prompt": prompt})

    @tool
    def finish(self) -> ToolResult:
        """Fake agent completion claim."""
        return ToolResult(data={"finished": True})


class State:
    def latest_record(self):
        return None


class TestToolkit(Toolkit):
    def _register_common_tools(self):
        pass

    def __init__(self):
        super().__init__(dashboard_events=NullDashboardEventSink(), memory=None, state=State())
        self._primitives = Primitives()
        for name in ("move_to", "vla_skill", "pi0_doubled", "finish"):
            self.add_tool(getattr(self._primitives, name))

    def get_env_state(self, *, command, result, elapsed_s):
        return ToolResult(data=dict(result), images=[b"fresh-test-image"])


class ReviewTests(unittest.TestCase):
    def setUp(self):
        review._window = None
        self.directory = tempfile.TemporaryDirectory()
        self.toolkit = review.install(TestToolkit(), 40, Path(self.directory.name))
        self.env = self.toolkit._primitives.env

    def tearDown(self):
        self.toolkit.close()
        review._window = None
        self.directory.cleanup()

    def deliver(self, result):
        payload = {"input": [{"type": "function_call_output",
                              "output": json.dumps(result.data)},
                             {"type": "input_image", "image_url": "test-image"}]}
        review.on_planner_request(payload)
        return payload

    def test_long_analytic_and_batched_calls_require_actual_review(self):
        result = self.toolkit.execute_tool("move_to", {"steps": 100})
        self.assertEqual(len(self.env.actions), 40)
        self.assertTrue(result.data["review_required"])
        self.assertFalse(result.is_error)
        blocked = self.toolkit.execute_tool("move_to", {"steps": 100})
        self.assertEqual(len(self.env.actions), 40)
        self.assertEqual(blocked.data["review_window"]["review_token"],
                         result.data["review_window"]["review_token"])
        review.on_planner_request({"input": []})
        self.assertEqual(review._window.remaining, 0)
        self.deliver(result)
        self.toolkit.execute_tool("move_to", {"steps": 100})
        self.assertEqual(len(self.env.actions), 80)

    def test_short_calls_accumulate_and_replayed_input_does_not_refill(self):
        first = self.toolkit.execute_tool("move_to", {"steps": 9})
        self.toolkit.execute_tool("move_to", {"steps": 31})
        self.assertEqual(len(self.env.actions), 40)
        # Old observation token cannot unlock a newer physical state.
        self.deliver(first)
        self.assertEqual(review._window.remaining, 0)
        latest = self.toolkit.execute_tool("move_to", {"steps": 1})
        payload = self.deliver(latest)
        self.toolkit.execute_tool("move_to", {"steps": 40})
        review.on_planner_request(payload)
        self.assertEqual(review._window.remaining, 0)
        self.assertEqual(len(self.env.actions), 80)

    def test_balance_is_absent_but_short_call_observation_refills_window(self):
        first = self.toolkit.execute_tool("move_to", {"steps": 11})
        self.assertNotIn("remaining_actions", first.data["review_window"])
        rows = [json.loads(line) for line in
                (Path(self.directory.name) / "review_windows.jsonl").read_text().splitlines()]
        self.assertTrue(all("remaining_actions" not in row for row in rows))
        self.assertEqual(rows[-1]["used"], 11)
        self.assertEqual(review._window.remaining, 29)
        self.deliver(first)
        self.assertEqual(review._window.remaining, 40)
        self.toolkit.execute_tool("vla_skill", {"prompt": "pick the cup"})
        self.assertEqual(len(self.env.actions), 51)

    def test_vla_native_chunks_and_history(self):
        first = self.toolkit.execute_tool("vla_skill", {"prompt": "pick the cup"})
        self.assertEqual(len(self.env.actions), 40)
        self.assertEqual(self.toolkit._primitives.forwards, 5)
        self.deliver(first)
        self.toolkit.execute_tool("vla_skill", {"prompt": "pick the cup", "max_chunks": 2})
        self.assertEqual(len(self.env.actions), 56)
        self.assertEqual(self.toolkit._primitives.history, ["pick the cup"] * 7)
        # Analytic + VLA share the remaining budget; no partial native chunks.
        self.toolkit.execute_tool("move_to", {"steps": 1})
        self.toolkit.execute_tool("vla_skill", {"prompt": "pick the cup"})
        self.assertEqual(len(self.env.actions), 73)
        self.assertEqual(review._window.remaining, 7)

    def test_libero_chunk_rpc_stops_at_five_forwards(self):
        result = self.toolkit.execute_tool("pi0_doubled", {"prompt": "open the drawer"})
        self.assertEqual(len(self.env.actions), 40)
        self.assertEqual(self.toolkit._primitives.forwards, 5)
        self.assertNotIn("success", result.data)
        self.deliver(result)
        self.toolkit.execute_tool("move_to", {"steps": 38})
        self.toolkit.execute_tool("pi0_doubled", {"prompt": "open the drawer"})
        self.assertEqual(len(self.env.actions), 78)
        self.assertEqual(self.toolkit._primitives.forwards, 5)

    def test_text_only_capture_and_active_motion_do_not_refill(self):
        review._window.run_actions(40, lambda: None)
        result = review._window.observe(ToolResult(data={}), "no-image")
        self.deliver(result)
        self.assertEqual(review._window.remaining, 0)
        result = review._window.observe(ToolResult(images=[b"image"]), "view")
        with review._window.motion():
            self.deliver(result)
        self.assertEqual(review._window.remaining, 0)
        self.deliver(result)
        self.assertEqual(review._window.remaining, 40)

    def test_finish_requires_review_of_latest_motion(self):
        result = self.toolkit.execute_tool("move_to", {"steps": 1})
        blocked = self.toolkit.execute_tool("finish", {})
        self.assertTrue(blocked.data["review_required"])
        self.assertNotIn("finished", blocked.data)
        self.deliver(result)
        self.assertTrue(self.toolkit.execute_tool("finish", {}).data["finished"])

    def test_subscription_request_boundary_drives_review(self):
        from research.planner_transport import RequestBoundary
        boundary = RequestBoundary(Path(self.directory.name) / "inputs", enforce=True)
        result = self.toolkit.execute_tool("move_to", {"steps": 100})
        payload = {"model": "gpt-6-astra", "reasoning": {"effort": "low"},
                   "tools": [{"type": "namespace", "name": "functions", "tools": [
                       {"type": "custom", "name": "exec", "description": ""}]}],
                   "input": [{"type": "function_call_output", "output": json.dumps(result.data)}]}
        boundary(payload)
        self.assertEqual(review._window.remaining, 0)
        payload["input"].append({"type": "input_image", "image_url": "fresh-test-image"})
        boundary(payload)
        self.assertEqual(review._window.remaining, 40)
        self.toolkit.execute_tool("move_to", {"steps": 100})
        boundary(payload)
        self.assertEqual(review._window.remaining, 0)


if __name__ == "__main__":
    unittest.main()
