"""Bind subgoal_core's cadence to tool dispatch and environment RPCs."""
from pathlib import Path
from rpent.tools import ToolResult
from subgoal_core.cadence import ReviewRequired, contains_image, ReviewWindow as CoreReviewWindow
from subgoal_core.prompts import CADENCE_GUIDANCE as GUIDANCE


class ReviewWindow(CoreReviewWindow):
    def observe(self, result, tool_name):
        from subgoal_core.observations import _window
        self.image_window = _window
        return super().observe(result, tool_name)


_window = None


def on_planner_request(payload):
    if _window is not None:
        _window.planner_request(payload)


def remaining_chunks(requested, chunk_actions):
    if _window is None:
        return requested
    _window.require(chunk_actions)
    return min(requested, _window.remaining // chunk_actions)


def install(toolkit, actions, private):
    """Wrap this toolkit's own env client; never mutate another framework."""
    global _window
    if _window is not None:
        raise RuntimeError("One review window per Planner process")
    window = _window = ReviewWindow(actions, Path(private) / "review_windows.jsonl")
    primitives = toolkit._primitives
    env = primitives.env
    native_step = env.step

    def step(action):
        return window.run_actions(1, native_step, action)

    env.step = step
    if hasattr(env, "chunk_step"):
        native_chunk = env.chunk_step

        def chunk_step(chunk, **kwargs):
            return window.run_actions(len(chunk), native_chunk, chunk, **kwargs)

        env.chunk_step = chunk_step
    motion_names = {
        "vla_skill", "vla_arm", "pi0_pick", "pi0_doubled", "move_to", "move_delta",
        "move_pose", "rotate_wrist", "rotate_pitch", "set_gripper", "release",
        "scripted_grasp", "navigate_to", "move_base", "execute_action",
    }
    for name, declaration in list(toolkit._tools.items()):
        if name not in motion_names | {"finish"}:
            continue

        def bounded(_declaration=declaration, **inputs):
            with window.motion():
                try:
                    if _declaration.name == "finish":
                        if window.used:
                            raise ReviewRequired("Inspect the latest observation before finishing")
                    else:
                        window.require()
                    return _declaration(**inputs)
                except ReviewRequired:
                    return ToolResult(data={
                        "interrupted": True, "review_required": True,
                        "stop_reason": "planner_review_window",
                        "notice": "Observe, then decide whether to continue this local goal or switch executor. "
                                  "A partially executed analytic program is not automatically resumed; "
                                  "recompute targets from the current state. This return is not task success.",
                    })

        toolkit._tools[name] = declaration.with_handler(bounded)
    original = toolkit.execute_tool

    def execute(name, inputs):
        return window.observe(original(name, inputs), name)

    toolkit.execute_tool = execute
    return toolkit
