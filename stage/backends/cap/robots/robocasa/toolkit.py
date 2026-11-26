# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboCasa toolkit: common tools + RoboCasa primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
RoboCasa primitives (``move_to``, ``vla_skill``, ``release``, ...) on top.
"""

from __future__ import annotations

import os

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robocasa import tools as robocasa_tools
from robots.robocasa.memory import public_output_dir
from robots.robocasa.restrictions import get_restrictions
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools, tool
from rpent.tools.direct_action import direct_action_tool
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, get_output_dir

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager

logger = get_logger("robocasa_toolkit")


class RoboCasaToolkit(Toolkit):
    """Toolkit for the RoboCasa robot."""

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        enable_direct_action: bool = False,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
    ) -> None:
        """Create a RoboCasa toolkit, wiring the primitives and tools."""
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported RoboCasa toolkit mode: {mode!r}")
        self._restrictions = get_restrictions()
        if self._restrictions.disable_bootstrap and mode == "exploration":
            raise ValueError("exploration is unavailable when bootstrap is disabled")
        state_type = (
            robocasa_tools.PublicEnvState
            if self._restrictions.hide_task_feedback
            else EnvState
        )
        artifact_dir = Path(state_output_dir or get_output_dir())
        if self._restrictions.restrict_file_access:
            artifact_dir = public_output_dir(get_output_dir())
        self._public_dir = artifact_dir
        state = state_type(artifact_dir)
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self._mode = mode
        self._attempt = 1
        self._attempts_per_session = max(0, int(attempts_per_session))
        self.init_primitives(
            runtime_kwargs={
                **runtime_kwargs,
                "allow_reset": mode == "exploration",
            }
        )
        self._register_robocasa_tools()
        if enable_direct_action:
            self.add_tool(
                direct_action_tool(
                    self._primitives.env.action_specs, self._primitives.execute_action
                )
            )

    def _register_robocasa_tools(self) -> None:
        for declaration in iter_tools(self._primitives):
            if declaration.name == "reset" and self._restrictions.disable_bootstrap:
                continue
            self.add_tool(declaration)
        for definition in iter_tools(robocasa_tools):
            handler = (
                definition
                if definition.name == "finish"
                else partial(definition, state=self._state)
            )
            self.add_tool(
                definition.with_handler(handler), replace=definition.name == "finish"
            )

        if self._restrictions.hide_task_feedback:
            replacements = {
                "finish": (
                    "robocasa_terminated becomes True (success detected)",
                    "you judge the task complete from current observations",
                ),
                "view_env_state": (
                    "robocasa_terminated flag, task_progress, ", "",
                ),
            }
            for name, (old, new) in replacements.items():
                if name in self._tools:
                    description = self._tools[name].description.replace(old, new)
                    self.add_tool(
                        replace(self._tools[name], description=description), replace=True
                    )

        from robots.robocasa.action_interface import vla_instruction_scope, vla_prompt_mode

        if vla_prompt_mode() == "provided":
            subgoal_only = vla_instruction_scope() == "subgoal_only"
            for name in ("vla_skill", "vla_arm"):
                declaration = self._tools[name]
                schema = declaration.input_schema
                schema["properties"]["prompt"]["description"] = (
                    "Only the next concrete local subgoal, passed verbatim to VLA. Full task descriptions are forbidden."
                    if subgoal_only else
                    "Language instruction passed verbatim to VLA: current subgoal or full task."
                )
                motion = "full-body" if name == "vla_skill" else "base-clamped"
                self.add_tool(replace(
                    declaration, _input_schema=schema,
                    description=(f"Execute the learned {motion} policy on the provided prompt verbatim. "
                                 + ("Supply only the next local subgoal, never the complete task. " if subgoal_only else "") +
                                 "The framework schedules observation checkpoints. Keep a suitable unfinished "
                                 "Subgoal active across calls; change it only on completion or recovery evidence."),
                ), replace=True)

        if self._mode == "exploration":
            self.add_tool(
                self._primitives.reset.with_handler(self._reset_episode), replace=True
            )
            self.add_tool(
                robocasa_tools.finish.with_handler(
                    partial(self._guarded_finish, robocasa_tools.finish)
                ),
                replace=True,
            )

    def _register_common_tools(self) -> None:
        self._restrictions = getattr(self, "_restrictions", get_restrictions())
        super()._register_common_tools()
        if not (
            self._restrictions.restrict_file_access
            or self._restrictions.disable_task_memory
            or self._restrictions.disable_global_memory
        ):
            return
        descriptions = {
            "read_text_file": "Read a UTF-8 file available to this episode. Paths are absolute or relative to the public output directory.",
            "write_text_file": "Write UTF-8 current-episode notes in the public output directory.",
            "list_dir": "List files available to this episode. An empty path selects the public output directory.",
        }
        for name, description in descriptions.items():
            declaration = self._tools[name]
            if name == "list_dir" and self._restrictions.restrict_file_access:
                inner = declaration

                @tool(name="list_dir", readonly=declaration.readonly)
                def public_list_dir(path: str = "") -> ToolResult:
                    """List current-episode public files.

                    Args:
                        path: Public directory path; empty selects this episode's public output directory.
                    """
                    return inner(path=path or str(self._public_dir))

                declaration = public_list_dir
            self.add_tool(replace(declaration, description=description), replace=True)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Apply the public boundary after native dispatch, including errors."""
        result = super().execute_tool(name, input_dict)
        if not self._restrictions.enabled:
            return result
        data = robocasa_tools.public_result(result.data)
        if self._restrictions.restrict_file_access and "traceback" in result.data:
            data.pop("traceback", None)
            data["error"] = "tool execution failed; details are in the private run log"
        return ToolResult(data=data, images=result.images)

    def _guarded_finish(self, inner: Any, **kwargs: Any) -> ToolResult:
        budget = self._attempts_per_session
        if budget and not self.solved() and self._attempt < budget:
            payload = {
                "error": "finish refused",
                "reason": (
                    f"This session has {budget - self._attempt} of its {budget} "
                    "attempts left. Archive the attempt, reset, and change the plan."
                ),
            }
            return ToolResult(data=payload)
        return inner(**kwargs)

    def _reset_episode(self) -> ToolResult:
        if self.solved():
            payload = {
                "error": "reset refused",
                "reason": "The task is already solved; save artifacts and finish.",
            }
            return ToolResult(data=payload)
        budget = self._attempts_per_session
        if budget and self._attempt >= budget:
            payload = {
                "error": "reset refused",
                "reason": (
                    f"This session's attempt budget is spent ({budget} attempts). "
                    "Update the handoff notes and finish this session."
                ),
            }
            return ToolResult(data=payload)
        result = self._primitives.reset().to_dict()
        if result.get("error"):
            payload = result
            return ToolResult(data=payload)
        self._attempt += 1
        payload = {
            **result,
            "attempt": self._attempt,
            "notice": "Fresh episode started. Re-run perception before acting.",
        }
        return ToolResult(data=payload)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                logger.warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = robocasa_tools.view_env_state(record.step_idx, state=self._state)
        out.data["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.data.update(result)
        return out

    def init_primitives(
        self,
        *,
        runtime_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the primitives, dump step 0."""
        self._state.reset()

        from robots.robocasa.primitives import RoboCasaPrimitives

        primitives = RoboCasaPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **runtime_kwargs,
        )
        # RoboCasaPrimitives initializes the environment in its constructor.
        # Keep the legacy evaluation call (where reset is disabled), but avoid
        # sampling and immediately discarding a second episode in exploration.
        if self._mode == "evaluation" and not self._restrictions.disable_bootstrap:
            primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = robocasa_tools.dump_state(primitives, self._state, log=None)
        if not self._restrictions.hide_task_feedback:
            try:
                self._state.save(
                    "success_criteria.md",
                    primitives.dump_success_criteria(),
                    step=None,
                )
            except Exception as e:
                logger.warning("failed to save success_criteria.md: %s", e)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """Flush the agent-side video buffer through ``EnvState``."""
        super().close()
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            logger.warning("failed to save episode video: %s", e)

    def solved(self) -> bool:
        """Return the success value from the final recorded environment state."""
        record = self._state.latest_record()
        return bool(record is not None and record.extras.get("success", False))

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the RoboCasa recipe JSONL from the dumped state trace."""
        return robocasa_tools.write_recipe_from_states(
            self._state,
            recipe_tag,
            output_dir=get_output_dir() if self._mode == "exploration" else None,
        )
