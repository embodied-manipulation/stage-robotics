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

"""RoboCasa env client — thin RPC layer over the env server."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from robots.robocasa.restrictions import get_restrictions
from rpent.robots.components.env_client_base import BaseEnvClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient

CAM_ALIAS = {
    "agentview": "robot0_agentview_left",
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "wrist": "robot0_eye_in_hand",
    "eye_in_hand": "robot0_eye_in_hand",
    "robot0_eye_in_hand": "robot0_eye_in_hand",
    "robot0_agentview_left": "robot0_agentview_left",
    "robot0_agentview_right": "robot0_agentview_right",
    "navview": "mobilebase0_navview",  # base-mounted forward-down floor/nav camera
    "mobilebase0_navview": "mobilebase0_navview",
}


class RoboCasaEnvClient(BaseEnvClient):
    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.grasp_contact": 10.0,
    }

    def __init__(self, client: RpcClient, *, expected_meta: dict, attach_only=None):
        self.restrictions = get_restrictions()
        requested_attach = os.environ.get("STAGE_ATTACH_ONLY") == "1"
        self.attach_only = (
            self.restrictions.disable_bootstrap or requested_attach
            if attach_only is None
            else bool(attach_only)
        )
        if self.restrictions.disable_bootstrap and not self.attach_only:
            raise ValueError("disable_bootstrap requires attach_only")
        self.action_budget_exhausted = False
        super().__init__(
            client, expected_meta=expected_meta, reset_on_connect=not self.attach_only
        )
        if self.attach_only:
            self.last_obs = self._client.call(
                "env.get_observation", timeout_s=self._TIMEOUT_S["default"]
            )
        self.camera_h = expected_meta["camera_h"]
        self.camera_w = expected_meta["camera_w"]

    def _resolve_cam(self, name):
        return CAM_ALIAS.get(name, name)

    # ---- state accessors ----
    def reset(self):
        if self.restrictions.disable_bootstrap:
            raise PermissionError("environment reset disabled")
        self.last_obs = self._client.call(
            "env.reset", timeout_s=self._TIMEOUT_S["env.reset"]
        )
        self.action_budget_exhausted = False
        return self.last_obs

    def step(self, flat_action):
        if self.action_budget_exhausted:
            raise ActionBudgetExhausted("physical action budget exhausted")
        result = self._client.call(
            "env.step", args=(flat_action,), timeout_s=self._TIMEOUT_S["env.step"]
        )
        self.last_obs = result[0]
        self.action_budget_exhausted = bool(
            isinstance(result[-1], dict)
            and result[-1].get("stage_action_budget_exhausted", False)
        )
        return result

    def set_action_source(self, source: str, tool: str):
        """Label actions only when using the instrumented research server."""
        if os.environ.get("STAGE_ACTION_PROVENANCE") == "1":
            self._client.call(
                "env.set_action_source",
                args=(source, tool),
                timeout_s=self._TIMEOUT_S["default"],
            )

    def check_success(self):
        if self.restrictions.hide_task_feedback:
            raise PermissionError("task feedback disabled")
        return self._success_for_stop()

    def success_for_stop(self):
        """Internal stop oracle, independently controlled from displayed feedback."""
        if self.restrictions.disable_success_stop:
            raise PermissionError("success stopping disabled")
        route = (
            "env.success_for_stop"
            if os.environ.get("STAGE_ACTION_PROVENANCE") == "1"
            else "env.check_success"
        )
        return self._success_for_stop(route)

    def _success_for_stop(self, route="env.check_success"):
        return self._client.call(
            route, timeout_s=self._TIMEOUT_S["default"]
        )

    @property
    def terminated(self):
        return self.check_success()

    @property
    def current_raw_obs(self):
        return self.last_obs

    @property
    def eef_pos(self):
        return np.array(self.last_obs["robot0_eef_pos"], dtype=np.float64)

    @property
    def eef_quat(self):
        return np.array(self.last_obs["robot0_eef_quat"], dtype=np.float64)

    @property
    def gripper_qpos(self):
        return np.array(self.last_obs.get("robot0_gripper_qpos", []), dtype=np.float64)

    # ---- task info ----
    def get_success_criteria_text(self):
        if self.restrictions.hide_task_feedback:
            raise PermissionError("task feedback disabled")
        return self._client.call(
            "env.get_success_criteria_text", timeout_s=self._TIMEOUT_S["default"]
        )

    def get_task_progress(self):
        if self.restrictions.hide_task_feedback:
            raise PermissionError("task feedback disabled")
        return self._client.call(
            "env.get_task_progress", timeout_s=self._TIMEOUT_S["default"]
        )

    # ---- rendering / perception ----
    def render_camera(self, camera_name, height=None, width=None, depth=False):
        """Agent-facing: vertically flip to top-down so the rgb reads naturally and
        is pixel-aligned with world_map()."""
        cam = self._resolve_cam(camera_name)
        h = height or self.camera_h
        w = width or self.camera_w
        if depth:
            rgb, real = super().render_camera(cam, height=h, width=w, depth=True)
            return rgb[::-1], real[::-1]
        return super().render_camera(cam, height=h, width=w, depth=False)[::-1]

    def get_camera_meta(self, camera_name, height=None, width=None):
        cam = self._resolve_cam(camera_name)
        h = height or self.camera_h
        w = width or self.camera_w
        return super().get_camera_meta(cam, height=h, width=w)

    def world_map(self, camera_name, height=None, width=None):
        """HxWx3 world xyz per pixel, TOP-DOWN (row 0 = image top), pixel-aligned
        with render_camera()'s rgb. Uses robosuite's camera transform matrix; the
        sim's OpenGL depth is bottom-up, so it is vertically flipped to match the
        transform's top-down convention (VERIFIED: GT object back-projects within
        ~2cm). Dense vectorized: world = T_p2w @ [col*z, row*z, z, 1]."""
        cam = self._resolve_cam(camera_name)
        h = height or self.camera_h
        w = width or self.camera_w
        _, z_native = super().render_camera(
            cam, height=h, width=w, depth=True
        )  # metric depth, bottom-up
        z = z_native[::-1]  # -> top-down
        T_p2w = self._client.call(
            "env.get_camera_transform",
            kwargs={"camera_name": cam, "height": h, "width": w},
            timeout_s=self._TIMEOUT_S["default"],
        )
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        homog = np.stack([cols * z, rows * z, z, np.ones_like(z)], axis=-1)  # HxWx4
        world = homog @ T_p2w.T  # HxWx4
        return world[..., :3]  # top-down, aligned with rgb

    def world_xyz_at(self, camera_name, row, col, height=None, width=None):
        return self.world_map(camera_name, height, width)[row, col]

    # ---- other ----
    def grasp_contact(self):
        if self.restrictions.disable_privileged_grasp:
            raise PermissionError("privileged grasp feedback disabled")
        contact, object_name = self._client.call(
            "env.grasp_contact", timeout_s=self._TIMEOUT_S["env.grasp_contact"]
        )
        return contact, None if self.restrictions.task_language_names_only else object_name

    def reassemble_env_action(self, unmap_result):
        return self._client.call(
            "env.reassemble_env_action",
            args=(unmap_result,),
            timeout_s=self._TIMEOUT_S["default"],
        )


class ActionBudgetExhausted(RuntimeError):
    """The driver-owned physical action budget has been consumed."""
