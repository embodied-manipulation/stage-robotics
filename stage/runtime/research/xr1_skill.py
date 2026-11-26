"""XR1 sensor history, with resets owned by explicit Subgoal transitions."""
from __future__ import annotations

import numpy as np
from robots.robocasa.vla_skill import VLASkill
from research.transition_history import TransitionOwnedHistory


class XR1Skill(TransitionOwnedHistory, VLASkill):

    def _build_obs(self, task_text):
        observation = super()._build_obs(task_text)
        history = list(self._hist)
        indices = [max(0, min(len(history) - 1, len(history) + int(delta) - 1)) for delta in self._vdi]
        for key in observation:
            if key.startswith('state.'):
                observation[key] = np.stack([history[index][key] for index in indices])[None]
        return observation
