"""RLDX native current-state encoder with explicit goal-owned sensor history."""
from robots.robocasa.vla_skill import VLASkill
from research.transition_history import TransitionOwnedHistory


class VLATransitionSkill(TransitionOwnedHistory, VLASkill):
    pass
