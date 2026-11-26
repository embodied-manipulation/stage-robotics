"""Compatibility import; maintained implementation: subgoal_core.observations."""
import sys
from subgoal_core import observations as _implementation
sys.modules[__name__] = _implementation
