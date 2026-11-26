"""Compatibility import; maintained implementation: subgoal_core.protocol."""
import sys
from subgoal_core import protocol as _implementation
sys.modules[__name__] = _implementation
