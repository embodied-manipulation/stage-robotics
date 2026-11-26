"""Compatibility import; maintained implementation: stage_bridge.review."""
import sys
from stage_bridge import review as _implementation
sys.modules[__name__] = _implementation
