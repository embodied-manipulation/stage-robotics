"""Compatibility import; maintained adapter: stage_bridge.cadence."""
import sys
from stage_bridge import cadence as _implementation
sys.modules[__name__] = _implementation
