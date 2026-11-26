"""Compatibility import for the LIBERO adapter owned by SubgoalVLA."""
import sys
from stage_bridge import libero as _implementation
sys.modules[__name__] = _implementation
