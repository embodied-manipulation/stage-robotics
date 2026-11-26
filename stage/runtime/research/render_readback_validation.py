"""Reject failed native OpenGL frames before they reach any camera consumer."""

from functools import wraps

VERSION = 'checked-opengl-readback/v2'


def _errors(get_error):
    values = []
    for _ in range(16):
        value = int(get_error())
        if not value:
            return values
        values.append(value)
    raise RuntimeError(f'OpenGL error queue did not clear: {values}')


def checked_method(original, stage, get_error, begin_debug=None):
    @wraps(original)
    def checked(self, *args, **kwargs):
        self.gl_ctx.make_current()
        previous = _errors(get_error)
        if previous:
            raise RuntimeError(f'OpenGL errors before {stage}: {previous}')
        messages = begin_debug(self) if begin_debug is not None else []
        result = original(self, *args, **kwargs)
        errors = _errors(get_error)
        if errors:
            # Native RoboCasa assets can emit this material warning while the
            # scene renders correctly. Do not change the material or pixels.
            # Every error must have the exact driver diagnosis; an unclassified
            # GL_INVALID_VALUE or any readback error is still rejected.
            material_only = (stage == 'render' and set(errors) == {1281} and messages
                             and all(code == 1281 and 'Shininess value is out of range' in message
                                     for code, message in messages))
            if not material_only:
                raise RuntimeError(f'OpenGL {stage} failed: {errors}; frame rejected')
            if not getattr(self, '_agentic_material_warning_logged', False):
                print('OpenGL native material shininess warning; retaining native rendering and checking RGB readback', flush=True)
                self._agentic_material_warning_logged = True
        return result
    checked._agentic_checked_readback = True
    return checked


def checked_native_readback(original):
    """Detect the observed silent no-write failure without modifying valid pixels."""
    @wraps(original)
    def readback(*args, **kwargs):
        import numpy as np
        rgb = kwargs.get('rgb', args[0] if args else None)
        if rgb is not None:
            # An alternating pattern cannot be confused with a normal black or
            # uniform camera image. The native call must overwrite this buffer.
            rgb[...] = (165, 90, 195)
            rgb[::2, ::2] = (90, 195, 165)
        result = original(*args, **kwargs)
        if rgb is not None and np.all(rgb[::2, ::2] == (90, 195, 165)):
            if (np.all(rgb[1::2] == (165, 90, 195))
                    and np.all(rgb[::2, 1::2] == (165, 90, 195))):
                raise RuntimeError('Native OpenGL readback did not write the RGB buffer; frame rejected')
        return result
    readback._agentic_native_readback = True
    return readback


def install():
    import ctypes
    import mujoco
    from OpenGL import GL
    from robosuite.utils import binding_utils as binding

    if not getattr(mujoco.mjr_readPixels, '_agentic_native_readback', False):
        mujoco.mjr_readPixels = checked_native_readback(mujoco.mjr_readPixels)

    cls = binding.MjRenderContext
    def begin_debug(context):
        if not hasattr(context, '_agentic_gl_debug_callback'):
            context._agentic_gl_errors = []
            @GL.GLDEBUGPROC
            def callback(source, type_, code, severity, length, message, user):
                if type_ == GL.GL_DEBUG_TYPE_ERROR:
                    entry = (int(code), ctypes.string_at(message, length).decode(errors='replace'))
                    if entry not in context._agentic_gl_errors:
                        context._agentic_gl_errors.append(entry)
            context._agentic_gl_debug_callback = callback
            GL.glEnable(GL.GL_DEBUG_OUTPUT)
            GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS)
            GL.glDebugMessageCallback(callback, None)
        context._agentic_gl_errors.clear()
        return context._agentic_gl_errors

    for name, stage in (('render', 'render'), ('read_pixels', 'readback')):
        original = getattr(cls, name)
        if not getattr(original, '_agentic_checked_readback', False):
            setattr(cls, name, checked_method(original, stage, GL.glGetError, begin_debug))
