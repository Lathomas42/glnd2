"""GPU-accelerated multi-channel LUT compositor.

Each channel is uploaded once as a 16-bit GPU texture. All LUT curve
adjustments (black point, white point, gamma) and channel tinting/blending
happen per-frame in a fragment shader, so dragging sliders stays instant
regardless of source image size.
"""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QOpenGLContext, QPainter, QPen, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

MAX_CHANNELS = 8
_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
_PER_CHANNEL_UNIFORMS = ("uTex", "uComp", "uEnabled", "uBlack", "uWhite", "uGamma", "uColor")
# (internal format, pixel format) for a texture packing 1..4 channels.
_PACKED_FORMATS = {
    1: (gl.GL_R16, gl.GL_RED),
    2: (gl.GL_RG16, gl.GL_RG),
    3: (gl.GL_RGB16, gl.GL_RGB),
    4: (gl.GL_RGBA16, gl.GL_RGBA),
}


@dataclass
class ChannelState:
    """Display settings for one channel. `name` is the raw ND2 channel name."""
    name: str
    color: tuple[float, float, float]
    black: float
    white: float
    gamma: float = 1.0
    enabled: bool = True
    # True while black/white still come from auto-contrast, i.e. the user
    # hasn't hand-tuned them; such channels follow the auto-headroom knob.
    auto: bool = False


def default_gl_format() -> QSurfaceFormat:
    # 4.1 core is the newest profile macOS supports.
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    return fmt


@dataclass
class GpuTextures:
    """An image's channels, already on the GPU as mipmapped textures in the GL
    namespace shared by every context in the app."""
    ids: list[int]  # every texture to free
    channels: list[tuple[int, int]]  # per channel: (texture id, component 0-3)
    step: int  # downsampling applied to fit GL_MAX_TEXTURE_SIZE (1 = none)


def _compile_program(vert_name: str, frag_name: str) -> int:
    def compile_shader(name, kind):
        with open(os.path.join(_SHADER_DIR, name)) as f:
            src = f.read()
        sid = gl.glCreateShader(kind)
        gl.glShaderSource(sid, src)
        gl.glCompileShader(sid)
        if not gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(sid).decode())
        return sid

    vert = compile_shader(vert_name, gl.GL_VERTEX_SHADER)
    frag = compile_shader(frag_name, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vert)
    gl.glAttachShader(program, frag)
    gl.glLinkProgram(program)
    gl.glDeleteShader(vert)
    gl.glDeleteShader(frag)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(program).decode())
    return program


def _new_texture(internal_format, pixel_format, data: np.ndarray) -> int:
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    # Mipmapped minification: zoomed out, each screen pixel averages the image
    # pixels under it instead of point-sampling one, so small bright cells
    # don't shimmer or vanish. (Most of a big image's upload time is the
    # driver re-laying out the texture for mipmaps; a hand-written mip shader
    # wasn't meaningfully faster.)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    h, w = data.shape[:2]
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, w, h, 0, pixel_format, gl.GL_UNSIGNED_SHORT, data)
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    return tex


def upload_textures(nd2_image) -> GpuTextures:
    """Put an image's channels on the GPU, in the current GL context.

    Interleaved images with up to four channels go up as a single texture
    straight from the file's buffer (no per-channel copies); anything else
    gets one single-channel texture per channel.
    """
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    max_tex = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
    longest = max(nd2_image.width, nd2_image.height)
    step = -(-longest // max_tex) if longest > max_tex else 1  # ceil division
    ids, channels = [], []
    try:
        packed = getattr(nd2_image, "packed", None)
        if step == 1 and packed is not None:
            tex = _new_texture(*_PACKED_FORMATS[packed.shape[2]], packed)
            ids.append(tex)
            channels = [(tex, k) for k in range(packed.shape[2])]
        else:
            for ch in nd2_image.channels[:MAX_CHANNELS]:
                tex = _new_texture(gl.GL_R16, gl.GL_RED, np.ascontiguousarray(ch.array[::step, ::step]))
                ids.append(tex)
                channels.append((tex, 0))
    except Exception:
        if ids:
            gl.glDeleteTextures(ids)
        raise
    finally:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return GpuTextures(ids, channels, step)


def upload_in_background(nd2_image, surface) -> bool:
    """For a worker thread: upload `nd2_image` through a temporary context that
    shares textures with the viewer, attach the result as `nd2_image.gpu`, and
    drop the CPU pixel copies. Returns False (image untouched) if that isn't
    possible, in which case the viewer uploads it itself when shown."""
    share = QOpenGLContext.globalShareContext()
    if surface is None or share is None:
        return False
    ctx = QOpenGLContext()
    ctx.setShareContext(share)
    ctx.setFormat(default_gl_format())
    if not ctx.create() or not ctx.makeCurrent(surface):
        return False
    try:
        nd2_image.gpu = upload_textures(nd2_image)
        gl.glFinish()  # complete before another context samples them
    except Exception:
        return False
    finally:
        ctx.doneCurrent()
    nd2_image.packed = None
    for ch in nd2_image.channels:
        ch.array = None
    return True


class CompositeGLWidget(QOpenGLWidget):
    # Emits (distance_px, distance_um_or_None) when a two-click measurement
    # completes, or None when the current measurement is cleared/reset.
    measured = Signal(object)
    # Emits (x, y, w, h) in image pixel coordinates when a rectangle ROI
    # drag completes with a non-trivial size.
    roiDrawn = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(default_gl_format())

        self._program = None
        self._uniforms: dict[str, int] = {}
        self._vao = None
        self._textures: list[int] = []
        self._channel_tex: list[tuple[int, int]] = []  # per channel: (texture, component)
        self._states: list[ChannelState] = []
        self._img_w = 0
        self._img_h = 0
        self._downsampled = False
        self._downsample_step = 1
        self._pixel_size_um: float | None = None

        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._drag_start = None
        self._pan_start = None

        self._measure_mode = False
        self._measure_points: list[tuple[float, float]] = []

        self._roi_mode = False
        self._roi_drag_start: tuple[float, float] | None = None
        self._roi_drag_current: tuple[float, float] | None = None
        self._rois: list = []  # objects with .name/.x/.y/.w/.h (roi_store.ROI)

        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self):
        self._program = _compile_program("composite.vert", "composite.frag")
        names = ["uScale", "uPan", "uCount"] + [
            f"{u}[{i}]" for u in _PER_CHANNEL_UNIFORMS for i in range(MAX_CHANNELS)
        ]
        self._uniforms = {n: gl.glGetUniformLocation(self._program, n) for n in names}
        self._build_quad()
        gl.glClearColor(0.06, 0.06, 0.08, 1.0)
        gl.glDisable(gl.GL_DEPTH_TEST)

    def _build_quad(self):
        # xy in [-1,1], uv with row 0 = image top
        verts = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                1.0, -1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0,
                -1.0, 1.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)

        stride = 4 * 4
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def set_image(self, nd2_image, states: list[ChannelState]):
        """Display `nd2_image` with `states` (one per channel, in channel
        order). Uses its pre-uploaded textures (`nd2_image.gpu`, taking
        ownership) if a loader thread made them, else uploads now. Deciding
        the states is the caller's job, so the screen and exports agree.
        """
        self.makeCurrent()
        if self._textures:
            gl.glDeleteTextures(self._textures)
        gpu, nd2_image.gpu = nd2_image.gpu, None
        if gpu is None:
            gpu = upload_textures(nd2_image)
        self._textures = gpu.ids
        self._channel_tex = gpu.channels
        self._states = list(states[:MAX_CHANNELS])
        step = gpu.step
        self._downsampled = step > 1
        self._img_w = -(-nd2_image.width // step)
        self._img_h = -(-nd2_image.height // step)
        self._downsample_step = step
        self._pixel_size_um = nd2_image.pixel_size_um
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._measure_points = []
        self._rois = []
        self._roi_drag_start = None
        self._roi_drag_current = None
        self.doneCurrent()
        self.measured.emit(None)
        self.update()

    def clear(self):
        """Drop the current image and free its GPU memory."""
        if self._textures:
            self.makeCurrent()
            gl.glDeleteTextures(self._textures)
            self.doneCurrent()
        self._textures = []
        self._channel_tex = []
        self._states = []
        self.update()

    def discard(self, nd2_image):
        """Free the pre-uploaded textures of an image that won't be shown."""
        if nd2_image.gpu is not None:
            self.makeCurrent()
            gl.glDeleteTextures(nd2_image.gpu.ids)
            self.doneCurrent()
            nd2_image.gpu = None

    def is_downsampled(self) -> bool:
        """True if the last loaded image exceeded GL_MAX_TEXTURE_SIZE and had
        to be downsampled for GPU display/export."""
        return self._downsampled

    def get_states(self) -> list[ChannelState]:
        return self._states

    def set_channel_param(self, index: int, **kwargs):
        st = self._states[index]
        for k, v in kwargs.items():
            setattr(st, k, v)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self.update()

    def set_measure_mode(self, on: bool):
        self._measure_mode = on
        self._measure_points = []
        if on:
            self._roi_mode = False
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.measured.emit(None)
        self.update()

    def set_roi_mode(self, on: bool):
        self._roi_mode = on
        self._roi_drag_start = None
        self._roi_drag_current = None
        if on:
            self._measure_mode = False
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def set_rois(self, rois: list):
        """Set the ROIs to draw as an overlay (typically all ROIs belonging
        to the currently loaded file)."""
        self._rois = rois
        self.update()

    def center_on_roi(self, x: float, y: float, w: float, h: float, fill: float = 0.7):
        """Zoom/pan so the given image-space rectangle is centered and fills
        about `fill` of the viewport."""
        if not self._img_w or not self._img_h:
            return
        base_sx, base_sy = self._compute_scale(self.width(), self.height())
        # base_sx/base_sy are the scale at zoom=1; back them out to get the
        # zoom-independent "fit whole image" scale.
        fit_sx = base_sx / self._zoom if self._zoom else base_sx
        fit_sy = base_sy / self._zoom if self._zoom else base_sy

        frac_w = max(w / self._img_w, 1e-6)
        frac_h = max(h / self._img_h, 1e-6)
        zoom_x = fill / (frac_w * fit_sx) if fit_sx else 1.0
        zoom_y = fill / (frac_h * fit_sy) if fit_sy else 1.0
        self._zoom = max(0.1, min(60.0, min(zoom_x, zoom_y)))

        sx, sy = self._compute_scale(self.width(), self.height())
        cx, cy = x + w / 2.0, y + h / 2.0
        u, v = cx / self._img_w, cy / self._img_h
        apos_x, apos_y = 2.0 * u - 1.0, 1.0 - 2.0 * v
        self._pan = [-apos_x * sx, -apos_y * sy]
        self.update()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _compute_scale(self, vp_w: int, vp_h: int) -> tuple[float, float]:
        if not self._img_w or not self._img_h or not vp_w or not vp_h:
            return 1.0, 1.0
        img_aspect = self._img_w / self._img_h
        vp_aspect = vp_w / vp_h
        if img_aspect > vp_aspect:
            sx = self._zoom
            sy = self._zoom * vp_aspect / img_aspect
        else:
            sy = self._zoom
            sx = self._zoom * img_aspect / vp_aspect
        return sx, sy

    def _screen_to_image(self, sx: float, sy: float) -> tuple[float, float]:
        """Map a widget-space point (e.g. a mouse click) to image pixel coordinates."""
        w, h = self.width(), self.height()
        scale_x, scale_y = self._compute_scale(w, h)
        ndc_x = -1.0 + 2.0 * sx / w if w else 0.0
        ndc_y = 1.0 - 2.0 * sy / h if h else 0.0
        apos_x = (ndc_x - self._pan[0]) / scale_x if scale_x else 0.0
        apos_y = (ndc_y - self._pan[1]) / scale_y if scale_y else 0.0
        u = (apos_x + 1.0) / 2.0
        v = (1.0 - apos_y) / 2.0
        return u * self._img_w, v * self._img_h

    def _image_to_screen(self, px: float, py: float) -> tuple[float, float]:
        """Inverse of `_screen_to_image`, for drawing overlays at the right spot."""
        w, h = self.width(), self.height()
        scale_x, scale_y = self._compute_scale(w, h)
        u = px / self._img_w if self._img_w else 0.0
        v = py / self._img_h if self._img_h else 0.0
        apos_x = 2.0 * u - 1.0
        apos_y = 1.0 - 2.0 * v
        ndc_x = apos_x * scale_x + self._pan[0]
        ndc_y = apos_y * scale_y + self._pan[1]
        sx = (ndc_x + 1.0) / 2.0 * w
        sy = (1.0 - ndc_y) / 2.0 * h
        return sx, sy

    def _measure_distance(self) -> tuple[float, float | None]:
        """Distance between the two current measure points, in pixels and
        (if the file had calibration metadata) micrometers."""
        (x1, y1), (x2, y2) = self._measure_points
        dist_px = math.hypot(x2 - x1, y2 - y1)
        dist_um = None
        if self._pixel_size_um:
            dist_um = dist_px * self._downsample_step * self._pixel_size_um
        return dist_px, dist_um

    def _draw(self, vp_w: int, vp_h: int):
        sx, sy = self._compute_scale(vp_w, vp_h)
        self._draw_with_transform(sx, sy, self._pan[0], self._pan[1])

    def _draw_with_transform(self, scale_x: float, scale_y: float, pan_x: float, pan_y: float):
        u = self._uniforms
        gl.glUseProgram(self._program)
        gl.glUniform2f(u["uScale"], scale_x, scale_y)
        gl.glUniform2f(u["uPan"], pan_x, pan_y)
        gl.glUniform1i(u["uCount"], len(self._states))

        for i, st in enumerate(self._states):
            tex, component = self._channel_tex[i]
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(u[f"uTex[{i}]"], i)
            gl.glUniform1i(u[f"uComp[{i}]"], component)
            gl.glUniform1i(u[f"uEnabled[{i}]"], 1 if st.enabled else 0)
            gl.glUniform1f(u[f"uBlack[{i}]"], st.black / 65535.0)
            gl.glUniform1f(u[f"uWhite[{i}]"], st.white / 65535.0)
            gl.glUniform1f(u[f"uGamma[{i}]"], st.gamma)
            gl.glUniform3f(u[f"uColor[{i}]"], *st.color)

        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if not self._states:
            return
        self._draw(self.width(), self.height())

    def _render_offscreen(self, out_w: int, out_h: int, draw_fn) -> np.ndarray:
        """Render into an off-screen FBO of the given size, calling `draw_fn()`
        to issue the actual draw call, and return an (H, W, 3) uint8 RGB array."""
        self.makeCurrent()
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        color_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, out_w, out_h, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_tex, 0)

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glDeleteFramebuffers(1, [fbo])
            gl.glDeleteTextures([color_tex])
            self.doneCurrent()
            raise RuntimeError(f"Offscreen framebuffer incomplete: {status}")

        gl.glViewport(0, 0, out_w, out_h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        draw_fn()
        gl.glFinish()

        pixels = gl.glReadPixels(0, 0, out_w, out_h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(out_h, out_w, 3)
        arr = np.flipud(arr).copy()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDeleteFramebuffers(1, [fbo])
        gl.glDeleteTextures([color_tex])
        gl.glViewport(0, 0, self.width(), self.height())
        self.doneCurrent()
        return arr

    def render_to_array(self, scale: float = 1.0) -> np.ndarray | None:
        """Render the full composite off-screen at `scale` * source resolution.

        `scale=1.0` renders at full native resolution; `scale=0.5` renders
        directly at half resolution on the GPU (cheaper than rendering full
        size and shrinking afterwards). Returns an (H, W, 3) uint8 RGB
        numpy array, or None if no image is loaded. Always shows the full
        image regardless of the interactive pan/zoom currently on screen.
        """
        if not self._states or not self._img_w or not self._img_h:
            return None
        out_w = max(1, round(self._img_w * scale))
        out_h = max(1, round(self._img_h * scale))
        saved_zoom, saved_pan = self._zoom, self._pan
        self._zoom, self._pan = 1.0, [0.0, 0.0]
        try:
            return self._render_offscreen(out_w, out_h, lambda: self._draw(out_w, out_h))
        finally:
            self._zoom, self._pan = saved_zoom, saved_pan

    def render_region_to_array(self, x: float, y: float, w: float, h: float,
                                scale: float = 1.0) -> np.ndarray | None:
        """Render just the given image-space rectangle (in current, possibly
        downsample-adjusted pixel coordinates), scaled by `scale`, filling
        the whole output. Used for exporting ROI crops."""
        if not self._states or not self._img_w or not self._img_h or w <= 0 or h <= 0:
            return None
        out_w = max(1, round(w * scale))
        out_h = max(1, round(h * scale))

        u0, u1 = x / self._img_w, (x + w) / self._img_w
        v0, v1 = y / self._img_h, (y + h) / self._img_h
        apos_x0, apos_x1 = 2.0 * u0 - 1.0, 2.0 * u1 - 1.0
        apos_y0, apos_y1 = 1.0 - 2.0 * v1, 1.0 - 2.0 * v0
        scale_x = 2.0 / (apos_x1 - apos_x0) if apos_x1 != apos_x0 else 1.0
        scale_y = 2.0 / (apos_y1 - apos_y0) if apos_y1 != apos_y0 else 1.0
        pan_x = -1.0 - apos_x0 * scale_x
        pan_y = -1.0 - apos_y0 * scale_y

        return self._render_offscreen(
            out_w, out_h,
            lambda: self._draw_with_transform(scale_x, scale_y, pan_x, pan_y),
        )

    # ------------------------------------------------------------------
    # Interaction: wheel to zoom, drag to pan, double-click to reset
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        factor = 1.0015 ** event.angleDelta().y()
        if not self._img_w or not self._img_h:
            self._zoom = max(0.1, min(60.0, self._zoom * factor))
            self.update()
            return

        sx, sy = event.position().x(), event.position().y()
        img_x, img_y = self._screen_to_image(sx, sy)  # point under the cursor, before zoom

        self._zoom = max(0.1, min(60.0, self._zoom * factor))

        # Re-pan so that same image point stays under the cursor after
        # zooming, instead of the view re-centering on the image's middle.
        w, h = self.width(), self.height()
        scale_x, scale_y = self._compute_scale(w, h)
        u, v = img_x / self._img_w, img_y / self._img_h
        apos_x, apos_y = 2.0 * u - 1.0, 1.0 - 2.0 * v
        ndc_x = -1.0 + 2.0 * sx / w if w else 0.0
        ndc_y = 1.0 - 2.0 * sy / h if h else 0.0
        self._pan[0] = ndc_x - apos_x * scale_x
        self._pan[1] = ndc_y - apos_y * scale_y

        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self._measure_mode:
            if event.button() == Qt.LeftButton:
                img_pt = self._screen_to_image(event.position().x(), event.position().y())
                if len(self._measure_points) >= 2:
                    self._measure_points = [img_pt]
                    self.measured.emit(None)
                else:
                    self._measure_points.append(img_pt)
                    if len(self._measure_points) == 2:
                        self.measured.emit(self._measure_distance())
                self.update()
            elif event.button() == Qt.RightButton:
                self._measure_points = []
                self.measured.emit(None)
                self.update()
            return

        if self._roi_mode:
            if event.button() == Qt.LeftButton:
                img_pt = self._screen_to_image(event.position().x(), event.position().y())
                self._roi_drag_start = img_pt
                self._roi_drag_current = img_pt
                self.update()
            elif event.button() == Qt.RightButton:
                self._roi_drag_start = None
                self._roi_drag_current = None
                self.update()
            return

        if event.button() == Qt.LeftButton:
            self._drag_start = event.position()
            self._pan_start = tuple(self._pan)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._roi_mode and self._roi_drag_start is not None:
            self._roi_drag_current = self._screen_to_image(event.position().x(), event.position().y())
            self.update()
            return
        if self._drag_start is not None:
            d = event.position() - self._drag_start
            self._pan[0] = self._pan_start[0] + 2.0 * d.x() / max(self.width(), 1)
            self._pan[1] = self._pan_start[1] - 2.0 * d.y() / max(self.height(), 1)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._roi_mode and event.button() == Qt.LeftButton and self._roi_drag_start is not None:
            (x1, y1), (x2, y2) = self._roi_drag_start, self._roi_drag_current
            # Snap to whole pixels inside the image, so crops are exact and
            # a drag that strays past the edge doesn't export black margin.
            def clamp(v, hi):
                return int(round(min(max(v, 0.0), hi)))
            left, right = clamp(min(x1, x2), self._img_w), clamp(max(x1, x2), self._img_w)
            top, bottom = clamp(min(y1, y2), self._img_h), clamp(max(y1, y2), self._img_h)
            self._roi_drag_start = None
            self._roi_drag_current = None
            self.update()
            if right - left >= 5 and bottom - top >= 5:  # ignore accidental clicks/tiny drags
                self.roiDrawn.emit(float(left), float(top), float(right - left), float(bottom - top))
            return
        self._drag_start = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if not self._measure_mode and not self._roi_mode:
            self.reset_view()

    # ------------------------------------------------------------------
    # Measurement / ROI overlay (drawn with QPainter on top of the GL content)
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._measure_points and not self._rois and not self._roi_drag_start:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self._paint_measure_overlay(painter)
        self._paint_roi_overlay(painter)

        painter.end()

    def _paint_measure_overlay(self, painter: QPainter):
        if not self._measure_points:
            return
        points = [self._image_to_screen(px, py) for px, py in self._measure_points]

        pen = QPen(QColor(255, 235, 59), 2)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 235, 59))
        for x, y in points:
            painter.drawEllipse(QPointF(x, y), 4, 4)

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            dist_px, dist_um = self._measure_distance()
            label = f"{dist_um:.2f} µm" if dist_um is not None else f"{dist_px:.1f} px"
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QPointF((x1 + x2) / 2 + 8, (y1 + y2) / 2 - 8), label)

    def _paint_roi_overlay(self, painter: QPainter):
        painter.setBrush(Qt.NoBrush)
        roi_pen = QPen(QColor(0, 230, 180), 2)
        for roi in self._rois:
            x0, y0 = self._image_to_screen(roi.x, roi.y)
            x1, y1 = self._image_to_screen(roi.x + roi.w, roi.y + roi.h)
            painter.setPen(roi_pen)
            painter.drawRect(x0, y0, x1 - x0, y1 - y0)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QPointF(min(x0, x1) + 4, min(y0, y1) + 16), roi.name)

        if self._roi_drag_start is not None and self._roi_drag_current is not None:
            (ix1, iy1), (ix2, iy2) = self._roi_drag_start, self._roi_drag_current
            x0, y0 = self._image_to_screen(min(ix1, ix2), min(iy1, iy2))
            x1, y1 = self._image_to_screen(max(ix1, ix2), max(iy1, iy2))
            pen = QPen(QColor(255, 150, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(x0, y0, x1 - x0, y1 - y0)
