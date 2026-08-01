"""GPU-accelerated multi-channel LUT compositor.

Each channel is uploaded once as a 16-bit GPU texture. All LUT curve
adjustments (black point, white point, gamma) and channel tinting/blending
happen per-frame in a fragment shader, so dragging sliders stays instant
regardless of source image size.
"""
from __future__ import annotations

import ctypes
import os

import math

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

MAX_CHANNELS = 8
_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


class ChannelState:
    def __init__(self, name, data_max, color, black, white, gamma=1.0, enabled=True):
        self.name = name
        self.data_max = data_max
        self.color = color
        self.black = black
        self.white = white
        self.gamma = gamma
        self.enabled = enabled


def default_gl_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    return fmt


class CompositeGLWidget(QOpenGLWidget):
    # Emits (distance_px, distance_um_or_None) when a two-click measurement
    # completes, or None when the current measurement is cleared/reset.
    measured = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(default_gl_format())

        self._program = None
        self._vao = None
        self._textures: list[int] = []
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

        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self):
        self._program = self._build_program()
        self._build_quad()
        gl.glClearColor(0.06, 0.06, 0.08, 1.0)
        gl.glDisable(gl.GL_DEPTH_TEST)

    def _build_program(self):
        with open(os.path.join(_SHADER_DIR, "composite.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(_SHADER_DIR, "composite.frag")) as f:
            frag_src = f.read()

        def compile_shader(src, kind):
            sid = gl.glCreateShader(kind)
            gl.glShaderSource(sid, src)
            gl.glCompileShader(sid)
            if not gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(sid).decode())
            return sid

        vert = compile_shader(vert_src, gl.GL_VERTEX_SHADER)
        frag = compile_shader(frag_src, gl.GL_FRAGMENT_SHADER)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vert)
        gl.glAttachShader(program, frag)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(program).decode())

        gl.glDeleteShader(vert)
        gl.glDeleteShader(frag)
        return program

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
    def set_image(self, nd2_image, carried_states: dict[str, ChannelState] | None = None):
        """Upload a new image's channels as GPU textures.

        `carried_states` maps channel name -> previous ChannelState, used to
        keep LUT settings consistent across files in a batch when channel
        names match (e.g. same wavelength setup across an acquisition run).
        """
        self.makeCurrent()

        if self._textures:
            gl.glDeleteTextures(self._textures)
        self._textures = []
        self._states = []

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        carried_states = carried_states or {}

        max_tex = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
        longest = max(nd2_image.width, nd2_image.height)
        self._downsampled = longest > max_tex
        step = 1
        if self._downsampled:
            step = -(-longest // max_tex)  # ceil division

        for ch in nd2_image.channels[:MAX_CHANNELS]:
            plane = ch.array[::step, ::step] if step > 1 else ch.array
            tex = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            h, w = plane.shape
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_R16, w, h, 0,
                gl.GL_RED, gl.GL_UNSIGNED_SHORT, np.ascontiguousarray(plane),
            )
            self._textures.append(tex)

            prev = carried_states.get(ch.name)
            if prev is not None:
                state = ChannelState(
                    name=ch.name, data_max=max(ch.data_max, 1), color=prev.color,
                    black=prev.black, white=prev.white, gamma=prev.gamma, enabled=prev.enabled,
                )
            else:
                state = ChannelState(
                    name=ch.name, data_max=max(ch.data_max, 1), color=ch.default_color,
                    black=ch.p_low, white=ch.p_high, gamma=1.0, enabled=True,
                )
            self._states.append(state)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._img_w = -(-nd2_image.width // step)
        self._img_h = -(-nd2_image.height // step)
        self._downsample_step = step
        self._pixel_size_um = nd2_image.pixel_size_um
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._measure_points = []
        self.doneCurrent()
        self.measured.emit(None)
        self.update()

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
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.measured.emit(None)
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
        gl.glUseProgram(self._program)
        sx, sy = self._compute_scale(vp_w, vp_h)
        gl.glUniform2f(gl.glGetUniformLocation(self._program, "uScale"), sx, sy)
        gl.glUniform2f(gl.glGetUniformLocation(self._program, "uPan"), self._pan[0], self._pan[1])
        gl.glUniform1i(gl.glGetUniformLocation(self._program, "uCount"), len(self._states))

        for i, st in enumerate(self._states):
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._textures[i])
            gl.glUniform1i(gl.glGetUniformLocation(self._program, f"uTex[{i}]"), i)
            gl.glUniform1i(gl.glGetUniformLocation(self._program, f"uEnabled[{i}]"), 1 if st.enabled else 0)
            gl.glUniform1f(gl.glGetUniformLocation(self._program, f"uBlack[{i}]"), st.black / 65535.0)
            gl.glUniform1f(gl.glGetUniformLocation(self._program, f"uWhite[{i}]"), st.white / 65535.0)
            gl.glUniform1f(gl.glGetUniformLocation(self._program, f"uGamma[{i}]"), st.gamma)
            gl.glUniform3f(gl.glGetUniformLocation(self._program, f"uColor[{i}]"), *st.color)

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

    def render_to_array(self, scale: float = 1.0) -> np.ndarray | None:
        """Render the composite off-screen at `scale` * source resolution.

        `scale=1.0` renders at full native resolution; `scale=0.5` renders
        directly at half resolution on the GPU (cheaper than rendering full
        size and shrinking afterwards). Returns an (H, W, 3) uint8 RGB
        numpy array, or None if no image is loaded.
        """
        if not self._states or not self._img_w or not self._img_h:
            return None

        out_w = max(1, round(self._img_w * scale))
        out_h = max(1, round(self._img_h * scale))

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
        # Export always shows the full image regardless of the interactive
        # pan/zoom currently on screen.
        saved_zoom, saved_pan = self._zoom, self._pan
        self._zoom, self._pan = 1.0, [0.0, 0.0]
        self._draw(out_w, out_h)
        self._zoom, self._pan = saved_zoom, saved_pan
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

    # ------------------------------------------------------------------
    # Interaction: wheel to zoom, drag to pan, double-click to reset
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        factor = 1.0015 ** event.angleDelta().y()
        self._zoom = max(0.1, min(60.0, self._zoom * factor))
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

        if event.button() == Qt.LeftButton:
            self._drag_start = event.position()
            self._pan_start = tuple(self._pan)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            d = event.position() - self._drag_start
            self._pan[0] = self._pan_start[0] + 2.0 * d.x() / max(self.width(), 1)
            self._pan[1] = self._pan_start[1] - 2.0 * d.y() / max(self.height(), 1)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if not self._measure_mode:
            self.reset_view()

    # ------------------------------------------------------------------
    # Measurement overlay (drawn with QPainter on top of the GL content)
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._measure_points:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
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
        painter.end()
