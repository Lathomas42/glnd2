import os
import shutil
import subprocess
import sys


def _nvidia_gpu_usable() -> bool:
    """True if an NVIDIA driver is installed *and working*. A driver whose
    kernel module doesn't match its libraries (e.g. after an unrebooted
    upgrade) can't create GL contexts, so offloading to it would break."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ensure_discrete_gpu():
    """On Linux hybrid AMD/NVIDIA (PRIME "on-demand") systems, GL defaults to
    the integrated GPU unless offload is requested; re-exec once with the
    offload env vars so the NVIDIA GPU is used. Set GLND2_NO_PRIME=1 to skip."""
    if (not sys.platform.startswith("linux")
            or os.environ.get("__NV_PRIME_RENDER_OFFLOAD") == "1"
            or os.environ.get("GLND2_NO_PRIME")
            or not _nvidia_gpu_usable()):
        return
    env = dict(os.environ, __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia")
    # A packaged executable is its own interpreter: argv[0] already is it.
    args = sys.argv if getattr(sys, "frozen", False) else [sys.executable] + sys.argv
    os.execve(sys.executable, args, env)


def main():
    _ensure_discrete_gpu()

    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from gl_widget import default_gl_format
    from main_window import MainWindow

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    # Lets the loader thread upload textures the viewer can use directly.
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
