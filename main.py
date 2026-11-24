import os
import shutil
import subprocess
import sys

_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "glnd2.png")


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


def _self_check(nd2_paths: list[str]) -> int:
    """`--check [file.nd2 ...]`: verify a packaged build is complete (every
    module, the shaders, the TIFF/PNG writers, and reading any given ND2
    files) without needing a display or GPU."""
    import tempfile

    import numpy as np
    import tifffile
    from PIL import Image

    import gl_widget
    import main_window  # noqa: F401  (pulls in the rest of the app)
    import nd2_loader

    for name in ("composite.vert", "composite.frag"):
        with open(os.path.join(gl_widget._SHADER_DIR, name)) as f:
            assert "#version" in f.read(), name
    assert os.path.isfile(_ICON_PATH), _ICON_PATH
    arr = np.zeros((4, 4, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        tifffile.imwrite(os.path.join(tmp, "t.tif"), arr, photometric="rgb")
        Image.fromarray(arr).save(os.path.join(tmp, "t.png"))
    for path in nd2_paths:
        img = nd2_loader.load_nd2(path)
        print(f"{os.path.basename(path)}: {img.width}x{img.height}, "
              + ", ".join(f"{c.display_name} bg={c.stats.background:.0f}" for c in img.channels))
    print("glnd2 self-check OK")
    return 0


def main():
    if sys.argv[1:2] == ["--check"]:
        sys.exit(_self_check(sys.argv[2:]))
    _ensure_discrete_gpu()

    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QIcon, QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from gl_widget import default_gl_format
    from main_window import MainWindow

    QSurfaceFormat.setDefaultFormat(default_gl_format())
    # Lets the loader thread upload textures the viewer can use directly.
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(_ICON_PATH))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
