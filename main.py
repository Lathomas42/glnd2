import os
import shutil
import sys

# On hybrid AMD/NVIDIA (PRIME "on-demand") systems, GL contexts default to
# the integrated GPU unless offload is explicitly requested. Re-exec once
# with the offload env vars set so the discrete NVIDIA GPU is always used,
# regardless of how this script is launched. Skipped on machines with no
# NVIDIA driver so this stays safe to run elsewhere.
if os.environ.get("__NV_PRIME_RENDER_OFFLOAD") != "1" and shutil.which("nvidia-smi"):
    env = dict(os.environ, __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from gl_widget import default_gl_format
from main_window import MainWindow


def main():
    QSurfaceFormat.setDefaultFormat(default_gl_format())
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
