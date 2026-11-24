"""Regenerate the raster icons from glnd2.svg: python icons/make_icons.py

glnd2.png (window icon), glnd2.ico (Windows exe), glnd2.icns (macOS app).
"""
import os
import sys
from io import BytesIO

from PIL import Image
from PySide6.QtCore import QBuffer, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = os.path.dirname(os.path.abspath(__file__))


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    renderer.render(p)
    p.end()
    buf = QBuffer()
    buf.open(QBuffer.ReadWrite)
    img.save(buf, "PNG")
    return Image.open(BytesIO(bytes(buf.data()))).convert("RGBA")


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication(sys.argv[:1])  # noqa: F841  (QPainter needs one)
    renderer = QSvgRenderer(os.path.join(HERE, "glnd2.svg"))
    big = render(renderer, 1024)
    render(renderer, 256).save(os.path.join(HERE, "glnd2.png"))
    # Render each size from the vector instead of letting Pillow downscale.
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [render(renderer, s) for s in ico_sizes]
    frames[-1].save(os.path.join(HERE, "glnd2.ico"), sizes=[(s, s) for s in ico_sizes],
                    append_images=frames[:-1])
    big.save(os.path.join(HERE, "glnd2.icns"))
    print("wrote glnd2.png, glnd2.ico, glnd2.icns")


if __name__ == "__main__":
    main()
