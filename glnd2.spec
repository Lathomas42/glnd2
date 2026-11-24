# PyInstaller build spec.  pip install pyinstaller && pyinstaller glnd2.spec
#   -> dist/glnd2/glnd2(.exe) plus its _internal/ folder; on macOS also dist/glnd2.app
# A one-folder build (not --onefile) so the app starts instantly instead of
# unpacking ~200 MB of Qt to a temp folder on every launch.
import sys

a = Analysis(
    ["main.py"],
    datas=[("shaders", "shaders"), ("icons/glnd2.png", "icons")],
    # Optional extras dask/nd2 import only if installed; the viewer never uses
    # them, and they'd add ~300 MB when building from a full scientific env.
    excludes=[
        "tkinter", "matplotlib", "IPython", "pytest", "pandas", "scipy",
        "numba", "llvmlite", "h5py", "xarray", "zarr", "cryptography",
        "Cython", "zmq", "jupyter_client", "ipykernel", "distributed",
    ],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="glnd2",
    console=False,
    icon="icons/glnd2.ico",
    # UPX-compressed Qt DLLs trip antivirus scanners and save little.
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="glnd2", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="glnd2.app",
        icon="icons/glnd2.icns",
        bundle_identifier="io.github.lathomas42.glnd2",
        info_plist={"NSHighResolutionCapable": True},
    )
