# glnd2

GPU-accelerated ND2 viewer for fast per-channel LUT (contrast/gamma/color) adjustment
and composite export, built as a fast alternative to opening large multi-channel ND2
files one by one in ImageJ.

Each channel is uploaded once to the GPU as a 16-bit texture; black point, white
point, gamma, tint color, and additive blending are all computed per-frame in a
GLSL fragment shader, so dragging sliders stays instant regardless of image size.

## Features

- Fast ND2 loading via the [`nd2`](https://github.com/tlambert03/nd2) package
- Per-channel controls: visibility, color, black/white point, gamma, live histogram
- Auto-contrast (percentile-based) and reset per channel
- Folder-based batch workflow: Prev/Next, Save & Next, Batch Export All
- LUT settings carry over between files by channel name, and can be saved/loaded
  as JSON presets
- Composite export to PNG or TIFF at full source resolution

## Requirements

- Python 3.10+
- An OpenGL 4.3+ capable GPU
- On hybrid AMD/NVIDIA (PRIME "on-demand") systems, `main.py` automatically
  re-execs itself with NVIDIA PRIME render offload env vars so the discrete
  GPU is used instead of the integrated one

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 main.py
```

Open a folder of `.nd2` files, adjust each channel's LUT curve, then use
`Ctrl+S` to save the composite next to the source file, or `Ctrl+Enter` to
save and advance to the next file.

## Layout

- `nd2_loader.py` — ND2 reading, channel stats, default wavelength-based coloring
- `gl_widget.py` — GPU compositor (`QOpenGLWidget` + GLSL shader)
- `shaders/` — vertex/fragment shaders for the compositor
- `channel_panel.py` — per-channel LUT control widget
- `main_window.py` — file browser, batch workflow, save/export, presets
- `main.py` — entry point
