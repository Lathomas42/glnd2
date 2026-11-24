# glnd2

GPU-accelerated ND2 viewer for fast per-channel LUT (contrast/gamma/color)
adjustment and composite export: a fast alternative to opening large
multi-channel ND2 files one by one in ImageJ.

Each channel is uploaded once to the GPU as a 16-bit texture; black point,
white point, gamma, tint and additive blending all happen per-frame in a
fragment shader, so dragging sliders stays instant regardless of image size.

## Install and run

```bash
pip install -r requirements.txt
python main.py
```

Needs Python 3.10+ and an OpenGL 4.1 capable GPU (Windows, Linux, macOS).
On Linux hybrid AMD/NVIDIA laptops/desktops, `main.py` automatically switches
to the NVIDIA GPU when its driver is working; set `GLND2_NO_PRIME=1` to opt out.

## How a file's LUTs are chosen

Every file opens with, in order of precedence:

1. **Its own saved LUTs**: press **Save** (`Ctrl+S`) to keep the current
   settings for that file only. Saved files are marked **●** in the file
   list; **Revert** deletes them. Saved LUTs never carry over to other files.
2. **Whatever you had on the previous file**: levels, gamma, colors and
   visibility carry over as you browse.
3. **The Default LUT** (if **Use Default LUT** is on), for channels with
   nothing to carry over; **Set Default LUT…** picks the preset.
4. **Auto-contrast**: black just above the background peak (inky black
   background), white well above the 98th percentile (mostly dark, a few
   bright pixels, almost nothing saturated).

**Auto** above the channel list auto-contrasts every channel of the current
file (each channel also has its own **Auto**). **Auto headroom** in the toolbar
sets how far above the 98th percentile white goes: higher is darker.

**Export…** renders other files the same way, using what's on screen now as
the carried-over settings, in a hidden view so the screen isn't disturbed.

The next file in the list is read ahead in the background, so **Next ▶** only
has to upload it to the GPU.

## Other features

- Per-channel controls: visibility, color, black/white point, gamma, histogram
- Channels recognized by wavelength (`640 nm`) or name (`AF647 Prescan`, `GFP`)
  and labeled by fluorophore; switchable color palettes, plus your own
  (**Save Palette…**, stored locally in `palettes/`)
- **Export…** (`Ctrl+E`): current file or whole folder, PNG/TIFF, downsampling,
  optional preset, or just the ROI rectangles
- **Measure** (`Ctrl+M`): two-click ruler in µm (from the ND2's calibration)
- **Draw ROI** (`Ctrl+R`): named rectangles, listed in the ROIs panel; click one
  to jump to it
- Wheel to zoom (around the cursor), drag to pan, double-click to reset

## Files glnd2 writes next to your data

- `glnd2_luts/<file>.json`: per-file saved LUTs
- `glnd2_rois.json`: ROIs for the folder

## Layout

- `main.py`: entry point (GPU selection)
- `main_window.py`: file browser, LUT resolution, tools, export, presets
- `gl_widget.py` + `shaders/`: GPU compositor
- `channel_panel.py`: per-channel LUT controls
- `nd2_loader.py`: ND2 reading, intensity stats, auto-contrast, palettes
- `export_dialog.py`, `roi_store.py`, `file_lut_store.py`
