# glnd2

GPU-accelerated ND2 viewer for fast per-channel LUT (contrast/gamma/color)
adjustment and composite export: a fast alternative to opening large
multi-channel ND2 files one by one in ImageJ.

Each channel is uploaded once to the GPU as a 16-bit texture; black point,
white point, gamma, tint and additive blending all happen per-frame in a
fragment shader, so dragging sliders stays instant regardless of image size.

## Download (no Python needed)

Ready-made apps are on the repo's **Releases** page (or, for the latest
build, the **Actions** tab > **Build** > a run > *Artifacts*):

- **Windows**: unzip `glnd2-windows-x64.zip`, run `glnd2\glnd2.exe`. Keep the
  `_internal` folder next to it. If SmartScreen warns about an unrecognized
  app, choose *More info* > *Run anyway*.
- **macOS**: unzip and drag `glnd2.app` to Applications (`macos-arm64` for
  Apple Silicon, `macos-x64` for Intel). The app is unsigned, so the first
  launch is blocked: open *System Settings* > *Privacy & Security* and click
  *Open Anyway*, or run `xattr -dr com.apple.quarantine /Applications/glnd2.app`.
- **Linux**: `tar -xzf glnd2-linux-x64.tar.gz` and run `glnd2/glnd2`.

`glnd2 --check [file.nd2]` verifies an install without opening a window.

## Run from source

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
  (**Save Palette…**, stored in `palettes/` when run from source, or the user
  data folder for the downloaded app: `%APPDATA%\glnd2`,
  `~/Library/Application Support/glnd2`, `~/.local/share/glnd2`)
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
- `icons/`: app icon (`glnd2.svg`; `make_icons.py` regenerates the rest)

## Building the apps

`.github/workflows/build.yml` builds all platforms with PyInstaller
(`glnd2.spec`). Run it from the Actions tab, or publish a release by pushing a
tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

To build locally (each OS builds only its own app):
`pip install pyinstaller && pyinstaller glnd2.spec`, output in `dist/`.
