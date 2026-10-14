# ComfyUI-TerminalTextEffects

ComfyUI nodes that bring [**terminaltexteffects**](https://github.com/ChrisBuilds/terminaltexteffects)
(decrypt, beams, burn, blackhole, matrix, fireworks, … 37 effects, auto-discovered) to images and videos.

* **Image in → video out.** The image is converted to colored ASCII art, and the chosen effect builds
  (or dissolves) it. You pick the effect and the exact length of the final video.
* **Video in → ASCII video out.** Feed an image *batch*: every frame becomes ASCII, optionally with an
  effect intro / outro. In "live" mode the video keeps playing *inside* the effect.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-TerminalTextEffects      # or unzip the folder here
# use ComfyUI's own Python (venv / python_embeded):
python -m pip install -r ComfyUI-TerminalTextEffects/requirements.txt
```
Restart ComfyUI. Nodes appear under **video/TerminalTextEffects**. (Tested with terminaltexteffects 0.15.0.)

## Quick start

```
Load Image ─▶ TTE ASCII Effect Video ─▶ video  ─▶ Save Video
                                       └ frames + fps ─▶ Video Combine (VHS) / any IMAGE node
```
Outputs: `frames` (IMAGE batch), `fps` (FLOAT), `video` (native ComfyUI VIDEO), `info` (STRING).
For a video file: `Load Video` → this node (batch in) → `Save Video`.

## Main inputs

| input | meaning |
|---|---|
| `effect` | Any terminaltexteffects effect. |
| `duration_seconds` | **Length of the final video.** The effect is time-stretched to fit. `0` = natural effect speed (1 tick = 1 frame). For video input it is only used with `video_timing = fit to duration_seconds`. |
| `fps` | Output frame rate. |
| `columns` | Characters across (approx.). More = finer detail **and much slower effects**. 80–120 is the sweet spot. |
| `charset` | `classic`, `detailed`, `blocks`, `minimal`, `binary`, `dots`, `custom`. |
| `color_mode` | `image + effect colors` (effect colors while animating, image colors at the end), `image colors only`, `effect palette only`, `single color`. |
| `playback` | `forward` (build), `reverse` (dissolve), `forward then reverse`. |
| `hold_end_seconds` | How long the finished ASCII picture stays on screen (part of `duration_seconds`). |
| `seed` | Same seed = same animation. |

### Video input (image batch with more than one frame)

| input | meaning |
|---|---|
| `video_mode` | `ascii only`, `effect intro`, `effect outro`, `effect intro + outro`. |
| `video_effect_seconds` | Length of each intro/outro. |
| `video_keeps_playing` | **ON (live):** the effect plays over the first/last frames *while the video keeps moving* – flying characters show the live video. **OFF:** effect runs on a frozen first/last frame and the clip gets longer. |
| `video_timing` | Keep the original length, or retime the clip to `duration_seconds`. |
| `input_fps` | Frame rate of the input clip (0 = same as `fps`), used to retime to the output fps. |

### Look & advanced
`invert`, `gamma`, `auto_contrast`, `color_boost`, `background_color`, `foreground_color`, `font` / `font_path`
(any monospace TTF; drop fonts into `fonts/`), `line_spacing`, `output_scale`, `max_side`, `max_effect_frames`.

### Tuning an effect (`effect_options`)
Add a **TTE Effect Options** node, pick the effect, and connect its output to a *Preview Any* node: it prints
every setting of the effect with its default. Paste only the keys you want to change into `effect_options`:

```json
{"typing_speed": 3, "ciphertext_colors": ["ff0066", "ff9900"], "final_gradient_direction": "horizontal"}
```
Colors are hex strings, enums are lowercase names. Unknown keys give an error that lists the valid ones.

## Performance notes

* Cost grows with `columns × rows`. At ~100–120 columns on 1280×720, a 6 s / 24 fps clip takes roughly 15–20 s
  (Matrix, Thunderstorm and Rings are the slowest effects).
* The node renders frames in RAM (`frames × H × W × 3`), so use `max_side` / `output_scale` for long videos.
* Some effects run thousands of internal ticks; the node samples them evenly to fit your duration, and
  skips formatting of frames it does not need.

## Known limitations

* Blackhole (an upstream quirk) can crash on *very* small grids (about 12 rows). The node retries with other
  seeds and then tells you to change `columns`.
* `video_keeps_playing` relies on terminaltexteffects' character API; if a future release changes it, the node
  falls back to the frozen-frame behavior and logs a message.
* Spaces/blank cells are real characters in live mode (shown as `.` placeholders until the effect settles).

## Credits / licenses
Effects by [ChrisBuilds/terminaltexteffects](https://github.com/ChrisBuilds/terminaltexteffects) (separate
project, see its license). Bundled font: DejaVu Sans Mono (`fonts/LICENSE_DEJAVU`). Node code: MIT.
