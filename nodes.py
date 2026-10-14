"""ComfyUI nodes for ComfyUI-TerminalTextEffects (https://github.com/ChrisBuilds/terminaltexteffects)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from . import tte_core as core

try:  # available inside ComfyUI, absent when testing standalone
    import comfy.model_management as _mm
    import comfy.utils as _cu
except Exception:  # pragma: no cover
    _mm = _cu = None


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _to_uint8(images: torch.Tensor) -> np.ndarray:
    """[B,H,W,C] float 0..1  ->  [B,H,W,3] uint8 (converted in chunks to keep peak memory low)."""
    chunks = []
    for i in range(0, images.shape[0], 8):
        c = images[i:i + 8].detach().cpu().float().clamp(0.0, 1.0)
        chunks.append((c.mul(255.0).add(0.5)).byte().numpy())
    arr = np.concatenate(chunks, axis=0)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr[..., :3]


def _make_video(frames: torch.Tensor, fps: float):
    """Build ComfyUI's native VIDEO object (works with Save Video, Get Video Components, ...)."""
    frac = Fraction(fps).limit_denominator(1000)
    try:                                                   # current ComfyUI
        from comfy_api.latest import InputImpl, Types
        return InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=None, frame_rate=frac))
    except Exception:
        pass
    try:                                                   # older ComfyUI (0.3.4x - 0.3.5x)
        from comfy_api.input_impl import VideoFromComponents
        from comfy_api.util import VideoComponents
        return VideoFromComponents(VideoComponents(images=frames, audio=None, frame_rate=frac))
    except Exception:
        return None                                        # very old ComfyUI: use `frames` + `fps`


def _check_interrupt():
    if _mm is not None:
        _mm.throw_exception_if_processing_interrupted()


# --------------------------------------------------------------------------- #
#  Main node
# --------------------------------------------------------------------------- #

class TTE_AsciiEffectVideo:
    """Image  -> ASCII animation made with terminaltexteffects.
    Batch  -> ASCII video, optionally with a terminaltexteffects intro / outro."""

    CATEGORY = "video/TerminalTextEffects"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "FLOAT", "VIDEO", "STRING")
    RETURN_NAMES = ("frames", "fps", "video", "info")

    @classmethod
    def INPUT_TYPES(cls):
        effects = core.effect_names()
        d = core.Params()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "One image = animated still. An image batch (video) = ASCII video."}),
                "effect": (effects, {
                    "default": "decrypt" if "decrypt" in effects else effects[0],
                    "tooltip": "Terminal Text Effect used to animate the ASCII art. Use the 'TTE Effect Options' "
                               "node to see what each effect can be tuned with."}),
                "duration_seconds": ("FLOAT", {
                    "default": d.duration_seconds, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Length of the final video. Single image: the effect is time-stretched to fit "
                               "(0 = play at the effect's natural speed, 1 effect tick per video frame). "
                               "Video input: only used when video_timing = 'fit to duration_seconds'."}),
                "fps": ("INT", {"default": d.fps, "min": 1, "max": 120, "step": 1,
                                "tooltip": "Frames per second of the output video."}),
                "columns": ("INT", {
                    "default": d.columns, "min": 16, "max": 400, "step": 1,
                    "tooltip": "Approximate number of characters across. More columns = more detail but the "
                               "effects get much slower (cost grows with columns x rows)."}),
                "charset": (list(core.RAMPS.keys()), {
                    "default": d.charset,
                    "tooltip": "Characters used to draw the image, from darkest to brightest."}),
                "color_mode": (core.COLOR_MODES, {
                    "default": d.color_mode,
                    "tooltip": "image + effect colors: the effect's own colours while animating, image colours at "
                               "the end.  image colors only: characters always keep the image colours.  "
                               "effect palette only: colours come from the effect's gradient.  "
                               "single color: everything ends in foreground_color."}),
                "playback": (core.PLAYBACK_MODES, {
                    "default": d.playback,
                    "tooltip": "Single image: forward builds the picture, reverse dissolves it, "
                               "'forward then reverse' does both. (Video input: intro plays forward, outro reversed.)"}),
                "hold_end_seconds": ("FLOAT", {
                    "default": d.hold_end_seconds, "min": 0.0, "max": 60.0, "step": 0.1,
                    "tooltip": "Single image: how long the finished ASCII picture stays on screen "
                               "(included in duration_seconds)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Effects are random; the same seed gives the same animation."}),
            },
            "optional": {
                # ---- video input -------------------------------------------------------------
                "video_mode": (core.VIDEO_MODES, {
                    "default": d.video_mode,
                    "tooltip": "Only used when the input is an image batch (video). 'ascii only' just converts "
                               "every frame to ASCII; the other modes add a terminaltexteffects intro and/or outro."}),
                "video_timing": (core.VIDEO_TIMING, {
                    "default": d.video_timing,
                    "tooltip": "Keep the original length, or speed up / slow down the clip to duration_seconds."}),
                "video_effect_seconds": ("FLOAT", {
                    "default": d.video_effect_seconds, "min": 0.2, "max": 60.0, "step": 0.1,
                    "tooltip": "Length of the intro and of the outro effect (video input)."}),
                "video_keeps_playing": ("BOOLEAN", {
                    "default": d.video_keeps_playing,
                    "tooltip": "ON: the video keeps playing inside the effect (the flying characters show the "
                               "live video). OFF: the effect runs on a frozen first/last frame and the clip is "
                               "lengthened by the intro/outro."}),
                "input_fps": ("INT", {"default": d.input_fps, "min": 0, "max": 240,
                                      "tooltip": "Frame rate of the input video (0 = same as fps). "
                                                 "Used to retime the clip to the output fps."}),
                # ---- look --------------------------------------------------------------------
                "custom_charset": ("STRING", {"default": d.custom_charset,
                                              "tooltip": "Used when charset = custom (dark -> bright)."}),
                "invert": ("BOOLEAN", {"default": d.invert, "tooltip": "Swap dense/sparse characters."}),
                "gamma": ("FLOAT", {"default": d.gamma, "min": 0.2, "max": 4.0, "step": 0.05,
                                    "tooltip": "<1 brightens the ASCII picture, >1 darkens it."}),
                "auto_contrast": ("BOOLEAN", {"default": d.auto_contrast,
                                              "tooltip": "Stretch the image contrast before choosing characters."}),
                "color_boost": ("FLOAT", {"default": d.color_boost, "min": 0.2, "max": 4.0, "step": 0.05,
                                          "tooltip": "Brightness multiplier for image colours."}),
                "background_color": ("STRING", {"default": d.background_color, "tooltip": "Hex colour, e.g. #000000"}),
                "foreground_color": ("STRING", {"default": d.foreground_color,
                                                "tooltip": "Hex colour, used by color_mode = single color."}),
                "font": (core.list_fonts(), {"tooltip": "Monospace fonts found in this node's 'fonts' folder."}),
                "font_path": ("STRING", {"default": "", "tooltip": "Optional absolute path to a .ttf/.otf "
                                                                   "monospace font (overrides 'font')."}),
                "line_spacing": ("FLOAT", {"default": d.line_spacing, "min": 0.6, "max": 2.0, "step": 0.05,
                                           "tooltip": "Row height multiplier."}),
                "output_scale": ("FLOAT", {"default": d.output_scale, "min": 0.25, "max": 4.0, "step": 0.05,
                                           "tooltip": "Output size relative to the input image."}),
                "max_side": ("INT", {"default": d.max_side, "min": 0, "max": 8192, "step": 2,
                                     "tooltip": "Clamp the longest side of the output (0 = no limit). "
                                                "Protects your RAM on long videos."}),
                # ---- advanced ----------------------------------------------------------------
                "effect_options": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional JSON overriding the chosen effect's settings, e.g. "
                               "{\"typing_speed\": 3, \"final_gradient_stops\": [\"ff0066\", \"00d1ff\"]}. "
                               "Get the list of valid keys from the 'TTE Effect Options' node."}),
                "max_effect_frames": ("INT", {"default": d.max_effect_frames, "min": 200, "max": 50000,
                                              "tooltip": "Safety limit on the number of effect ticks."}),
            },
        }

    def run(self, images, effect, duration_seconds, fps, columns, charset, color_mode, playback,
            hold_end_seconds, seed, **opt):
        params = core.Params(
            effect=effect, duration_seconds=duration_seconds, fps=fps, columns=columns, charset=charset,
            color_mode=color_mode, playback=playback, hold_end_seconds=hold_end_seconds, seed=int(seed) & 0xFFFFFFFF,
        )
        for k, v in opt.items():
            if hasattr(params, k) and v is not None:
                setattr(params, k, v)
        if params.font and params.font.startswith("("):
            params.font = ""

        bar = _cu.ProgressBar(1) if _cu is not None else None

        def progress(done, total):
            if bar is not None:
                bar.update_absolute(done, total)

        frames_u8, out_fps, info = core.generate(
            _to_uint8(images), params, progress=progress, check=_check_interrupt)
        frames = torch.from_numpy(frames_u8).float().div_(255.0)
        return (frames, out_fps, _make_video(frames, out_fps), info)


# --------------------------------------------------------------------------- #
#  Helper node: list the tunable options of an effect
# --------------------------------------------------------------------------- #

class TTE_EffectOptions:
    """Outputs the default settings of an effect as JSON - edit it and feed it into `effect_options`."""

    CATEGORY = "video/TerminalTextEffects"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("options_json",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        effects = core.effect_names()
        return {"required": {"effect": (effects, {"default": "decrypt" if "decrypt" in effects else effects[0]})}}

    def run(self, effect):
        text = core.describe_effect(effect)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "TTE_AsciiEffectVideo": TTE_AsciiEffectVideo,
    "TTE_EffectOptions": TTE_EffectOptions,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TTE_AsciiEffectVideo": "TTE ASCII Effect Video (Terminal Text Effects)",
    "TTE_EffectOptions": "TTE Effect Options",
}
