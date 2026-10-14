"""
Core engine for ComfyUI-TerminalTextEffects.

Torch-free on purpose (numpy + Pillow + terminaltexteffects only) so it can be
tested outside ComfyUI.  `nodes.py` is just a thin tensor <-> numpy wrapper.

Pipeline
--------
 image(s) -> ASCII cell grid (chars + colours)
          -> ANSI text -> terminaltexteffects effect -> text frames
          -> frames parsed back to (char, colour) grids
          -> rasterised with a glyph atlas (vectorised numpy) -> RGB frames
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(NODE_DIR, "fonts")

# --------------------------------------------------------------------------- #
#  Effect discovery / configuration
# --------------------------------------------------------------------------- #

FALLBACK_EFFECTS = [
    "beams", "binarypath", "blackhole", "bouncyballs", "bubbles", "burn", "colorshift",
    "crumble", "decrypt", "errorcorrect", "expand", "fireworks", "highlight", "laseretch",
    "matrix", "middleout", "orbittingvolley", "overflow", "pour", "print", "rain",
    "random_sequence", "rings", "scattered", "slice", "slide", "smoke", "spotlights",
    "spray", "swarm", "sweep", "synthgrid", "thunderstorm", "unstable", "vhstape",
    "waves", "wipe",
]

_EFFECTS_CACHE: Optional[dict] = None


def tte_import_error() -> Optional[str]:
    try:
        import terminaltexteffects  # noqa: F401
        return None
    except Exception as e:  # pragma: no cover
        return str(e)


def discover_effects() -> dict:
    """{name: EffectClass} for every effect shipped with the installed terminaltexteffects.

    Auto-discovered, so effects added in newer TTE releases show up automatically.
    """
    global _EFFECTS_CACHE
    if _EFFECTS_CACHE is not None:
        return _EFFECTS_CACHE
    out = {}
    try:
        import terminaltexteffects.effects as E
        from terminaltexteffects.engine.base_effect import BaseEffect
        for m in pkgutil.iter_modules(E.__path__):
            if not m.name.startswith("effect_"):
                continue
            try:
                mod = importlib.import_module(f"{E.__name__}.{m.name}")
            except Exception:
                continue
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, BaseEffect) and obj is not BaseEffect and obj.__module__ == mod.__name__:
                    out[m.name[len("effect_"):]] = obj
                    break
    except Exception:
        out = {}
    _EFFECTS_CACHE = dict(sorted(out.items()))
    return _EFFECTS_CACHE


def effect_names() -> list:
    names = list(discover_effects().keys())
    return names or list(FALLBACK_EFFECTS)


def _hex_of(color) -> str:
    return color.rgb_color


def _jsonable(v):
    from terminaltexteffects.utils.graphics import Color
    if isinstance(v, Color):
        return _hex_of(v)
    if isinstance(v, Enum):
        return v.name.lower()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)


def _config_items(cfg):
    import dataclasses
    for f in dataclasses.fields(cfg):
        if f.name == "parser_spec":
            continue
        yield f.name, getattr(cfg, f.name)


def describe_effect(name: str) -> str:
    """Default effect configuration as JSON, usable as the `effect_options` input."""
    cls = discover_effects().get(name)
    if cls is None:
        return "{}"
    eff = cls("x")
    d = {k: _jsonable(v) for k, v in _config_items(eff.effect_config)}
    return json.dumps(d, indent=2, ensure_ascii=False)


def _coerce(current, value, key):
    from terminaltexteffects.utils.graphics import Color
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Color):
        return Color(value if isinstance(value, int) else str(value))
    if isinstance(current, Enum):
        if isinstance(value, str):
            for m in type(current):
                if m.name.lower() == value.strip().lower():
                    return m
            raise EffectOptionError(f"'{key}': '{value}' is not one of {[m.name.lower() for m in type(current)]}")
        return type(current)(value)
    if isinstance(current, (tuple, list)):
        if not isinstance(value, (list, tuple)):
            value = [value]
        sample = current[0] if len(current) else None
        if isinstance(sample, Color):
            return tuple(Color(v if isinstance(v, int) else str(v)) for v in value)
        if isinstance(sample, bool):
            return tuple(bool(v) for v in value)
        if isinstance(sample, int):
            return tuple(int(v) for v in value)
        if isinstance(sample, float):
            return tuple(float(v) for v in value)
        if isinstance(sample, (tuple, list)):
            return tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in value)
        return tuple(value)
    if isinstance(current, str):
        return str(value)
    return value


class EffectOptionError(ValueError):
    """The user-supplied effect_options are invalid (never retried)."""


def apply_effect_options(cfg, options: dict):
    valid = {k: v for k, v in _config_items(cfg)}
    for key, value in options.items():
        if key not in valid:
            raise EffectOptionError(
                f"Unknown effect option '{key}'. Valid options for this effect: {sorted(valid)}"
            )
        try:
            setattr(cfg, key, _coerce(valid[key], value, key))
        except EffectOptionError:
            raise
        except Exception as e:
            raise EffectOptionError(f"Invalid value {value!r} for effect option '{key}': {e}") from e


def parse_effect_options(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"effect_options is not valid JSON: {e}") from e
    if not isinstance(d, dict):
        raise ValueError("effect_options must be a JSON object, e.g. {\"movement_speed\": 0.8}")
    return d


# --------------------------------------------------------------------------- #
#  ANSI parsing
# --------------------------------------------------------------------------- #

ANSI16 = [
    (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
]


def _xterm_rgb(n: int):
    n = max(0, min(255, n))
    if n < 16:
        return ANSI16[n]
    if n < 232:
        n -= 16
        lv = (0, 95, 135, 175, 215, 255)
        return (lv[n // 36], lv[(n // 6) % 6], lv[n % 6])
    g = 8 + 10 * (n - 232)
    return (g, g, g)


_TOKEN = re.compile(
    r"\x1b\[([0-9;:]*)m"                       # SGR
    r"|\x1b\[[0-9;:?<=>]*[ -/]*[@-~]"          # any other CSI -> ignored
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"      # OSC -> ignored
    r"|(.)",                                   # a visible character
    re.S,
)
_SGR_CACHE: dict = {}


def _decode_sgr(params: str):
    nums = [int(x) if x else 0 for x in re.split(r"[;:]", params)] if params else [0]
    ops, i = [], 0
    while i < len(nums):
        n = nums[i]
        if n == 0:
            ops.append(("reset", None))
        elif n == 39:
            ops.append(("fg", None))
        elif n == 49:
            ops.append(("bg", None))
        elif 30 <= n <= 37:
            ops.append(("fg", ANSI16[n - 30]))
        elif 90 <= n <= 97:
            ops.append(("fg", ANSI16[n - 90 + 8]))
        elif 40 <= n <= 47:
            ops.append(("bg", ANSI16[n - 40]))
        elif 100 <= n <= 107:
            ops.append(("bg", ANSI16[n - 100 + 8]))
        elif n in (38, 48):
            kind = "fg" if n == 38 else "bg"
            if i + 1 < len(nums) and nums[i + 1] == 2 and i + 4 < len(nums):
                ops.append((kind, tuple(max(0, min(255, v)) for v in nums[i + 2:i + 5])))
                i += 4
            elif i + 1 < len(nums) and nums[i + 1] == 5 and i + 2 < len(nums):
                ops.append((kind, _xterm_rgb(nums[i + 2])))
                i += 2
        i += 1
    return tuple(ops)


def parse_frame(text: str, rows: int, cols: int, default_fg=(255, 255, 255), default_bg=(0, 0, 0)):
    """ANSI text frame -> (codepoints[R,C] int32, fg[R,C,3] uint8, bg[R,C,3] uint8 | None)."""
    cps = np.full((rows, cols), 32, np.int32)
    fg = np.empty((rows, cols, 3), np.uint8)
    fg[:] = default_fg
    bg = None
    default_fg = tuple(default_fg)
    for r, line in enumerate(text.split("\n")[:rows]):
        cur_fg, cur_bg = None, None
        row_cp, row_fg, row_bg, any_bg = [], [], [], False
        for m in _TOKEN.finditer(line):
            p = m.group(1)
            if p is not None:
                ops = _SGR_CACHE.get(p)
                if ops is None:
                    ops = _SGR_CACHE[p] = _decode_sgr(p)
                for kind, val in ops:
                    if kind == "reset":
                        cur_fg = cur_bg = None
                    elif kind == "fg":
                        cur_fg = val
                    else:
                        cur_bg = val
                continue
            ch = m.group(2)
            if ch is None:
                continue
            if len(row_cp) >= cols:
                break
            row_cp.append(ord(ch))
            row_fg.append(cur_fg if cur_fg is not None else default_fg)
            row_bg.append(cur_bg)
            if cur_bg is not None:
                any_bg = True
        n = len(row_cp)
        if n:
            cps[r, :n] = row_cp
            fg[r, :n] = row_fg
            if any_bg:
                if bg is None:
                    bg = np.empty((rows, cols, 3), np.uint8)
                    bg[:] = default_bg
                for c, b in enumerate(row_bg):
                    if b is not None:
                        bg[r, c] = b
    return cps, fg, bg


# --------------------------------------------------------------------------- #
#  Fonts, glyph atlas, rasteriser
# --------------------------------------------------------------------------- #

_SYSTEM_MONO = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
]


def list_fonts() -> list:
    names = []
    if os.path.isdir(FONT_DIR):
        for f in sorted(os.listdir(FONT_DIR)):
            if f.lower().endswith((".ttf", ".otf", ".ttc")):
                names.append(f)
    if "DejaVuSansMono.ttf" in names:
        names.remove("DejaVuSansMono.ttf")
        names.insert(0, "DejaVuSansMono.ttf")
    return names or ["(system monospace)"]


def resolve_font(font_name: str, font_path: str) -> str:
    if font_path and os.path.isfile(font_path):
        return font_path
    cand = os.path.join(FONT_DIR, font_name or "")
    if font_name and os.path.isfile(cand):
        return cand
    for c in _SYSTEM_MONO:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "No monospace font found. Put a .ttf in the node's 'fonts' folder or set 'font_path'."
    )


class GlyphAtlas:
    """Caches one alpha mask (ch x cw) per codepoint, rendered lazily with Pillow."""

    def __init__(self, font_path: str, size: int, cw: int, ch: int):
        self.font = ImageFont.truetype(font_path, size)
        asc, desc = self.font.getmetrics()
        self.cw, self.ch = cw, ch
        self.baseline = asc + max(0, (ch - (asc + desc)) // 2)
        self._index = {32: 0}
        self._masks = np.zeros((128, ch, cw), np.uint8)
        self._n = 1

    def _add(self, cp: int) -> int:
        if cp < 32 or cp == 127:
            self._index[cp] = 0
            return 0
        img = Image.new("L", (self.cw, self.ch), 0)
        ImageDraw.Draw(img).text((0, self.baseline), chr(cp), font=self.font, fill=255, anchor="ls")
        if self._n >= self._masks.shape[0]:
            self._masks = np.concatenate([self._masks, np.zeros_like(self._masks)], axis=0)
        self._masks[self._n] = np.asarray(img)
        self._index[cp] = self._n
        self._n += 1
        return self._n - 1

    def lookup(self, cps: np.ndarray) -> np.ndarray:
        flat = cps.reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        ids = np.empty(len(uniq), np.int32)
        for i, cp in enumerate(uniq.tolist()):
            j = self._index.get(cp)
            ids[i] = j if j is not None else self._add(cp)
        return ids[inv.reshape(-1)].reshape(cps.shape)

    @property
    def masks(self):
        return self._masks


def fit_cell(font_path: str, target_cw: int, line_spacing: float):
    ref = ImageFont.truetype(font_path, 100)
    ratio = ref.getlength("M") / 100.0
    size = max(5, int(round(target_cw / ratio)))
    font = ImageFont.truetype(font_path, size)
    asc, desc = font.getmetrics()
    ch = max(4, int(round((asc + desc) * line_spacing)))
    return size, target_cw, ch


class Renderer:
    def __init__(self, atlas: GlyphAtlas, rows: int, cols: int, out_w: int, out_h: int, bg_rgb):
        self.atlas, self.rows, self.cols = atlas, rows, cols
        self.out_w, self.out_h = out_w, out_h
        self.ox = (out_w - cols * atlas.cw) // 2
        self.oy = (out_h - rows * atlas.ch) // 2
        self.bg = np.array(bg_rgb, np.uint8)

    def render(self, cps, fg, cell_bg=None) -> np.ndarray:
        R, C = cps.shape
        cw, ch = self.atlas.cw, self.atlas.ch
        idx = self.atlas.lookup(cps)
        a = self.atlas.masks[idx][..., None].astype(np.int32)              # R,C,ch,cw,1
        f = fg[:, :, None, None, :].astype(np.int32)                       # R,C,1,1,3
        if cell_bg is None:
            b = self.bg.astype(np.int32)[None, None, None, None, :]
        else:
            b = cell_bg[:, :, None, None, :].astype(np.int32)
        out = ((b * (255 - a) + f * a + 127) // 255).astype(np.uint8)
        out = out.transpose(0, 2, 1, 3, 4).reshape(R * ch, C * cw, 3)
        canvas = np.empty((self.out_h, self.out_w, 3), np.uint8)
        canvas[:] = self.bg
        canvas[self.oy:self.oy + R * ch, self.ox:self.ox + C * cw] = out
        return canvas


# --------------------------------------------------------------------------- #
#  Image -> ASCII grid
# --------------------------------------------------------------------------- #

RAMPS = {
    "classic": " .:-=+*#%@",
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks": " ░▒▓█",
    "minimal": " .oO@",
    "binary": " 01",
    "dots": " ·∙•●",
    "custom": None,
}
COLOR_MODES = ["image + effect colors", "image colors only", "effect palette only", "single color"]
_TTE_COLOR_HANDLING = {
    "image + effect colors": "dynamic",
    "image colors only": "always",
    "effect palette only": "ignore",
    "single color": "dynamic",
}

_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")
_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def hex_to_rgb(s: str, default=(0, 0, 0)):
    s = (s or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


@dataclass
class Grid:
    cps: np.ndarray        # (R,C) int32 codepoints
    rgb: np.ndarray        # (R,C,3) uint8 per-cell colour


def build_grid(img_u8, rows, cols, p: "Params", palette: Optional[np.ndarray] = None) -> Grid:
    ramp = RAMPS.get(p.charset) or (p.custom_charset or " .:-=+*#%@")
    if len(ramp) < 2:
        ramp = " " + (ramp or "#")
    pil = Image.fromarray(img_u8)
    down = pil.width * pil.height > cols * rows
    small = np.asarray(pil.convert("RGB").resize((cols, rows), _BOX if down else _BICUBIC))
    f = small.astype(np.float32) / 255.0
    lum = f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if p.auto_contrast:
        lo, hi = np.percentile(lum, (1, 99))
        if hi - lo > 1e-3:
            lum = np.clip((lum - lo) / (hi - lo), 0, 1)
    lum = np.clip(lum, 0, 1) ** max(p.gamma, 1e-3)
    if p.invert:
        lum = 1.0 - lum
    idx = np.rint(lum * (len(ramp) - 1)).astype(np.int32)
    cps = np.array([ord(c) for c in ramp], np.int32)[idx]

    fg_rgb = hex_to_rgb(p.foreground_color, (0, 255, 65))
    if p.color_mode in ("image + effect colors", "image colors only"):
        rgb = (np.clip(f * p.color_boost, 0, 1) * 255 + 0.5).astype(np.uint8)
    elif p.color_mode == "effect palette only" and palette is not None:
        rgb = palette
    else:  # single colour (also the fallback for 'effect palette' when no effect ran)
        rgb = np.empty((rows, cols, 3), np.uint8)
        rgb[:] = fg_rgb
    return Grid(cps, rgb)


def grid_to_ansi(grid: Grid, colored: bool = True, placeholders: bool = False) -> str:
    """Grid -> ANSI text for TTE.  `placeholders` replaces blanks by '.' so that every cell
    is a real, trackable character (needed for the live-video mode)."""
    R, C = grid.cps.shape
    lines = []
    for r in range(R):
        if colored:
            parts = []
            for c in range(C):
                ch = chr(int(grid.cps[r, c]))
                if ch == " " and placeholders:
                    ch = "."
                if ch == " ":
                    parts.append(" ")
                else:
                    cr, cg, cb = grid.rgb[r, c]
                    parts.append(f"\x1b[38;2;{cr};{cg};{cb}m{ch}")
            lines.append("".join(parts) + "\x1b[0m")
        else:
            lines.append("".join(("." if (x == 32 and placeholders) else chr(int(x))) for x in grid.cps[r]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Running an effect (with decimation so memory/time stay bounded)
# --------------------------------------------------------------------------- #

class _Gate:
    """Replaces terminal.get_formatted_output_string so that frames we are going to
    throw away are never formatted (formatting is ~2/3 of the per-frame cost)."""

    def __init__(self, orig):
        self.orig = orig
        self.want = True

    def __call__(self):
        return self.orig() if self.want else ""


@dataclass
class EffectRun:
    items: list                 # captured frames (text, or LiveSnap in live mode)
    ticks: np.ndarray           # effect tick index of every captured item
    total_ticks: int
    truncated: bool
    seconds: float


class LiveSnap:
    __slots__ = ("pr", "pc", "cp", "rgb", "sr", "sc", "sym_same", "col_same", "has_id")


def _snapshot_live(it, rows, cols, ctx) -> LiveSnap:
    term = it.terminal
    if "ids" not in ctx:
        ctx["ids"] = {c.character_id for c in term.get_characters()}
        ctx["hex"] = {}
    ids, hexc = ctx["ids"], ctx["hex"]
    chars = getattr(term, "_visible_characters", None)
    if chars is None:
        chars = [c for c in term.get_characters(input_chars=True, inner_fill_chars=True,
                                                outer_fill_chars=True, added_chars=True) if c.is_visible]
    chars = sorted(chars, key=lambda c: c.layer)
    pr, pc, cp, rgb, sr, sc, ss, cs, hid = [], [], [], [], [], [], [], [], []
    for ch in chars:
        cc = ch.motion.current_coord
        vis = ch.animation.current_character_visual
        ic = ch.input_coord
        sym = vis.symbol
        pr.append(rows - cc.row)
        pc.append(cc.column - 1)
        cp.append(ord(sym[0]) if sym else 32)
        colors = vis.colors
        fgc = colors.fg_color if colors is not None else None
        hx = fgc.rgb_color if fgc is not None else None
        t = hexc.get(hx)
        if t is None:
            t = hexc[hx] = hex_to_rgb(hx, (255, 255, 255)) if hx else (255, 255, 255)
        rgb.append(t)
        inp = ch.animation.input_fg_color
        cs.append(inp is not None and hx is not None and inp.rgb_color == hx)
        ss.append(sym == ch.input_symbol)
        hid.append(ch.character_id in ids)
        sr.append(rows - ic.row)
        sc.append(ic.column - 1)
    s = LiveSnap()
    s.pr, s.pc = np.array(pr, np.int32), np.array(pc, np.int32)
    s.cp = np.array(cp, np.int32)
    s.rgb = np.array(rgb, np.uint8).reshape(-1, 3)
    s.sr, s.sc = np.array(sr, np.int32), np.array(sc, np.int32)
    s.sym_same, s.col_same, s.has_id = np.array(ss, bool), np.array(cs, bool), np.array(hid, bool)
    return s


def run_effect(effect_cls, text: str, rows: int, cols: int, color_handling: str, options: dict,
               seed: int, keep: int, hard_cap: int, capture: str = "text",
               check: Optional[Callable] = None) -> EffectRun:
    """Run a TTE effect headlessly and capture up to ~2*keep frames, uniformly spread over
    the whole animation (stride doubles whenever the buffer fills up).  The final frame is
    always captured as the last item."""
    t0 = time.time()
    state = random.getstate()
    random.seed(seed)
    try:
        effect = effect_cls(text)
        tc = effect.terminal_config
        tc.ignore_terminal_dimensions = True      # we're headless: never clip to 80x24
        tc.canvas_width, tc.canvas_height = cols, rows
        tc.frame_rate = 0                         # never sleep between frames
        tc.wrap_text = False
        tc.existing_color_handling = color_handling
        apply_effect_options(effect.effect_config, options)

        it = iter(effect)
        term = getattr(it, "terminal", None)
        orig = getattr(term, "get_formatted_output_string", None)
        gate = None
        if orig is not None:
            gate = _Gate(orig)
            term.get_formatted_output_string = gate
        ctx: dict = {}

        kept, stride, n, truncated, s = [], 1, 0, False, ""
        while True:
            want = (n % stride == 0)
            if gate is not None:
                gate.want = want and capture == "text"
            try:
                s = next(it)
            except StopIteration:
                break
            if want:
                kept.append((n, s if capture == "text" else _snapshot_live(it, rows, cols, ctx)))
                if len(kept) > 2 * keep:
                    kept = kept[::2]
                    stride *= 2
            n += 1
            if check is not None and (n & 15) == 0:
                check()
            if n >= hard_cap:
                truncated = True
                break
        if gate is not None:
            gate.want = True
        final = orig() if (capture == "text" and orig is not None) else (
            _snapshot_live(it, rows, cols, ctx) if capture != "text" else s)
        kept.append((n, final))
    finally:
        random.setstate(state)
    return EffectRun([k[1] for k in kept], np.array([k[0] for k in kept]), n, truncated, time.time() - t0)


def pick(run: EffectRun, count: int) -> list:
    """Indices into run.items of `count` frames evenly spread over the animation."""
    count = max(1, count)
    if count == 1:
        return [len(run.items) - 1]
    if len(run.items) == 1:
        return [0] * count
    targets = np.linspace(0, run.ticks[-1], count)
    pos = np.searchsorted(run.ticks, targets)
    pos = np.clip(pos, 1, len(run.ticks) - 1)
    left, right = run.ticks[pos - 1], run.ticks[pos]
    pos = np.where(np.abs(targets - left) <= np.abs(right - targets), pos - 1, pos)
    out = pos.tolist()
    out[0], out[-1] = 0, len(run.items) - 1
    return out


# --------------------------------------------------------------------------- #
#  Parameters & the main generate() function
# --------------------------------------------------------------------------- #

PLAYBACK_MODES = ["forward", "reverse", "forward then reverse"]
VIDEO_MODES = ["ascii only (no effect)", "effect intro", "effect outro", "effect intro + outro"]
VIDEO_TIMING = ["keep original length", "fit to duration_seconds"]


@dataclass
class Params:
    effect: str = "decrypt"
    duration_seconds: float = 6.0
    fps: int = 24
    columns: int = 80
    charset: str = "classic"
    custom_charset: str = " .:-=+*#%@"
    color_mode: str = "image + effect colors"
    playback: str = "forward"
    hold_end_seconds: float = 1.0
    video_mode: str = "effect intro"
    video_timing: str = "keep original length"
    video_effect_seconds: float = 2.0
    video_keeps_playing: bool = True
    input_fps: int = 0
    seed: int = 0
    background_color: str = "#000000"
    foreground_color: str = "#00ff41"
    invert: bool = False
    gamma: float = 1.0
    auto_contrast: bool = True
    color_boost: float = 1.25
    font: str = "DejaVuSansMono.ttf"
    font_path: str = ""
    line_spacing: float = 1.0
    output_scale: float = 1.0
    max_side: int = 1280
    effect_options: str = ""
    max_effect_frames: int = 8000


def _even(v: float) -> int:
    return max(16, int(round(v / 2.0)) * 2)


def _palette_from_snap(snap: LiveSnap, rows, cols, default_fg):
    pal = np.empty((rows, cols, 3), np.uint8)
    pal[:] = default_fg
    m = snap.has_id & (snap.sr >= 0) & (snap.sr < rows) & (snap.sc >= 0) & (snap.sc < cols)
    pal[snap.sr[m], snap.sc[m]] = snap.rgb[m]
    return pal


def _compose_live(snap: LiveSnap, live_grid: Grid, rows: int, cols: int):
    """Place every flying character of the effect, but show the *current video frame's*
    glyph / colour for characters whose symbol / colour is still the 'input' one."""
    cp, rgb = snap.cp.copy(), snap.rgb.copy()
    ok = snap.has_id & (snap.sr >= 0) & (snap.sr < rows) & (snap.sc >= 0) & (snap.sc < cols)
    use_sym, use_col = ok & snap.sym_same, ok & snap.col_same
    cp[use_sym] = live_grid.cps[snap.sr[use_sym], snap.sc[use_sym]]
    rgb[use_col] = live_grid.rgb[snap.sr[use_col], snap.sc[use_col]]
    inside = (snap.pr >= 0) & (snap.pr < rows) & (snap.pc >= 0) & (snap.pc < cols)
    out_cp = np.full((rows, cols), 32, np.int32)
    out_fg = np.zeros((rows, cols, 3), np.uint8)
    out_cp[snap.pr[inside], snap.pc[inside]] = cp[inside]       # snapshot is sorted by layer
    out_fg[snap.pr[inside], snap.pc[inside]] = rgb[inside]
    return out_cp, out_fg


def generate(images: np.ndarray, p: Params, progress: Optional[Callable] = None,
             check: Optional[Callable] = None, log: Callable = print):
    """images: (B,H,W,3|4) uint8.  Returns (frames (T,H,W,3) uint8, fps, info str).

    B == 1  -> still image: the effect builds (or destroys) the ASCII picture.
    B  > 1  -> video: every frame becomes ASCII, optionally with an effect intro / outro.
    """
    if images.ndim != 4:
        raise ValueError("Expected an IMAGE batch of shape [B,H,W,C]")
    images = np.ascontiguousarray(images[..., :3])
    B, H_in, W_in = images.shape[:3]

    classes = discover_effects()
    if p.effect not in classes:
        err = tte_import_error()
        if err:
            raise RuntimeError("terminaltexteffects is not installed in ComfyUI's Python environment "
                               "(run: pip install terminaltexteffects). Import error: " + err)
        raise ValueError(f"Unknown effect '{p.effect}'. Available: {sorted(classes)}")
    effect_cls = classes[p.effect]
    options = parse_effect_options(p.effect_options)
    fps = max(1, int(p.fps))

    # ---- output geometry ---------------------------------------------------------
    ow, oh = W_in * p.output_scale, H_in * p.output_scale
    if p.max_side and max(ow, oh) > p.max_side:
        k = p.max_side / max(ow, oh)
        ow, oh = ow * k, oh * k
    ow, oh = _even(ow), _even(oh)
    font_path = resolve_font(p.font, p.font_path)
    size, cw, ch = fit_cell(font_path, max(3, int(round(ow / max(8, p.columns)))), p.line_spacing)
    cols, rows = max(8, ow // cw), max(4, oh // ch)
    rend = Renderer(GlyphAtlas(font_path, size, cw, ch), rows, cols, ow, oh,
                    hex_to_rgb(p.background_color, (0, 0, 0)))
    bg_rgb = tuple(int(v) for v in rend.bg)
    default_fg = hex_to_rgb(p.foreground_color, (0, 255, 65))
    handling = _TTE_COLOR_HANDLING[p.color_mode]
    colored_input = p.color_mode != "effect palette only"
    is_video = B > 1
    log(f"[TTE] effect '{p.effect}': grid {cols}x{rows} chars, cell {cw}x{ch}px, output {ow}x{oh}, "
        + (f"video input ({B} frames)" if is_video else "single image"))

    def run(grid: Grid, keep: int, capture: str = "text") -> EffectRun:
        text = grid_to_ansi(grid, colored_input, placeholders=(capture != "text"))
        last_err = None
        for attempt in range(6):          # some upstream effects crash on certain seeds/sizes -> retry
            try:
                r = run_effect(effect_cls, text, rows, cols, handling, options, p.seed + attempt,
                               keep, p.max_effect_frames, capture, check)
                break
            except EffectOptionError:
                raise
            except (IndexError, ZeroDivisionError, RuntimeError, ValueError) as e:
                last_err = e
                log(f"[TTE]   '{p.effect}' crashed inside terminaltexteffects ({type(e).__name__}: {e}); "
                    f"retrying with seed {p.seed + attempt + 1}")
        else:
            raise RuntimeError(
                f"The '{p.effect}' effect keeps crashing inside terminaltexteffects for this grid size "
                f"({last_err!r}). Try changing 'columns' by a few, or pick another effect.") from last_err
        log(f"[TTE]   {r.total_ticks} effect ticks in {r.seconds:.1f}s"
            + (" (hit max_effect_frames -> truncated)" if r.truncated else ""))
        if r.truncated and capture == "text":      # still end on the finished picture
            r.items[-1] = grid_to_ansi(grid, True)
        return r

    def parse_text(s):
        return parse_frame(s, rows, cols, default_fg, bg_rgb)

    def palette_of(r):
        """Per-cell colours chosen by the effect (only used for 'effect palette only')."""
        if p.color_mode != "effect palette only" or r is None:
            return None
        last = r.items[-1]
        return parse_text(last)[1] if isinstance(last, str) else _palette_from_snap(last, rows, cols, default_fg)

    specs: list = []          # one hashable "spec" per output frame; identical specs are rendered once
    runs: dict = {}
    palette = None
    infos: list = []

    # =============================== still image ===================================
    if not is_video:
        native = p.duration_seconds <= 0
        pingpong = p.playback == "forward then reverse"
        grid0 = build_grid(images[0], rows, cols, p)
        if native:                                  # 1 effect tick == 1 video frame
            hold_n = int(round(p.hold_end_seconds * fps))
            run0 = run(grid0, keep=min(p.max_effect_frames, 2400))
            seq = list(range(len(run0.items)))
        else:
            total = max(4, int(round(p.duration_seconds * fps)))
            hold_n = min(int(round(p.hold_end_seconds * fps)), total - 2)
            n_anim = max(2, total - hold_n)
            half = max(2, n_anim // 2) if pingpong else n_anim
            run0 = run(grid0, keep=max(2 * half, 96))
            seq = pick(run0, half)
        hold = [seq[-1]] * hold_n
        order = {"forward": seq + hold,
                 "reverse": hold + seq[::-1],
                 "forward then reverse": seq + hold + seq[::-1]}[p.playback]
        if not native and len(order) != total:       # rounding fix-up so the length is exact
            order = (order + [order[-1]] * total)[:total]
        runs["a"] = run0
        specs = [("text", "a", i) for i in order]
        palette = palette_of(run0)
        infos.append(f"{run0.total_ticks} effect ticks")

    # ================================= video =======================================
    else:
        src_fps = p.input_fps if p.input_fps > 0 else fps
        want_in = p.video_mode in ("effect intro", "effect intro + outro")
        want_out = p.video_mode in ("effect outro", "effect intro + outro")
        n_eff = int(want_in) + int(want_out)
        natural = p.video_timing != "fit to duration_seconds" or p.duration_seconds <= 0
        eff0 = max(2, int(round(p.video_effect_seconds * fps))) if n_eff else 0
        total = None if natural else max(4, int(round(p.duration_seconds * fps)))

        def plan(live: bool):
            eff = eff0
            body_n = max(2, int(round(B / src_fps * fps))) if natural else (total if live else total - n_eff * eff)
            if live and n_eff:
                eff = min(eff, body_n // n_eff)
            if body_n < 2 or (n_eff and eff < 2):
                raise ValueError("duration_seconds is too short for the requested intro/outro effect(s)")
            if natural:
                idx = np.clip((np.arange(body_n) * src_fps / fps).astype(int), 0, B - 1)
            else:
                idx = np.clip(np.round(np.linspace(0, B - 1, body_n)).astype(int), 0, B - 1)
            return eff, body_n, [int(i) for i in idx]

        live = bool(p.video_keeps_playing) and n_eff > 0
        in_run = out_run = None
        for attempt in ([True, False] if live else [False]):
            eff, body_n, body = plan(attempt)
            cap = "state" if attempt else "text"
            try:
                if want_in:
                    in_run = run(build_grid(images[0], rows, cols, p), max(2 * eff, 96), cap)
                if want_out:
                    out_run = run(build_grid(images[-1], rows, cols, p), max(2 * eff, 96), cap)
                live = attempt
                break
            except (AttributeError, KeyError, TypeError) as e:
                if not attempt:
                    raise
                log(f"[TTE] 'video keeps playing' is not supported by this terminaltexteffects version "
                    f"({e!r}); falling back to frozen first/last frame for the effect.")
        runs["in"], runs["out"] = in_run, out_run
        palette = palette_of(in_run if in_run is not None else out_run)
        in_pick = pick(in_run, eff) if in_run is not None else []
        out_pick = pick(out_run, eff)[::-1] if out_run is not None else []
        if live:
            specs = [("ascii", s) for s in body]
            for k, i in enumerate(in_pick):
                specs[k] = ("live", "in", i, body[k])
            for k, i in enumerate(out_pick):
                j = body_n - len(out_pick) + k
                specs[j] = ("live", "out", i, body[j])
        else:
            specs = ([("text", "in", i) for i in in_pick] + [("ascii", s) for s in body]
                     + [("text", "out", i) for i in out_pick])
        infos.append(f"video mode '{p.video_mode}'" + (" (live)" if live and n_eff else ""))

    # ============================== rasterise ======================================
    T = len(specs)
    out = np.empty((T, oh, ow, 3), np.uint8)
    seen: dict = {}
    grid_cache: dict = {}

    def grid_for(src):
        g = grid_cache.get(src)
        if g is None:
            g = build_grid(images[src], rows, cols, p, palette)
            grid_cache.clear()          # sources are visited in order -> only the latest is needed
            grid_cache[src] = g
        return g

    for t, sp in enumerate(specs):
        if check is not None and (t & 7) == 0:
            check()
        prev = seen.get(sp)
        if prev is not None:
            out[t] = out[prev]
        else:
            if sp[0] == "text":
                cps, fg, cbg = parse_text(runs[sp[1]].items[sp[2]])
                out[t] = rend.render(cps, fg, cbg)
            elif sp[0] == "ascii":
                g = grid_for(sp[1])
                out[t] = rend.render(g.cps, g.rgb)
            else:
                cps, fg = _compose_live(runs[sp[1]].items[sp[2]], grid_for(sp[3]), rows, cols)
                out[t] = rend.render(cps, fg)
            seen[sp] = t
        if progress is not None:
            progress(t + 1, T)

    info = f"{T} frames @ {fps} fps = {T / fps:.2f}s | {ow}x{oh} | grid {cols}x{rows} | " + " | ".join(infos)
    return out, float(fps), info
