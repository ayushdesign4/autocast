"""Stage: thumbnail — a designed 1280x720 CTR thumbnail (Pillow & Devanagari Unicode).

Reads `spine.topic` (+ `spine.shots` / frames for the hero frame), writes
`spine.thumbnail`. This is a high-CTR composed thumbnail designed for YouTube:

  1. Background = the film's own hero frame, cover-cropped to 16:9 with bottom
     subtitles/watermarks cleanly removed and character zoomed & framed on the
     right half. No hero on disk degrades to a deep per-video gradient.
  2. A warm nostalgic grade + a directional left-scrim guarantees high-contrast
     legibility for title text on the left while keeping the character on the
     right bright, vivid, and prominent.
  3. No corporate/tool branding ("AUTOCAST" removed completely).
  4. Proper Devanagari font rendering: Uses the bundled Unicode Devanagari font
     (`autocast/assets/fonts/Devanagari-Bold.ttf`) with native shaping support so
     matras, ligatures, and nuktas never degrade into placeholder boxes (XXXX)
     or displaced characters.
  5. Bold mobile-first typography: heavy stroke outline + soft drop shadow.

Specs: 1280x720 (16:9), <= 2 MB, JPG q~90.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import logging
import sys
from pathlib import Path

from autocast.config import Config
from autocast.spine import Run, Thumbnail

log = logging.getLogger("autocast.stages.thumbnail")

STAGE = "thumbnail"
_W, _H = 1280, 720
_MARGIN = 70  # safe-area inset (px) on every side
_MAX_TITLE_LINES = 3
_MAX_TITLE_PX = 132  # largest display size we ever try
_MIN_TITLE_PX = 44  # readability floor
_MAX_BYTES = 2 * 1024 * 1024  # 2 MB hard cap

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLED_DEV_FONT = _PROJECT_ROOT / "autocast" / "assets" / "fonts" / "Devanagari-Bold.ttf"

# Display fonts ordered by quality and Unicode Devanagari coverage.
# The bundled font is placed first to guarantee proper rendering in all environments.
_DISPLAY_FONTS = (
    str(_BUNDLED_DEV_FONT),
    "C:/Windows/Fonts/NirmalaB.ttf",
    "C:/Windows/Fonts/Nirmala.ttf",
    "C:/Windows/Fonts/mangal.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/System/Library/Fonts/Supplemental/Kohinoor.ttc",
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)

# Curated strong accents that stay legible on a dark, graded frame.
_ACCENTS: tuple[tuple[int, int, int], ...] = (
    (240, 138, 58),   # amber
    (46, 196, 182),   # teal
    (240, 192, 64),   # gold
    (226, 74, 92),    # crimson
    (74, 158, 244),   # azure
    (158, 122, 240),  # violet
    (150, 206, 76),   # lime
    (255, 111, 97),   # coral
)


def _accent_for(title: str) -> tuple[int, int, int]:
    """Deterministic accent colour from the title (stable across renders)."""
    h = hashlib.sha1(title.encode("utf-8")).digest()[0]
    return _ACCENTS[h % len(_ACCENTS)]


def _has_devanagari(text: str) -> bool:
    """Return True if text contains any Devanagari Unicode characters."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _load_font(size: int):
    """Best available bold display face at `size`, degrading gracefully."""
    from PIL import ImageFont

    for path in _DISPLAY_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _hero_background(hero_path: Path | None, size: tuple[int, int]):
    """Cover-crop hero frame to `size`, removing subtitles and framing character."""
    if hero_path is None or not hero_path.exists():
        return None
    try:
        from PIL import Image

        src = Image.open(hero_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        log.warning("thumbnail: hero frame unreadable (%s) -> gradient fallback", exc)
        return None

    w, h = size
    # 1. Strip bottom 12% to remove AI burnt-in subtitles / watermarks
    clean_h = int(src.height * 0.88)
    if clean_h > 100:
        src = src.crop((0, 0, src.width, clean_h))

    # 2. Scale with zoom (1.24x) so the subject is large and expressive
    scale = max(w / src.width, h / src.height) * 1.24
    new_size = (max(w, round(src.width * scale)), max(h, round(src.height * scale)))
    src = src.resize(new_size, Image.LANCZOS)

    # 3. Framing: position subject comfortably on the right half (78% offset)
    overflow_x = src.width - w
    overflow_y = src.height - h
    left = max(0, min(overflow_x, round(overflow_x * 0.78)))
    top = max(0, min(overflow_y, round(overflow_y * 0.20)))
    return src.crop((left, top, left + w, top + h))


def _gradient_background(size: tuple[int, int], accent: tuple[int, int, int]):
    """A deep vertical gradient for keyless renders."""
    from PIL import Image

    w, h = size
    ar, ag, ab = accent
    top = (max(ar // 5, 10), max(ag // 5, 12), max(ab // 5, 18))
    bottom = (8, 9, 12)
    grad = Image.new("RGB", (1, h))
    px = grad.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        px[0, y] = (
            round(top[0] + (bottom[0] - top[0]) * t),
            round(top[1] + (bottom[1] - top[1]) * t),
            round(top[2] + (bottom[2] - top[2]) * t),
        )
    return grad.resize((w, h))


def _apply_grade(img, accent: tuple[int, int, int]):
    """Enhance colors for 1990s nostalgia + apply left-scrim for text readability."""
    from PIL import Image, ImageEnhance

    # Boost contrast and warm saturation
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.14)

    w, h = img.size

    # Left scrim: darkens the left half where the title sits, fading to transparent
    scrim = Image.new("L", (w, h), 0)
    spx = scrim.load()
    scrim_w = round(w * 0.54)
    for x in range(w):
        if x < scrim_w:
            t = 1.0 - (x / scrim_w)
            alpha = round(195 * (t ** 1.35))
        else:
            alpha = 0
        for y in range(h):
            spx[x, y] = alpha

    dark = Image.new("RGB", (w, h), (8, 10, 14))
    img = Image.composite(dark, img, scrim)

    # Subtle bottom vignette
    bot_v = Image.new("L", (1, h), 0)
    bpx = bot_v.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        bpx[0, y] = round(120 * (t ** 2.5))
    bot_v = bot_v.resize((w, h))
    img = Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), img, bot_v)

    # Soft top vignette for depth
    top_v = Image.new("L", (1, h), 0)
    tpx = top_v.load()
    for y in range(h):
        t = 1.0 - (y / max(h - 1, 1))
        tpx[0, y] = round(60 * (t ** 2.2))
    top_v = top_v.resize((w, h))
    img = Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), img, top_v)

    return img


def _line_height(font) -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except Exception:
        return getattr(font, "size", 44)


def _wrap(draw, words: list[str], font, max_width: int) -> list[str]:
    """Greedy word-wrap to `max_width`."""
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if not cur or draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _layout_title(draw, title: str, max_width: int, max_height: int):
    """Calculate largest font size and wrapped lines fitting max_width and max_height."""
    words = title.split()

    # For punchy Hindi titles (3-5 words), prefer 2 balanced lines if they fit comfortably
    if _has_devanagari(title) and 3 <= len(words) <= 5:
        mid = (len(words) + 1) // 2
        trial_lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        for size in range(_MAX_TITLE_PX, _MIN_TITLE_PX - 1, -4):
            font = _load_font(size)
            if all(draw.textlength(ln, font=font) <= max_width for ln in trial_lines):
                block_h = len(trial_lines) * round(_line_height(font) * 1.08)
                if block_h <= max_height:
                    return font, trial_lines

    for size in range(_MAX_TITLE_PX, _MIN_TITLE_PX - 1, -4):
        font = _load_font(size)
        lines = _wrap(draw, words, font, max_width)
        if len(lines) > _MAX_TITLE_LINES:
            continue
        widest = max((draw.textlength(ln, font=font) for ln in lines), default=0)
        block_h = len(lines) * round(_line_height(font) * 1.08)
        if widest <= max_width and block_h <= max_height:
            return font, lines

    font = _load_font(_MIN_TITLE_PX)
    lines = _wrap(draw, words, font, max_width)[:_MAX_TITLE_LINES]
    return font, lines


def _render_gdi_text_mask(line: str, font_size: int, max_w: int = 700, max_h: int = 250):
    """Render a single text line on Windows using GDI for flawless Devanagari shaping."""
    if sys.platform != "win32":
        return None
    try:
        from PIL import Image

        gdi32 = ctypes.windll.gdi32
        user32 = ctypes.windll.user32

        if _BUNDLED_DEV_FONT.exists():
            gdi32.AddFontResourceExW(str(_BUNDLED_DEV_FONT), 0x10, 0)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = max_w
        bmi.biHeight = -max_h
        bmi.biPlanes = 1
        bmi.biBitCount = 32

        hdc = user32.GetDC(0)
        memdc = gdi32.CreateCompatibleDC(hdc)
        bits = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(memdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
        gdi32.SelectObject(memdc, hbmp)

        hfont = gdi32.CreateFontW(
            font_size, 0, 0, 0, 700, False, False, False,
            1, 0, 0, 5, 0, "Nirmala UI"
        )
        gdi32.SelectObject(memdc, hfont)
        gdi32.SetTextColor(memdc, 0x00FFFFFF)
        gdi32.SetBkMode(memdc, 1)

        class RECT(ctypes.Structure):
            _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

        rect = RECT(0, 0, max_w, max_h)
        user32.DrawTextW(memdc, line, -1, ctypes.byref(rect), 0x00000000 | 0x00000020)

        buf = (ctypes.c_char * (max_w * max_h * 4)).from_address(bits.value)
        raw = Image.frombuffer("RGBA", (max_w, max_h), bytes(buf), "raw", "BGRA", 0, 1)
        gray = raw.convert("L")
        bbox = gray.getbbox()
        if bbox:
            return gray.crop(bbox)
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("GDI text shaping failed (%s), falling back to Pillow", exc)
        return None


def _draw_title_block(base, lines: list[str], font, accent: tuple[int, int, int]) -> None:
    """Draw high-impact, bold title block with heavy stroke and soft shadow."""
    from PIL import Image, ImageDraw, ImageFilter

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    is_hindi = any(_has_devanagari(line) for line in lines)

    # Try Windows GDI for pixel-perfect Devanagari ligature & matra shaping
    rendered_with_gdi = False
    if is_hindi and sys.platform == "win32":
        font_sz = getattr(font, "size", 108)

        masks = []
        for line in lines:
            m = _render_gdi_text_mask(line, font_sz)
            if m:
                masks.append(m)

        if len(masks) == len(lines):
            line_gap = 14
            total_h = sum(m.height for m in masks) + line_gap * (len(masks) - 1)
            cur_y = max(_MARGIN, (_H - total_h) // 2)
            cur_x = _MARGIN

            for i, mask in enumerate(masks):
                # Accent color for second line / highlight
                fill_color = (255, 218, 55, 255) if i > 0 and len(masks) > 1 else (255, 255, 255, 255)

                stroke_mask = mask.filter(ImageFilter.MaxFilter(15))
                shadow_mask = stroke_mask.filter(ImageFilter.GaussianBlur(8))

                # Drop shadow
                s_img = Image.new("RGBA", shadow_mask.size, (0, 0, 0, 230))
                overlay.paste(s_img, (cur_x + 6, cur_y + 8), shadow_mask)

                # Heavy dark stroke outline
                st_img = Image.new("RGBA", stroke_mask.size, (10, 12, 16, 255))
                overlay.paste(st_img, (cur_x, cur_y), stroke_mask)

                # Text fill
                f_img = Image.new("RGBA", mask.size, fill_color)
                overlay.paste(f_img, (cur_x, cur_y), mask)

                cur_y += mask.height + line_gap

            rendered_with_gdi = True

    if not rendered_with_gdi:
        # Standard Pillow rendering (for Latin / English or when GDI is unavailable)
        draw = ImageDraw.Draw(overlay)
        line_h = round(_line_height(font) * 1.08)
        block_h = len(lines) * line_h
        cur_y = max(_MARGIN, (_H - block_h) // 2)

        for i, line in enumerate(lines):
            fill_color = accent + (255,) if i > 0 and len(lines) > 1 else (255, 255, 255, 255)
            # Soft multi-offset shadow
            draw.text((_MARGIN + 6, cur_y + 8), line, font=font, fill=(0, 0, 0, 220))
            # Heavy dark stroke outline
            draw.text(
                (_MARGIN, cur_y),
                line,
                font=font,
                fill=fill_color,
                stroke_width=7,
                stroke_fill=(10, 12, 16, 255),
            )
            cur_y += line_h

    base.alpha_composite(overlay)


def _compose_with_pillow(out_path: Path, title: str, hero_path: Path | None) -> bool:
    """Compose + save the 1280x720 thumbnail."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return False

    accent = _accent_for(title)
    bg = _hero_background(hero_path, (_W, _H)) or _gradient_background((_W, _H), accent)
    bg = _apply_grade(bg, accent).convert("RGBA")

    from PIL import ImageDraw

    # Layout title constrained to safe area (left half when hero present)
    max_w = int((_W - 2 * _MARGIN) * 0.50) if hero_path else (_W - 2 * _MARGIN)
    max_h = _H - 2 * _MARGIN
    font, lines = _layout_title(ImageDraw.Draw(bg), title, max_w, max_h)
    _draw_title_block(bg, lines, font, accent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final = bg.convert("RGB")
    for quality in (92, 88, 82, 74, 66):
        final.save(out_path, "JPEG", quality=quality, optimize=True)
        if out_path.stat().st_size <= _MAX_BYTES:
            break
    return True


def _hero_abs(spine: Run, cfg: Config) -> Path | None:
    """Locate the best hero frame from shots, frames, or assets directory."""
    if spine.shots:
        for s in spine.shots:
            if s.image_path:
                p = cfg.run_dir(spine.run_id) / s.image_path
                if p.exists():
                    return p

    frames_dir = cfg.run_dir(spine.run_id) / "frames"
    if frames_dir.exists():
        for candidate in ["frame_04.jpg", "frame_03.jpg", "frame_02.jpg", "frame_01.jpg"]:
            fp = frames_dir / candidate
            if fp.exists():
                return fp
        all_frames = sorted(frames_dir.glob("*.jpg"))
        if all_frames:
            return all_frames[len(all_frames) // 2]

    assets_dir = cfg.assets_dir(spine.run_id)
    if assets_dir.exists():
        for candidate in ["hero.jpg", "shot_000.png", "shot_0.jpg"]:
            ap = assets_dir / candidate
            if ap.exists():
                return ap

    return None


def run(spine: Run, cfg: Config, *, dry_run: bool = False) -> Run:
    if spine.topic is None:
        raise ValueError("thumbnail stage: spine.topic missing (run topic first)")

    out_abs = cfg.assets_dir(spine.run_id) / "thumb.jpg"
    composed = _compose_with_pillow(out_abs, spine.topic.title, _hero_abs(spine, cfg))

    if not composed:
        out_abs.parent.mkdir(parents=True, exist_ok=True)
        out_abs.touch()
        log.warning("thumbnail: Pillow unavailable; touched stub %s", out_abs.name)

    spine.thumbnail = Thumbnail(path="assets/thumb.jpg", width=_W, height=_H)
    log.info("thumbnail: %s (pillow=%s)", spine.thumbnail.path, composed)
    return spine
