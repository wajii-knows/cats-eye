"""Turns a time range into a vertical clip: 9:16 layout, bouncy word captions, hook title, cover."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config, roman, timeutil
from .media import NO_WINDOW, run

W, H = config.OUT_W, config.OUT_H

# ASS colours are &HBBGGRR. `size` is the ASS font size (line height); _Font turns it into real pixels.
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
               "strong": "FFFFFF", "outline": 7, "pop": 112, "upper": True, "per": 3},
    "hype": {"label": "Hype", "font": "Anton", "size": 152, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 7, "pop": 114, "upper": True, "per": 2},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 100, "base": "FFFFFF", "active": "6BD7FF",
              "strong": "FFFFFF", "outline": 5, "pop": 106, "upper": False, "per": 4},
}
HOOK = {"font": "Poppins ExtraBold", "size": 96, "ink": "14100E", "box": "FFFFFF", "pad_x": 36, "pad_y": 22, "radius": 26,
        "max_w": 900}
FONT_FILES = {"Poppins ExtraBold": "Poppins-ExtraBold.ttf", "Anton": "Anton-Regular.ttf", "Archivo Black": "ArchivoBlack-Regular.ttf"}
CAP_MAX_W = 960   # captions never run wider than this, whatever is said


class _Font:
    """Measures text exactly as the renderer will draw it, so every word can be placed by hand: the highlighted
    word can grow without pushing its neighbours, and the heading box fits its text."""
    _cache: dict = {}

    def __init__(self, name):
        import struct
        path = Path(__file__).resolve().parent / "fonts" / FONT_FILES.get(name, FONT_FILES["Poppins ExtraBold"])
        d = path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", d[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", d[tables[b"head"] + 18: tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", d[o + 74: o + 78])      # what the renderer sizes the font by
        self.cap = struct.unpack(">h", d[o + 88: o + 90])[0] or int(self.upm * 0.7)
        self.path = str(path)
        self._pil = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def em(self, size):
        return size * self.upm / (self.asc + self.desc)

    def width(self, text, size):
        """Pixels wide at this ASS size."""
        try:
            from PIL import ImageFont
            if "f" not in self._pil:
                self._pil["f"] = ImageFont.truetype(self.path, 200)
            return self._pil["f"].getlength(text) * self.em(size) / 200
        except Exception:  # noqa: BLE001 - no measuring library: a fair estimate keeps clips rendering
            return len(text) * self.em(size) * 0.62

    def cap_shift(self, size):
        """How far below the anchor the middle of a capital letter sits when text is centred on a point."""
        return ((self.asc - self.desc) / 2 - self.cap / 2) * size / (self.asc + self.desc)


CAPTION_LEAD = 0.0  # measured: word stamps are within 0.15 s of the voice, so no shift is needed
def install_fonts():
    """Copy the bundled caption fonts next to the data, so captions look the same on every machine and server."""
    src = Path(__file__).resolve().parent / "fonts"
    for f in src.glob("*.ttf"):
        if not (config.FONTS_DIR / f.name).exists():
            shutil.copyfile(f, config.FONTS_DIR / f.name)


FACECAM_CORNERS = {
    "cam_tl": (0.0, 0.0, 0.28, 0.32),    # Top-Left (Techno Gamerz, xQc)
    "cam_tr": (0.72, 0.0, 0.28, 0.32),   # Top-Right (IShowSpeed, Kai Cenat)
    "cam_bl": (0.0, 0.65, 0.28, 0.35),   # Bottom-Left
    "cam_br": (0.72, 0.65, 0.28, 0.35),  # Bottom-Right
}
FACECAM_LAYOUTS = ("facecam", "cam_tl", "cam_tr", "cam_bl", "cam_br")
COLOR_PUNCH = "eq=brightness=-0.02:contrast=1.06:saturation=1.18"

LAYOUTS = {
    "multishot": "Multishot (dynamic cuts)",
    "facecam": "Streamer cam (auto-detects facecam + gameplay)",
    "blur": "Full frame + blurred fill",
    "zoom": "Zoomed in + blurred fill",
    "crop": "Fill the screen (centre crop)",
}


def resolve_facecam_box(layout: str = "facecam", facecam_box=None):
    """Return normalized (x, y, w, h) from a detected facecam_box or explicit preset, else None."""
    if isinstance(facecam_box, (list, tuple)) and len(facecam_box) == 4:
        try:
            bx, by, bw, bh = [float(v) for v in facecam_box]
            if 0.0 <= bx < 1.0 and 0.0 <= by < 1.0 and 0.05 <= bw <= 0.65 and 0.05 <= bh <= 0.65:
                return (
                    round(max(0.0, min(1.0 - bw, bx)), 4),
                    round(max(0.0, min(1.0 - bh, by)), 4),
                    round(bw, 4),
                    round(bh, 4),
                )
        except (TypeError, ValueError):
            pass
    if layout in FACECAM_CORNERS:
        return FACECAM_CORNERS[layout]
    return None


def _compute_frame_stats(valid_frames: list) -> dict:
    """Compute time-averaged grayscale, gradient, and YCbCr skin probability maps for corner PiP detection."""
    import numpy as np
    fh, fw = valid_frames[0].shape[:2]
    grays, skins = [], []
    for rgb in valid_frames:
        if rgb.shape[:2] != (fh, fw):
            continue
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        skin = (
            (r > 60.0) & (g > 40.0) & (b > 20.0)
            & (r > g) & (r > b)
            & ((r - np.minimum(g, b)) > 12.0)
            & (cb >= 77.0) & (cb <= 127.0)
            & (cr >= 133.0) & (cr <= 173.0)
        )
        grays.append(gray)
        skins.append(skin.astype(np.float32))
    mean_gray = np.mean(grays, axis=0)
    mean_skin = np.mean(skins, axis=0)
    gx = np.abs(mean_gray[:, 2:] - mean_gray[:, :-2])
    gy = np.abs(mean_gray[2:, :] - mean_gray[:-2, :])
    global_edge = float(gx.mean() + gy.mean()) * 0.5 + 1e-3
    center_skin = float(mean_skin[int(fh * 0.35):int(fh * 0.65), int(fw * 0.35):int(fw * 0.65)].mean())
    return {
        "fh": fh, "fw": fw,
        "mean_gray": mean_gray, "mean_skin": mean_skin,
        "gx": gx, "gy": gy, "global_edge": global_edge,
        "center_skin": center_skin,
    }


def _corner_border_stats(stats: dict, is_left: bool, is_top: bool) -> tuple:
    """Measure persistent L-shaped rectangular border strength & coverage in one corner quadrant."""
    import numpy as np
    fh, fw = stats["fh"], stats["fw"]
    gx, gy, global_edge = stats["gx"], stats["gy"], stats["global_edge"]
    best_rank = -1.0
    best_min_b, best_min_cov = 0.0, 0.0
    best_w, best_h = 0.28, (0.32 if is_top else 0.35)
    for w_cand in (0.24, 0.28, 0.32):
        for h_cand in (0.28, 0.32, 0.35):
            vx = int(round((w_cand if is_left else (1.0 - w_cand)) * fw))
            hy = int(round((h_cand if is_top else (1.0 - h_cand)) * fh))
            vx_idx = max(1, min(gx.shape[1] - 2, vx - 1))
            hy_idx = max(1, min(gy.shape[0] - 2, hy - 1))
            y_sl = slice(0, min(gx.shape[0], hy)) if is_top else slice(max(0, hy), gx.shape[0])
            x_sl = slice(0, min(gy.shape[1], vx)) if is_left else slice(max(0, vx), gy.shape[1])
            if y_sl.stop <= y_sl.start or x_sl.stop <= x_sl.start:
                continue
            v_strip = gx[y_sl, max(0, vx_idx - 1):min(gx.shape[1], vx_idx + 2)]
            h_strip = gy[max(0, hy_idx - 1):min(gy.shape[0], hy_idx + 2), x_sl]
            v_row = np.max(v_strip, axis=1)
            h_col = np.max(h_strip, axis=0)
            v_edge = float(np.mean(v_row))
            h_edge = float(np.mean(h_col))
            v_cov = float(np.mean(v_row > 1.2 * global_edge))
            h_cov = float(np.mean(h_col > 1.2 * global_edge))
            min_b = min(v_edge, h_edge) / global_edge
            min_cov = min(v_cov, h_cov)
            rank = 0.6 * min_b + 1.4 * min_cov
            if rank > best_rank:
                best_rank = rank
                best_min_b, best_min_cov = min_b, min_cov
                best_w, best_h = w_cand, h_cand
    return best_min_b, best_min_cov, best_w, best_h


def analyze_framing_from_frames(frames: list) -> dict:
    """Analyze sampled frames to determine vertical framing mode without false-cropping empty corners.
    Returns {'mode': 'facecam'|'speaker_crop'|'wide', 'facecam_box': (x,y,w,h)|None, 'focus_x': float, 'face_count': int}."""
    import numpy as np
    valid = [np.asarray(f, dtype=np.uint8) for f in (frames or []) if f is not None and getattr(f, "ndim", 0) == 3 and f.shape[0] >= 36 and f.shape[1] >= 64]
    if not valid:
        return {"mode": "wide", "facecam_box": None, "focus_x": 0.5, "face_count": 0}

    stats = _compute_frame_stats(valid)

    # 1. Try FaceDetector (OpenCV / registered detector) first
    det_faces_per_frame = []
    det_available = False
    try:
        from .pipeline.stages.analyze_snapshots import get_face_detector
        det = get_face_detector()
        if det and det.available:
            det_available = True
            for rgb in valid:
                det_faces_per_frame.append(list(det.detect(rgb) or []))
    except Exception:  # noqa: BLE001
        det_faces_per_frame = []

    if det_available and det_faces_per_frame:
        quad_obs = {"cam_tl": [], "cam_tr": [], "cam_bl": [], "cam_br": []}
        main_faces = []
        for flist in det_faces_per_frame:
            for b in flist:
                if not isinstance(b, (list, tuple)) or len(b) != 4:
                    continue
                bx, by, bw, bh = [float(v) for v in b]
                area = bw * bh
                cx, cy = bx + 0.5 * bw, by + 0.5 * bh
                q = None
                if 0.002 <= area <= 0.14:
                    if cx < 0.36 and cy < 0.42:
                        q = "cam_tl"
                    elif cx > 0.64 and cy < 0.42:
                        q = "cam_tr"
                    elif cx < 0.36 and cy > 0.58:
                        q = "cam_bl"
                    elif cx > 0.64 and cy > 0.58:
                        q = "cam_br"
                if q:
                    quad_obs[q].append((bx, by, bw, bh, cx, cy))
                else:
                    main_faces.append((bx, by, bw, bh, cx, cy))

        min_hits = max(1, len(valid) // 2)
        best_q = None
        best_q_len = 0
        for q, obs in quad_obs.items():
            if len(obs) >= min_hits:
                cxs, cys = [o[4] for o in obs], [o[5] for o in obs]
                if float(np.std(cxs)) <= 0.06 and float(np.std(cys)) <= 0.06 and len(obs) > best_q_len:
                    best_q = q
                    best_q_len = len(obs)

        if best_q is not None:
            obs = quad_obs[best_q]
            fcx = float(np.median([o[4] for o in obs]))
            fcy = float(np.median([o[5] for o in obs]))
            fbw = float(np.median([o[2] for o in obs]))
            fbh = float(np.median([o[3] for o in obs]))
            is_left = best_q in ("cam_tl", "cam_bl")
            is_top = best_q in ("cam_tl", "cam_tr")
            min_b, min_cov, rw, rh = _corner_border_stats(stats, is_left, is_top)

            # Reject wide multi-person bottom-corner faces that have no PiP border when other people sit beside them
            same_row_mains = [m for m in main_faces if abs(m[5] - fcy) < 0.20]
            irl_multi = (not is_top) and bool(same_row_mains) and not (min_b >= 1.8 and min_cov >= 0.55)
            if not irl_multi:
                rx = 0.0 if is_left else round(1.0 - rw, 4)
                ry = 0.0 if is_top else round(1.0 - rh, 4)
                border_encloses = (rx <= fcx <= rx + rw) and (ry <= fcy <= ry + rh)
                if min_b >= 1.6 and min_cov >= 0.55 and border_encloses:
                    box = (rx, ry, rw, rh)
                else:
                    cw = round(max(0.24, min(0.36, fbw * 2.5)), 4)
                    ch = round(max(0.26, min(0.38, fbh * 2.3)), 4)
                    cx = round(max(0.0, min(1.0 - cw, fcx - 0.5 * cw)), 4)
                    cy = round(max(0.0, min(1.0 - ch, fcy - 0.5 * ch)), 4)
                    box = (cx, cy, cw, ch)
                return {"mode": "facecam", "facecam_box": box, "focus_x": 0.5, "face_count": 1}

        non_empty = [f for f in det_faces_per_frame if f]
        if non_empty:
            med_faces = int(round(float(np.median([len(f) for f in non_empty]))))
            if med_faces == 1:
                focus_x = round(float(np.median([f[0][0] + 0.5 * f[0][2] for f in non_empty])), 4)
                return {"mode": "speaker_crop", "facecam_box": None, "focus_x": focus_x, "face_count": 1}
            if med_faces >= 2:
                return {"mode": "wide", "facecam_box": None, "focus_x": 0.5, "face_count": med_faces}

    # 2. Strict Zero-CV corner webcam detector (requires BOTH continuous L-border AND inner skin face + brightness)
    fh, fw = stats["fh"], stats["fw"]
    mean_gray, mean_skin = stats["mean_gray"], stats["mean_skin"]
    center_skin = stats["center_skin"]

    best_box = None
    best_score = -1e9
    for key, (cx0, cy0, _cw0, _ch0) in FACECAM_CORNERS.items():
        is_left = cx0 < 0.5
        is_top = cy0 < 0.5
        min_b, min_cov, refined_w, refined_h = _corner_border_stats(stats, is_left, is_top)

        b_thresh = 1.5 if is_top else 2.0
        cov_thresh = 0.55 if is_top else 0.70
        if min_b < b_thresh or min_cov < cov_thresh:
            continue

        rx = 0.0 if is_left else round(1.0 - refined_w, 4)
        ry = 0.0 if is_top else round(1.0 - refined_h, 4)
        x0, x1 = int(round(rx * fw)), int(round((rx + refined_w) * fw))
        y0, y1 = int(round(ry * fh)), int(round((ry + refined_h) * fh))
        pw, ph = max(4, x1 - x0), max(4, y1 - y0)

        patch_gray = mean_gray[y0:y1, x0:x1]
        inner_gray = patch_gray[int(ph * 0.18):int(ph * 0.85), int(pw * 0.20):int(pw * 0.80)]
        if not inner_gray.size or float(inner_gray.mean()) < 30.0 or float(inner_gray.std()) < 10.0:
            continue

        patch_skin = mean_skin[y0:y1, x0:x1]
        inner_skin_patch = patch_skin[int(ph * 0.18):int(ph * 0.85), int(pw * 0.20):int(pw * 0.80)]
        inner_skin = float(inner_skin_patch.mean()) if inner_skin_patch.size else 0.0
        outer_skin = max(0.0, (float(patch_skin.mean()) * patch_skin.size - inner_skin * inner_skin_patch.size) / max(1, patch_skin.size - inner_skin_patch.size))
        skin_signal = max(0.0, inner_skin - 0.5 * center_skin) + max(0.0, inner_skin - outer_skin)
        if inner_skin < 0.08 or inner_skin <= outer_skin * 1.1 or skin_signal < 0.06:
            continue

        portrait_signal = min(1.0, float(inner_gray.std()) / 32.0)
        total_score = 3.2 * skin_signal + 1.1 * min(3.0, min_b) + 0.4 * portrait_signal
        if total_score > best_score:
            best_score = total_score
            best_box = (rx, ry, refined_w, refined_h)

    if best_box is not None:
        return {"mode": "facecam", "facecam_box": best_box, "focus_x": 0.5, "face_count": 1}

    return {"mode": "wide", "facecam_box": None, "focus_x": 0.5, "face_count": 0}


def detect_facecam_from_frames(frames: list):
    """Return normalized (x, y, w, h) corner PiP facecam box from sampled RGB frames, or None if no corner facecam exists."""
    return analyze_framing_from_frames(frames).get("facecam_box")


def analyze_clip_framing(src: Path, start: float = 0.0, end: float = 0.0) -> dict:
    """Sample up to 4 640x360 frames across [start, end] and determine clip framing ('facecam', 'speaker_crop', or 'wide')."""
    import numpy as np
    from PIL import Image
    dur = max(0.5, float(end) - float(start))
    offsets = [start + dur * frac for frac in ((0.15, 0.38, 0.62, 0.85) if dur >= 3.0 else (0.5,))]
    frames = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="fc_det_"))
    try:
        for idx, ts in enumerate(offsets):
            out_jpg = tmp_dir / f"f{idx}.jpg"
            p = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, ts):.2f}", "-i", str(src),
                 "-frames:v", "1", "-vf", "scale=640:360", "-q:v", "4", str(out_jpg)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                creationflags=NO_WINDOW, timeout=15,
            )
            if p.returncode == 0 and out_jpg.is_file():
                with Image.open(out_jpg) as im:
                    frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    except Exception:  # noqa: BLE001
        pass
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return analyze_framing_from_frames(frames)


def auto_detect_facecam(src: Path, start: float = 0.0, end: float = 0.0):
    """Return detected corner PiP facecam box (x, y, w, h) for [start, end], or None."""
    return analyze_clip_framing(src, start, end).get("facecam_box")


def _ts(t):
    return timeutil.sec_to_ass(t)


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _groups(words, per):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def geometry(src_w, src_h, layout):
    """Where the video sits in the 1080x1920 frame, and where hook and captions go."""
    if src_h >= src_w and layout not in ("crop", "multishot") and layout not in FACECAM_LAYOUTS:
        layout = "crop"  # already vertical or square: nothing to gain from a blurred fill
    if layout in FACECAM_LAYOUTS:
        top_h = int(round(H * 0.38 / 2)) * 2  # Top 38% (730px) facecam + Bottom 62% (1190px) gameplay
        return layout, top_h, top_h, 1420
    if layout in ("blur", "multishot"):
        fg_h = min(H, int(round(W * src_h / src_w / 2)) * 2) if (src_w and src_h) else int(W * 9 / 16)
    elif layout == "zoom":
        fg_h = 1000
    else:
        fg_h = H
    top = (H - fg_h) // 2
    if layout == "crop":
        return layout, fg_h, 300, 1380
    return layout, fg_h, max(230, top - 190), min(1640, top + fg_h + 180)


def _hook_events(hook, dur, y):
    """The heading: dark text in a white rounded box that fits it, on one line or two balanced ones."""
    f, size = _Font.get(HOOK["font"]), HOOK["size"]
    words = hook.split()
    if not words:
        return []
    lines = [" ".join(words)]
    if f.width(lines[0], size) > HOOK["max_w"] and len(words) > 1:      # two lines, split where they come out most even
        cut = min(range(1, len(words)), key=lambda i: max(f.width(" ".join(words[:i]), size), f.width(" ".join(words[i:]), size)))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    widest = max(f.width(l, size) for l in lines)
    if widest > HOOK["max_w"]:
        size = max(40, int(size * HOOK["max_w"] / widest))
        widest = max(f.width(l, size) for l in lines)
    line_h = f.em(size) * 1.16
    bw, bh = widest + 2 * HOOK["pad_x"], line_h * len(lines) + 2 * HOOK["pad_y"]
    x0, y0, r = (W - bw) / 2, y - bh / 2, min(HOOK["radius"], bh / 2)
    x1, y1 = x0 + bw, y0 + bh
    n = lambda v: f"{v:.0f}"                                                         # noqa: E731
    box = (f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
           f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
           f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
           f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}")
    span = f"{_ts(0)},{_ts(dur)}"
    ev = [f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(160,0)\\c&H{HOOK['box']}&\\p1}}{box}{{\\p0}}"]
    for i, line in enumerate(lines):
        cy = y - line_h * (len(lines) - 1) / 2 + i * line_h - f.cap_shift(size)
        ev.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({W // 2},{cy:.0f})\\fad(160,0)\\fs{size}}}{line}")
    return ev


def build_ass(words, start, end, hook, style_name, highlight, hook_y, cap_y) -> str:
    st = STYLES.get(style_name, STYLES["bounce"])
    words = [dict(w, s=w["s"] - CAPTION_LEAD, e=w["e"] - CAPTION_LEAD) for w in words]
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{st['outline']},0,5,0,0,0,1
Style: Hook,{HOOK['font']},{HOOK['size']},&H00{HOOK['ink']},&H00{HOOK['ink']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = end - start
    ev = _hook_events(_clean(hook), dur, hook_y)
    f = _Font.get(st["font"])
    inside = [dict(w, w=_clean(w["w"])) for w in words if w["s"] >= start - 0.05 and w["s"] < end]
    inside = sorted((w for w in inside if w["w"]), key=lambda w: w["s"])
    for i, w in enumerate(inside[:-1]):
        w["e"] = max(w["s"], min(w["e"], inside[i + 1]["s"]))
    groups = _groups(inside, st["per"])
    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = []
        for x in g:
            t = re.sub(r"[.,;:]+$", "", x["w"])
            texts.append(t.upper() if st["upper"] else t)
        if not all(roman.drawable(t) for t in texts):
            # A script the caption font cannot draw (Japanese, Thai, Chinese...): its words cannot be measured, so
            # they are not placed one by one. The line is drawn whole and centred, in a font the system has for it.
            line = " ".join(texts)
            fs = max(44, min(int(st["size"] * 0.7), int(1500 / max(1, len(line)) * 1.6)))
            ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(g_end)},Cap,,60,60,0,,{{\\an5\\pos({W // 2},{cap_y:.0f})\\q0\\fs{fs}}}{line}")
            continue
        size = st["size"]
        gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
        total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        if total > CAP_MAX_W:                                  # a long phrase: shrink it to fit, never run off the frame
            size = max(36, int(size * CAP_MAX_W / total))
            gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
            total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        cy = cap_y - f.cap_shift(size)
        x = (W - total) / 2
        p = st["pop"]
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = f.width(t, size)
            p = max(104, min(st["pop"], 100 + int(2200 / max(wd, 1))))   # long words grow less, so they never touch their neighbours
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            base = st["strong"] if _key(w["w"]) in strong else st["base"]
            if a - g_start >= 0.02:                                                   # waiting to be said
                ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base}&}}{t}")
            if b - a >= 0.02:                                                         # being said: colour and a quick pop, in front
                ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{{pos}\\c&H{st['active']}&\\fscx94\\fscy94"
                          f"\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,200,\\fscx{p - 5}\\fscy{p - 5})}}{t}")
            if g_end - b >= 0.02:                                                     # said
                ev.append(f"Dialogue: 1,{_ts(b)},{_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base}&\\fscx{p - 5}\\fscy{p - 5}"
                          f"\\t(0,90,\\fscx100\\fscy100)}}{t}")
    return head + "\n".join(ev) + "\n"


def _facecam_stack_chain(in_pad: str, out_pad: str, tag: str, cam_box: tuple) -> str:
    """Crops detected corner facecam into Top 38% (1080x730) and shifts clean gameplay (width 56%) away from the facecam into Bottom 62% (1080x1190)."""
    cx, cy, cw, ch = cam_box
    top_h = int(round(H * 0.38 / 2)) * 2   # 730
    bot_h = H - top_h                      # 1190
    gw = 0.5600
    if cx + cw * 0.5 < 0.5:
        gx = max(0.2200, min(1.0 - gw, round(cx + cw, 4)))
    else:
        gx = max(0.0, min(0.2200, round(cx - gw, 4)))
    return (
        f"{in_pad}split=2[fc_t_{tag}][fc_b_{tag}];"
        f"[fc_t_{tag}]crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{cx:.4f}:ih*{cy:.4f},"
        f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h},setsar=1[top_{tag}];"
        f"[fc_b_{tag}]crop=iw*{gw:.4f}:ih:iw*{gx:.4f}:0,"
        f"scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h},setsar=1[bot_{tag}];"
        f"[top_{tag}][bot_{tag}]vstack,{COLOR_PUNCH},setsar=1{out_pad};"
    )


def _filter_speaker_crop(focus_x: float = 0.5, src_w: int = 1920, src_h: int = 1080) -> str:
    """Comma-free 9:16 crop filter centered horizontally on normalized focus_x (0..1)."""
    sw = max(1, int(src_w or 1920))
    sh = max(1, int(src_h or 1080))
    crop_w_frac = min(1.0, (sh * (W / float(H))) / float(sw))
    if crop_w_frac >= 0.99:
        return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1"
    fx = max(0.0, min(1.0, float(focus_x)))
    crop_x_frac = max(0.0, min(1.0 - crop_w_frac, fx - crop_w_frac * 0.5))
    return f"crop=iw*{crop_w_frac:.4f}:ih:iw*{crop_x_frac:.4f}:0,scale={W}:{H},{COLOR_PUNCH},setsar=1"


def _filter(layout, fg_h, facecam_box=None, focus_x: float = 0.5, src_w: int = 1920, src_h: int = 1080):
    if layout in FACECAM_LAYOUTS:
        cam_box = resolve_facecam_box(layout, facecam_box)
        if cam_box is not None:
            return f"[0:v]fps={config.OUT_FPS}[v_fps];" + _facecam_stack_chain("[v_fps]", "[v0]", "main", cam_box)
        # Never crop a bogus corner when no PiP facecam exists; keep full wide frame + blurred fill
        layout = "blur"
    if layout == "crop":
        if abs(float(focus_x) - 0.5) > 0.02:
            return f"[0:v]fps={config.OUT_FPS}," + _filter_speaker_crop(focus_x, src_w, src_h) + "[v0];"
        return f"[0:v]fps={config.OUT_FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1[v0];"
    return (f"[0:v]fps={config.OUT_FPS},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:{fg_h}:force_original_aspect_ratio=increase,crop={W}:{fg_h},{COLOR_PUNCH}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v0];")


def _multishot_filter(shots, fg_h, dur, src_w, src_h, facecam_box=None, layout="multishot", focus_x: float = 0.5):
    """FFmpeg filter graph to switch camera shots / framings dynamically across the clip."""
    dur = max(0.5, float(dur))
    cam_box = resolve_facecam_box(layout, facecam_box)
    clean = []
    if shots and isinstance(shots, list):
        for s in shots:
            if not isinstance(s, dict):
                continue
            st = max(0.0, min(dur, float(s.get("start", 0.0))))
            en = max(0.0, min(dur, float(s.get("end", dur))))
            typ = str(s.get("type", "wide")).lower()
            if typ not in ("wide", "close", "crop", "zoom", "split"):
                typ = "wide"
            if en - st >= 0.4:
                clean.append({"start": st, "end": en, "type": typ})
    if not clean:
        count = max(2, min(8, int(round(dur / 4.0))))
        step = dur / count
        pats = ["split", "close", "zoom", "split", "crop", "wide"] if cam_box else ["wide", "zoom", "close", "crop", "wide"]
        for i in range(count):
            clean.append({"start": i * step, "end": min(dur, (i + 1) * step), "type": pats[i % len(pats)]})

    clean.sort(key=lambda s: s["start"])
    tiled = []
    cur = 0.0
    for s in clean:
        if s["start"] > cur + 0.1:
            tiled.append({"start": cur, "end": s["start"], "type": "wide"})
        tiled.append(s)
        cur = max(cur, s["end"])
    if cur < dur - 0.1:
        tiled.append({"start": cur, "end": dur, "type": "close" if tiled else "wide"})

    clean = [s for s in tiled if s["end"] - s["start"] >= 0.3][:6]
    if clean:
        clean[-1] = {**clean[-1], "end": dur}
    if not clean:
        return _filter("blur", fg_h, facecam_box=cam_box, focus_x=focus_x, src_w=src_w, src_h=src_h)

    if len(clean) == 1:
        st = clean[0]["type"]
        if st == "split" and cam_box is not None:
            return _filter("facecam", fg_h, facecam_box=cam_box, focus_x=focus_x, src_w=src_w, src_h=src_h)
        if st == "crop":
            return _filter("crop", fg_h, facecam_box=cam_box, focus_x=focus_x, src_w=src_w, src_h=src_h)
        if st == "zoom" or (st == "split" and cam_box is None):
            return _filter("zoom", 1000, facecam_box=cam_box, focus_x=focus_x, src_w=src_w, src_h=src_h)
        return _filter("blur", fg_h, facecam_box=cam_box, focus_x=focus_x, src_w=src_w, src_h=src_h)

    N = len(clean)
    fc = [f"[0:v]fps={config.OUT_FPS},split={N}" + "".join(f"[in{i}]" for i in range(N)) + ";"]
    is_vert = (src_h >= src_w) if (src_w and src_h) else False

    for i, s in enumerate(clean):
        t0, t1, stype = s["start"], s["end"], s["type"]
        trim = f"[in{i}]trim=start={t0:.2f}:end={t1:.2f},setpts=PTS-STARTPTS"
        if is_vert:
            if stype == "close":
                fc.append(f"{trim},scale={int(W*1.3/2)*2}:{int(H*1.3/2)*2}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1[v{i}_p];")
            elif stype == "zoom":
                fc.append(f"{trim},scale={int(W*1.15/2)*2}:{int(H*1.15/2)*2}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1[v{i}_p];")
            else:
                fc.append(f"{trim},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1[v{i}_p];")
        else:
            eff_type = "zoom" if (stype == "split" and cam_box is None) else stype
            if eff_type == "crop":
                if abs(float(focus_x) - 0.5) > 0.02:
                    fc.append(f"{trim}," + _filter_speaker_crop(focus_x, src_w, src_h) + f"[v{i}_p];")
                else:
                    fc.append(f"{trim},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1[v{i}_p];")
            elif eff_type == "close" and cam_box is not None:
                cx, cy, cw, ch = cam_box
                fc.append(f"{trim},crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{cx:.4f}:ih*{cy:.4f},"
                          f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{COLOR_PUNCH},setsar=1[v{i}_p];")
            elif eff_type == "close":
                ch = min(H, int(round(fg_h * 1.36 / 2)) * 2)
                fc.append(f"{trim},split=2[cb{i}][cf{i}];"
                          f"[cb{i}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,scale={W}:{H},eq=brightness=-0.14:saturation=1.15[cbg{i}];"
                          f"[cf{i}]scale={int(W*1.36/2)*2}:{ch}:force_original_aspect_ratio=increase,crop={W}:{ch},{COLOR_PUNCH}[cfg{i}];"
                          f"[cbg{i}][cfg{i}]overlay=0:(H-h)/2,setsar=1[v{i}_p];")
            elif eff_type == "zoom":
                fc.append(f"{trim},split=2[zb{i}][zf{i}];"
                          f"[zb{i}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,scale={W}:{H},eq=brightness=-0.14:saturation=1.15[zbg{i}];"
                          f"[zf{i}]scale={W}:1000:force_original_aspect_ratio=increase,crop={W}:1000,{COLOR_PUNCH}[zfg{i}];"
                          f"[zbg{i}][zfg{i}]overlay=0:(H-h)/2,setsar=1[v{i}_p];")
            elif eff_type == "split":
                fc.append(f"{trim}[tr_{i}];" + _facecam_stack_chain(f"[tr_{i}]", f"[v{i}_p]", f"s{i}", cam_box))
            else:  # wide / blur
                fc.append(f"{trim},split=2[wb{i}][wf{i}];"
                          f"[wb{i}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,scale={W}:{H},eq=brightness=-0.14:saturation=1.15[wbg{i}];"
                          f"[wf{i}]scale={W}:{fg_h}:force_original_aspect_ratio=increase,crop={W}:{fg_h},{COLOR_PUNCH}[wfg{i}];"
                          f"[wbg{i}][wfg{i}]overlay=0:(H-h)/2,setsar=1[v{i}_p];")

    concat_in = "".join(f"[v{i}_p]" for i in range(N))
    fc.append(f"{concat_in}concat=n={N}:v=1:a=0,setsar=1[v0];")
    return "".join(fc)


_qsv_ok = None
RENDER_SECONDS = 600      # one clip's video encode; far beyond normal, so a frozen encoder is stopped, not waited on


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int) -> dict:
    """Writes <name>.mp4, <name>.jpg and <name>.ass into out_dir. Returns file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    req_layout = clip.get("layout", "blur")
    shots = clip.get("shots") or []

    framing = {"mode": "wide", "facecam_box": None, "focus_x": 0.5, "face_count": 0}
    explicit_box = resolve_facecam_box(req_layout, clip.get("facecam_box"))
    if clip.get("facecam_box") and explicit_box is not None:
        framing = {"mode": "facecam", "facecam_box": explicit_box, "focus_x": 0.5, "face_count": 1}
    elif req_layout == "facecam" or req_layout == "multishot":
        framing = analyze_clip_framing(src, start, end)
        if framing.get("facecam_box") is not None:
            clip["facecam_box"] = list(framing["facecam_box"])

    effective_layout = req_layout
    if req_layout == "facecam":
        if framing["mode"] == "facecam" and framing["facecam_box"] is not None:
            effective_layout = "facecam"
        elif framing["mode"] == "speaker_crop":
            effective_layout = "crop"
        else:
            effective_layout = "blur"

    _, fg_h, hook_y, cap_y = geometry(src_w, src_h, effective_layout)
    ass = build_ass(words, start, end, clip.get("hook", ""), clip.get("style", "bounce"),
                    clip.get("highlight_words", []), hook_y, cap_y)
    work = Path(tempfile.mkdtemp(prefix="pk"))
    try:
        res = _render_in(
            work, src, out_dir, name, start, end, effective_layout, fg_h, ass,
            shots=shots, src_w=src_w, src_h=src_h,
            facecam_box=framing.get("facecam_box") or explicit_box,
            focus_x=framing.get("focus_x", 0.5),
            req_layout=req_layout, framing_mode=framing["mode"],
        )
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_in(work: Path, src: Path, out_dir: Path, name: str, start, end, layout, fg_h, ass: str,
               shots: list = None, src_w: int = 0, src_h: int = 0, facecam_box=None,
               focus_x: float = 0.5, req_layout: str = "", framing_mode: str = "") -> dict:
    global _qsv_ok
    (work / f"{name}.ass").write_text(ass, encoding="utf-8")
    (work / "f").mkdir()
    for f in config.FONTS_DIR.glob("*.ttf"):
        shutil.copyfile(f, work / "f" / f.name)
    dur = end - start
    if layout == "multishot":
        rel_shots = []
        for s in (shots or []):
            if isinstance(s, dict):
                rel_shots.append({"start": float(s.get("start", start)) - start,
                                  "end": float(s.get("end", end)) - start,
                                  "type": s.get("type", "wide")})
        fc = _multishot_filter(rel_shots, fg_h, dur, src_w, src_h, facecam_box=facecam_box, layout=layout, focus_x=focus_x) + f"[v0]ass={name}.ass:fontsdir=f[v]"
    else:
        fc = _filter(layout, fg_h, facecam_box=facecam_box, focus_x=focus_x, src_w=src_w or 1920, src_h=src_h or 1080) + f"[v0]ass={name}.ass:fontsdir=f[v]"
    base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
            "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
    encoders = []
    if _qsv_ok is not False:
        encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-global_quality", "23", "-preset", "fast"]))
    encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast"]))
    err = None
    for label, enc in encoders:
        p = subprocess.run(base + enc + [f"{name}.mp4"], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=RENDER_SECONDS)
        if p.returncode == 0:
            if label == "qsv":
                _qsv_ok = True
            break
        err = p.stderr[-1200:]
        if label == "qsv":
            _qsv_ok = False
    else:
        raise RuntimeError(f"Render failed: {err}")
    cover_t = min(1.2, (end - start) / 2)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
         "-q:v", "3", f"{name}.jpg"], cwd=work)
    for ext in ("mp4", "jpg", "ass"):
        shutil.move(str(work / f"{name}.{ext}"), str(out_dir / f"{name}.{ext}"))
    out = {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": req_layout or layout}
    if framing_mode:
        out["framing"] = framing_mode
    return out
