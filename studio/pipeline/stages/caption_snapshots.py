"""CaptionSnapshotsStage (v1): optional multimodal contact-sheet AI captions with strict JSON & ceiling."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from studio import ai, config, usage
from studio.pipeline.stage import Stage, StageContext, StageError, canonical_json

logger = logging.getLogger("studio.pipeline.caption_snapshots")

SHEET_GRID_COLS = 3
SHEET_GRID_ROWS = 3
SHEET_SIZE = SHEET_GRID_COLS * SHEET_GRID_ROWS
CELL_W, CELL_H = 320, 180

CAPTION_SHEET_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "labels": {"type": "ARRAY", "items": {"type": "STRING"}},
            "caption": {"type": "STRING"},
            "people_count": {"type": "INTEGER"},
            "text_on_screen": {"type": "STRING"},
            "emotion": {"type": "STRING"},
            "is_good_thumbnail": {"type": "NUMBER"},
        },
        "required": ["index", "labels", "caption", "people_count", "is_good_thumbnail"],
    },
}


def _sanitize_untrusted(value: Any, max_len: int = 160) -> str:
    """Clean untrusted AI/on-screen text: strip control chars & angle brackets and clamp length."""
    s = re.sub(r"[\x00-\x1f\x7f<>]+", " ", str(value or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def validate_sheet_response(raw: Any, expected_count: int) -> Tuple[Optional[Dict[str, Dict[str, Any]]], str]:
    """Validate and clamp contact-sheet AI JSON output. Returns (entries_by_str_index, error_reason)."""
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    if not isinstance(raw, list) or len(raw) == 0:
        return None, "Response must be a non-empty JSON array"

    entries: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx_val = item.get("index")
        if isinstance(idx_val, bool) or not isinstance(idx_val, (int, float)):
            continue
        idx = int(idx_val)
        if not (0 <= idx < expected_count):
            continue

        raw_labels = item.get("labels")
        if not isinstance(raw_labels, list):
            raw_labels = []
        labels = [t for t in (_sanitize_untrusted(x, max_len=32) for x in raw_labels[:5]) if t]

        cap_words = _sanitize_untrusted(item.get("caption", ""), max_len=160).split()
        caption = " ".join(cap_words[:12])

        try:
            people = int(float(item.get("people_count", 0)))
        except (TypeError, ValueError):
            people = 0
        people = max(0, min(50, people))

        try:
            thumb_score = float(item.get("is_good_thumbnail", 0.0))
        except (TypeError, ValueError):
            thumb_score = 0.0
        thumb_score = round(max(0.0, min(1.0, thumb_score)), 3)

        entry: Dict[str, Any] = {
            "labels": labels,
            "caption": caption,
            "people_count": people,
            "is_good_thumbnail": thumb_score,
        }
        tos = _sanitize_untrusted(item.get("text_on_screen", ""), max_len=160)
        if tos:
            entry["text_on_screen"] = tos
        emo = _sanitize_untrusted(item.get("emotion", ""), max_len=24)
        if emo:
            entry["emotion"] = emo
        entries[str(idx)] = entry

    if not entries:
        return None, "No valid indexed entries matched sheet slots"
    return entries, ""


def build_contact_sheet_b64(
    sheet_snaps: List[Dict[str, Any]],
    snap_root: Optional[Path],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Render up to 9 snapshots into a 3x3 labeled JPEG contact sheet and return (base64_str, actual_snaps)."""
    canvas = Image.new("RGB", (SHEET_GRID_COLS * CELL_W, SHEET_GRID_ROWS * CELL_H), (19, 21, 29))
    draw = ImageDraw.Draw(canvas)
    drawn_snaps: List[Dict[str, Any]] = []

    for i, sn in enumerate(sheet_snaps[:SHEET_SIZE]):
        col = i % SHEET_GRID_COLS
        row = i // SHEET_GRID_COLS
        x0, y0 = col * CELL_W, row * CELL_H

        cand_paths = []
        if snap_root and sn.get("thumb_path"):
            cand_paths.append(snap_root / sn["thumb_path"])
        if snap_root and sn.get("analysis_path"):
            cand_paths.append(snap_root / sn["analysis_path"])
        if sn.get("path"):
            cand_paths.append(Path(sn["path"]))
        img_p = next((p for p in cand_paths if p.is_file()), None)
        if img_p is not None:
            try:
                with Image.open(img_p) as im:
                    im.load()
                    rgb = im.convert("RGB")
                    rgb.thumbnail((CELL_W, CELL_H), Image.Resampling.BILINEAR)
                    px = x0 + (CELL_W - rgb.width) // 2
                    py = y0 + (CELL_H - rgb.height) // 2
                    canvas.paste(rgb, (px, py))
            except Exception:
                pass

        draw.rectangle([x0 + 6, y0 + 6, x0 + 38, y0 + 26], fill=(10, 11, 14), outline=(245, 158, 11))
        draw.text((x0 + 12, y0 + 10), f"[{i}]", fill=(253, 230, 138))
        drawn_snaps.append(sn)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii"), drawn_snaps


class CaptionSnapshotsStage(Stage):
    """Optional contact-sheet visual captioning with request fingerprinting, retry, ceiling, and fail-safe skip."""

    name = "caption_snapshots"
    version = 1
    inputs = ("snapshots_json", "analyzed_snapshots_json")
    outputs = ("captions_json", "captions_status", "captions_message", "requests_used")
    cacheable = True
    progress_unit = "contact_sheet"

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        caps = ai.capabilities()
        return {
            "ai_ready": bool(caps.get("ready")),
            "image_input": bool(caps.get("image_input")),
            "provider": caps.get("provider", ""),
            "model": caps.get("model", ""),
            "max_requests": int(ctx.params.get("snapshot_ai_max_requests", config.SNAPSHOT_AI_MAX_REQUESTS)),
        }

    def _resolve_snap_root(self, ctx: StageContext, snaps_json: Optional[str]) -> Optional[Path]:
        ext_fp = ctx.upstream_fingerprints.get("extract_snapshots")
        if ext_fp:
            man = ctx.store.get(ext_fp, stage="extract_snapshots")
            if man and man.get("_dir"):
                return Path(man["_dir"])
            if not man and ctx.durable_store:
                for rp in (
                    f"users/{ctx.user_id}/artifacts/extract_snapshots/{ext_fp[:2]}/{ext_fp}/{ext_fp}.tar",
                    f"artifacts/extract_snapshots/{ext_fp[:2]}/{ext_fp}/{ext_fp}.tar",
                ):
                    if ctx.durable_store.exists(rp):
                        dest = ctx.store.root / "extract_snapshots" / ext_fp[:2] / ext_fp
                        ctx.durable_store.get_dir_unpacked(rp, dest)
                        return dest
        if snaps_json and Path(snaps_json).is_file():
            return Path(snaps_json).parent
        return None

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.is_cancelled():
            raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

        d = Path(ctx.job["dir"])
        stage_work = d / "visual_work" / "caption_stage"
        stage_work.mkdir(parents=True, exist_ok=True)
        captions_path = stage_work / "captions.json"

        params = self.get_params(ctx)
        max_requests = max(1, int(params["max_requests"]))
        caps = ai.capabilities()

        if not caps["ready"] or not caps["image_input"]:
            reason = (
                "No vision-capable AI provider configured"
                if caps["ready"]
                else "No AI key or model configured"
            )
            payload = {
                "status": "not_available",
                "reason": reason,
                "captions": {},
                "requests_used": 0,
                "max_requests": max_requests,
                "ceiling_hit": False,
                "skipped_sheets": 0,
            }
            captions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return {
                "captions_json": str(captions_path),
                "captions_status": "not_available",
                "captions_message": reason,
                "requests_used": 0,
            }

        src_json = ctx.inputs.get("analyzed_snapshots_json") or ctx.inputs.get("snapshots_json")
        snap_payload = json.loads(Path(src_json).read_text(encoding="utf-8")) if src_json and Path(src_json).is_file() else {}
        snapshots = snap_payload.get("snapshots", [])
        snap_root = self._resolve_snap_root(ctx, ctx.inputs.get("snapshots_json"))

        prov = caps["provider"]
        model = caps["model"]
        captions_map: Dict[str, Dict[str, Any]] = {}
        requests_used = 0
        skipped_sheets = 0
        ceiling_hit = False
        ceiling_msg = ""

        usage.set_context(ctx.job.get("creator"), ctx.job.get("title") or "visual_captions")

        for start_i in range(0, len(snapshots), SHEET_SIZE):
            if ctx.is_cancelled():
                raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

            sheet = snapshots[start_i: start_i + SHEET_SIZE]
            if not sheet:
                continue

            req_fp = hashlib.sha256(
                canonical_json({
                    "v": self.version,
                    "provider": prov,
                    "model": model,
                    "snaps": [(s.get("id"), s.get("dhash", "")) for s in sheet],
                }).encode("utf-8")
            ).hexdigest()[:24]

            chk_key = f"sheet_{req_fp}"
            saved = ctx.resume_state(chk_key)
            if saved and isinstance(saved.get("entries"), dict):
                for idx_str, cap_entry in saved["entries"].items():
                    idx_int = int(idx_str)
                    if 0 <= idx_int < len(sheet):
                        captions_map[sheet[idx_int]["id"]] = cap_entry
                if saved.get("skipped"):
                    skipped_sheets += 1
                continue

            if requests_used >= max_requests:
                ceiling_hit = True
                ceiling_msg = f"AI visual caption request limit reached ({max_requests} requests per video). Remaining snapshots skipped."
                break

            sheet_b64, drawn = build_contact_sheet_b64(sheet, snap_root)
            n_slots = len(drawn)
            prompt = (
                f"Analyze this 3x3 contact sheet containing {n_slots} video frames labeled [0] to [{n_slots - 1}]. "
                "Return strict JSON array with one object per index {index, labels (<=5 short tags), "
                "caption (<=12 words), people_count (int), text_on_screen (optional), emotion (optional), "
                "is_good_thumbnail (0..1 float)}."
            )
            strict_prompt = (
                prompt
                + f" STRICT JSON ONLY. Array length MUST match frames [0..{n_slots - 1}]. "
                "No markdown, no explanation, keys required: index, labels, caption, people_count, is_good_thumbnail."
            )

            valid_entries: Optional[Dict[str, Dict[str, Any]]] = None
            for attempt in range(2):
                if requests_used >= max_requests:
                    ceiling_hit = True
                    ceiling_msg = f"AI visual caption request limit reached ({max_requests} requests per video). Remaining snapshots skipped."
                    break
                cur_prompt = prompt if attempt == 0 else strict_prompt
                try:
                    raw_ans, t_in, t_out = ai.call(
                        model,
                        cur_prompt,
                        CAPTION_SHEET_SCHEMA,
                        images=[{"mime": "image/jpeg", "data": sheet_b64}],
                    )
                    requests_used += 1
                    usage.record(int(t_in or 0), int(t_out or 0), "caption_snapshots")
                    parsed, err = validate_sheet_response(raw_ans, n_slots)
                    if parsed is not None:
                        valid_entries = parsed
                        break
                    logger.warning("Invalid AI caption JSON on sheet %s attempt %d: %s", req_fp, attempt + 1, err)
                except Exception as e:
                    requests_used += 1
                    logger.warning("AI caption call failed on sheet %s attempt %d: %s", req_fp, attempt + 1, e)

            if ceiling_hit and valid_entries is None:
                break

            if valid_entries is None:
                skipped_sheets += 1
                ctx.checkpoint(chk_key, {"sheet_fp": req_fp, "entries": {}, "skipped": True})
                continue

            for idx_str, cap_entry in valid_entries.items():
                idx_int = int(idx_str)
                if 0 <= idx_int < len(sheet):
                    captions_map[sheet[idx_int]["id"]] = cap_entry

            ctx.checkpoint(chk_key, {"sheet_fp": req_fp, "entries": valid_entries, "skipped": False})

        status = "done"
        message = ""
        if ceiling_hit:
            status = "partial"
            message = ceiling_msg
        elif skipped_sheets > 0:
            status = "partial"
            message = f"Skipped {skipped_sheets} sheet(s) due to invalid AI responses."

        payload = {
            "status": status,
            "reason": message,
            "provider": prov,
            "model": model,
            "captions": captions_map,
            "requests_used": requests_used,
            "max_requests": max_requests,
            "ceiling_hit": ceiling_hit,
            "skipped_sheets": skipped_sheets,
        }
        captions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return {
            "captions_json": str(captions_path),
            "captions_status": status,
            "captions_message": message,
            "requests_used": requests_used,
        }
