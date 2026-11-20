"""AnalyzeSnapshotsStage (v1): local no-AI frame metrics, optional face detection, scene layout & facecam box."""
from __future__ import annotations

import base64
import json
import logging
import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from studio import config
from studio.contracts import Scene
from studio.pipeline.stage import Stage, StageContext, StageError
from studio.pipeline.visual_index import VisualTimelineIndex

logger = logging.getLogger("studio.pipeline.analyze_snapshots")

LAYOUT_CLASSES = (
    "one_face_closeup",
    "two_faces",
    "facecam_plus_content",
    "slides_or_screen",
    "wide_scene",
    "black_or_transition",
)


def _strip_text(val: Any) -> Any:
    return val


class FaceDetector(ABC):
    """Pluggable face detector extension point returning normalized [x, y, w, h] boxes or None."""

    name: str = "base"
    available: bool = False

    @abstractmethod
    def detect(self, rgb_array: np.ndarray) -> Optional[List[List[float]]]:
        """Return list of [x, y, w, h] in 0..1 coordinates, or None when detector unavailable."""


class NullFaceDetector(FaceDetector):
    """Default zero-dependency detector when no face model/library is installed (returns None)."""

    name = "none"
    available = False

    def detect(self, rgb_array: np.ndarray) -> Optional[List[List[float]]]:
        return None


class OpenCVHaarFaceDetector(FaceDetector):
    """Optional Apache-2.0 Haar cascade detector used only when OpenCV is already installed."""

    name = "opencv_haar"

    def __init__(self) -> None:
        self._cascade = None
        self.available = False
        try:
            import cv2  # type: ignore
            xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if xml.is_file():
                clf = cv2.CascadeClassifier(str(xml))
                if not clf.empty():
                    self._cascade = (cv2, clf)
                    self.available = True
        except Exception:
            self._cascade = None
            self.available = False

    def detect(self, rgb_array: np.ndarray) -> Optional[List[List[float]]]:
        if not self.available or self._cascade is None:
            return None
        cv2, clf = self._cascade
        h, w = rgb_array.shape[:2]
        if h < 16 or w < 16:
            return []
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        rects = clf.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20))
        boxes: List[List[float]] = []
        for x, y, bw, bh in rects:
            boxes.append([
                round(max(0.0, min(1.0, float(x) / w)), 4),
                round(max(0.0, min(1.0, float(y) / h)), 4),
                round(max(0.0, min(1.0, float(bw) / w)), 4),
                round(max(0.0, min(1.0, float(bh) / h)), 4),
            ])
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes


class LayoutClassifier(ABC):
    """Pluggable scene layout classifier (rules now; ML/multimodal model later)."""

    @abstractmethod
    def classify(self, scene_snaps: List[Dict[str, Any]], scene: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """Return (layout_label, confidence) for a scene."""


class RuleBasedLayoutClassifier(LayoutClassifier):
    """Explainable rule-based classifier over local brightness, blur, edges, motion & faces."""

    def classify(self, scene_snaps: List[Dict[str, Any]], scene: Dict[str, Any]) -> Tuple[Optional[str], float]:
        if not scene_snaps:
            return None, 0.0

        dur = max(0.01, float(scene.get("end", 0.0)) - float(scene.get("start", 0.0)))
        brights = [float(s["brightness"]) for s in scene_snaps if s.get("brightness") is not None]
        edges = [float(s["edge_density"]) for s in scene_snaps if s.get("edge_density") is not None]
        motions = [float(s["motion"]) for s in scene_snaps if s.get("motion") is not None]
        face_counts = [int(s["faces"]) for s in scene_snaps if s.get("faces") is not None]

        mean_b = sum(brights) / len(brights) if brights else 128.0
        mean_e = sum(edges) / len(edges) if edges else 0.0
        mean_m = sum(motions) / len(motions) if motions else 0.0

        if mean_b < 16.0 or (dur < 1.8 and mean_m > 0.35 and (not face_counts or max(face_counts) == 0)):
            return "black_or_transition", round(0.88 if mean_b < 16.0 else 0.72, 3)

        if face_counts:
            med_faces = float(statistics.median(face_counts))
            l_boxes = [s["largest_face_box"] for s in scene_snaps if isinstance(s.get("largest_face_box"), list) and len(s["largest_face_box"]) == 4]
            corner_hits = 0
            areas = []
            for bx, by, bw, bh in l_boxes:
                area = bw * bh
                areas.append(area)
                cx, cy = bx + 0.5 * bw, by + 0.5 * bh
                if area <= 0.16 and (cx < 0.35 or cx > 0.65) and (cy < 0.38 or cy > 0.62):
                    corner_hits += 1

            if l_boxes and corner_hits >= max(1, len(l_boxes) // 2) and mean_e >= 0.05:
                return "facecam_plus_content", 0.85

            if med_faces >= 1.5:
                return "two_faces", 0.80

            if med_faces >= 0.5:
                med_area = float(statistics.median(areas)) if areas else 0.0
                if med_area >= 0.08:
                    return "one_face_closeup", 0.85
                if mean_e >= 0.06:
                    return "facecam_plus_content" if corner_hits > 0 else "wide_scene", 0.70
                return "wide_scene", 0.72

        if mean_e >= 0.06:
            return "slides_or_screen", round(min(0.95, 0.58 + mean_e * 2.0), 3)

        return "wide_scene", 0.62


_CUSTOM_FACE_DETECTOR: Optional[FaceDetector] = None
_DEFAULT_FACE_DETECTOR: Optional[FaceDetector] = None
_CUSTOM_LAYOUT_CLASSIFIER: Optional[LayoutClassifier] = None


def register_face_detector(detector: Optional[FaceDetector]) -> None:
    global _CUSTOM_FACE_DETECTOR
    _CUSTOM_FACE_DETECTOR = detector


def get_face_detector() -> FaceDetector:
    global _DEFAULT_FACE_DETECTOR
    if _CUSTOM_FACE_DETECTOR is not None:
        return _CUSTOM_FACE_DETECTOR
    if _DEFAULT_FACE_DETECTOR is None:
        ocv = OpenCVHaarFaceDetector()
        _DEFAULT_FACE_DETECTOR = ocv if ocv.available else NullFaceDetector()
    return _DEFAULT_FACE_DETECTOR


def register_layout_classifier(classifier: Optional[LayoutClassifier]) -> None:
    global _CUSTOM_LAYOUT_CLASSIFIER
    _CUSTOM_LAYOUT_CLASSIFIER = classifier


def get_layout_classifier() -> LayoutClassifier:
    return _CUSTOM_LAYOUT_CLASSIFIER or RuleBasedLayoutClassifier()


def compute_frame_metrics(
    img_path: Path,
    prev_tiny: Optional[np.ndarray] = None,
    detector: Optional[FaceDetector] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute local brightness, Laplacian blur, edge density, motion & optional faces."""
    with Image.open(img_path) as im:
        im.load()
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        gray_im = im.convert("L")
        gray = np.asarray(gray_im, dtype=np.float32)
        tiny = np.asarray(gray_im.resize((64, 36), Image.Resampling.BILINEAR), dtype=np.uint8)

    brightness = round(float(gray.mean()), 2)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * gray[1:-1, 1:-1]
        )
        blur = round(float(lap.var()), 2)
        gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
        gy = np.abs(gray[2:, 1:-1] - gray[:-2, 1:-1])
        mag = 0.5 * (gx + gy)
        edge_density = round(float((mag > 25.0).mean()), 4)
    else:
        blur = 0.0
        edge_density = 0.0

    blurriness = round(max(0.0, 1.0 - min(blur, 500.0) / 500.0), 4)
    r_m, g_m, b_m = [int(round(float(rgb[:, :, c].mean()))) for c in range(3)]
    dominant_color = f"#{r_m:02x}{g_m:02x}{b_m:02x}"

    if prev_tiny is not None and prev_tiny.shape == tiny.shape:
        motion = round(float(np.abs(tiny.astype(np.float32) - prev_tiny.astype(np.float32)).mean() / 255.0), 4)
    else:
        motion = 0.0

    det = detector or get_face_detector()
    boxes = det.detect(rgb)
    if boxes is None:
        faces = None
        face_count = None
        largest_box = None
    else:
        faces = len(boxes)
        face_count = len(boxes)
        largest_box = max(boxes, key=lambda b: b[2] * b[3]) if boxes else None

    return {
        "brightness": brightness,
        "blur": blur,
        "blurriness": blurriness,
        "edge_density": edge_density,
        "dominant_color": dominant_color,
        "motion": motion,
        "faces": faces,
        "face_count": face_count,
        "face_boxes": boxes,
        "largest_face_box": largest_box,
        "primary_face_bbox": largest_box,
    }, tiny


def detect_facecam_box(
    snapshots: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
) -> Optional[List[float]]:
    """Detect a stable small corner facecam box across scenes, returning padded normalized [x, y, w, h] or None."""
    corner_obs: List[Tuple[str, float, float, List[float]]] = []
    for sn in snapshots:
        boxes = sn.get("face_boxes") or ([] if sn.get("largest_face_box") is None else [sn["largest_face_box"]])
        for b in boxes:
            if not isinstance(b, list) or len(b) != 4:
                continue
            bx, by, bw, bh = [float(v) for v in b]
            area = bw * bh
            cx, cy = bx + 0.5 * bw, by + 0.5 * bh
            if 0.002 <= area <= 0.16 and (cx < 0.35 or cx > 0.65) and (cy < 0.38 or cy > 0.62):
                corner_obs.append((str(sn.get("scene_id") or ""), cx, cy, [bx, by, bw, bh]))

    if len(corner_obs) < 3:
        return None
    distinct_scenes = {x[0] for x in corner_obs if x[0]}
    if len(scenes) >= 2 and len(distinct_scenes) < 2 and len(corner_obs) < max(4, int(len(snapshots) * 0.4)):
        return None

    cxs = [x[1] for x in corner_obs]
    cys = [x[2] for x in corner_obs]
    if float(np.std(cxs)) > 0.08 or float(np.std(cys)) > 0.08:
        return None

    bx = float(statistics.median(x[3][0] for x in corner_obs))
    by = float(statistics.median(x[3][1] for x in corner_obs))
    bw = float(statistics.median(x[3][2] for x in corner_obs))
    bh = float(statistics.median(x[3][3] for x in corner_obs))
    pad_x = max(0.015, bw * 0.15)
    pad_y = max(0.015, bh * 0.15)
    x0 = max(0.0, bx - pad_x)
    y0 = max(0.0, by - pad_y)
    x1 = min(1.0, bx + bw + pad_x)
    y1 = min(1.0, by + bh + pad_y)
    return [round(x0, 4), round(y0, 4), round(max(0.01, x1 - x0), 4), round(max(0.01, y1 - y0), 4)]


class AnalyzeSnapshotsStage(Stage):
    """Batched local no-AI frame metrics, scene layout classification, and facecam box detection."""

    name = "analyze_snapshots"
    version = 1
    inputs = ("scenes_json", "snapshots_json")
    outputs = ("analyzed_snapshots_json", "analyzed_scenes_json", "visual_summary_json", "facecam_box")
    cacheable = True
    progress_unit = "batch_64"

    def get_params(self, ctx: StageContext) -> Dict[str, Any]:
        return {
            "batch_size": min(64, max(1, int(ctx.params.get("snapshot_batch_size", config.SNAPSHOT_BATCH_SIZE)))),
            "detector": get_face_detector().name,
        }

    def _resolve_snapshot_dir(self, ctx: StageContext, snaps_json_path: Optional[str]) -> Optional[Path]:
        art_dir = ctx.inputs.get("_artifact_dir")
        if art_dir and Path(art_dir).is_dir():
            return Path(art_dir)
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
        if snaps_json_path and Path(snaps_json_path).is_file():
            return Path(snaps_json_path).parent
        return None

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.is_cancelled():
            raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

        batch_size = int(self.get_params(ctx)["batch_size"])
        d = Path(ctx.job["dir"])
        stage_work = d / "visual_work" / "analyze_stage"
        stage_work.mkdir(parents=True, exist_ok=True)

        scenes_json = ctx.inputs.get("scenes_json")
        snaps_json = ctx.inputs.get("snapshots_json")
        scenes_payload = json.loads(Path(scenes_json).read_text(encoding="utf-8")) if scenes_json and Path(scenes_json).is_file() else {}
        snaps_payload = json.loads(Path(snaps_json).read_text(encoding="utf-8")) if snaps_json and Path(snaps_json).is_file() else {}

        scenes = [dict(s) for s in scenes_payload.get("scenes", [])]
        snapshots = [dict(s) for s in snaps_payload.get("snapshots", [])]
        snap_root = self._resolve_snapshot_dir(ctx, snaps_json)

        detector = get_face_detector()
        classifier = get_layout_classifier()

        analyzed_snaps: List[Dict[str, Any]] = []
        prev_tiny: Optional[np.ndarray] = None
        num_batches = (len(snapshots) + batch_size - 1) // batch_size if snapshots else 0

        for b in range(num_batches):
            if ctx.is_cancelled():
                raise StageError("CANCELLED", "Cancelled by user", retryable=False, detail={"cancelled": True})

            batch = snapshots[b * batch_size: (b + 1) * batch_size]
            chk_key = f"batch_{b:04d}"
            saved = ctx.resume_state(chk_key)
            if saved and saved.get("detector") == detector.name and isinstance(saved.get("snapshots"), list) and len(saved["snapshots"]) == len(batch):
                analyzed_snaps.extend(saved["snapshots"])
                if saved.get("last_tiny_b64"):
                    prev_tiny = np.frombuffer(base64.b64decode(saved["last_tiny_b64"]), dtype=np.uint8).reshape((36, 64))
                continue

            batch_out: List[Dict[str, Any]] = []
            for sn in batch:
                an_rel = sn.get("analysis_path")
                cand_paths = []
                if snap_root and an_rel:
                    cand_paths.append(snap_root / an_rel)
                if snap_root and sn.get("thumb_path"):
                    cand_paths.append(snap_root / sn["thumb_path"])
                if sn.get("path"):
                    cand_paths.append(Path(sn["path"]))
                img_path = next((p for p in cand_paths if p.is_file()), None)

                item = dict(sn)
                if img_path is not None:
                    try:
                        metrics, prev_tiny = compute_frame_metrics(img_path, prev_tiny=prev_tiny, detector=detector)
                        item.update(metrics)
                    except Exception as e:
                        logger.warning("Failed analyzing frame %s: %s", sn.get("id"), e)
                batch_out.append(item)

            prev_b64 = base64.b64encode(prev_tiny.tobytes()).decode("ascii") if prev_tiny is not None else None
            ctx.checkpoint(chk_key, {
                "batch_idx": b,
                "detector": detector.name,
                "snapshots": batch_out,
                "last_tiny_b64": prev_b64,
            })
            analyzed_snaps.extend(batch_out)

        facecam_box = detect_facecam_box(analyzed_snaps, scenes)
        tl = VisualTimelineIndex(scenes=scenes, snapshots=analyzed_snaps)

        analyzed_scenes: List[Dict[str, Any]] = []
        for sc in scenes:
            sc_copy = dict(sc)
            sc_snaps = tl.snapshots_in(float(sc["start"]), float(sc["end"]))
            layout, conf = classifier.classify(sc_snaps, sc_copy)
            motions = [float(s["motion"]) for s in sc_snaps if s.get("motion") is not None]
            colors = [s["dominant_color"] for s in sc_snaps if s.get("dominant_color")]
            sc_copy["layout"] = layout
            sc_copy["layout_confidence"] = round(float(conf), 4)
            if motions:
                sc_copy["motion_score"] = round(sum(motions) / len(motions), 4)
            if colors:
                sc_copy["dominant_color"] = colors[len(colors) // 2]
            if layout == "facecam_plus_content" and facecam_box is not None:
                sc_copy["facecam_box"] = facecam_box
            Scene(**sc_copy).validate(check_files=False)
            analyzed_scenes.append(sc_copy)

        an_snaps_path = stage_work / "analyzed_snapshots.json"
        an_scenes_path = stage_work / "analyzed_scenes.json"
        summary_path = stage_work / "visual_summary.json"

        an_snaps_path.write_text(
            json.dumps({
                "video_fingerprint": snaps_payload.get("video_fingerprint") or scenes_payload.get("video_fingerprint", ""),
                "duration": snaps_payload.get("duration") or scenes_payload.get("duration", 0.0),
                "has_video": snaps_payload.get("has_video", scenes_payload.get("has_video", True)),
                "pending": snaps_payload.get("pending", scenes_payload.get("pending", False)),
                "pending_from": snaps_payload.get("pending_from", scenes_payload.get("pending_from")),
                "face_detector": detector.name,
                "face_detection_available": detector.available,
                "snapshots": analyzed_snaps,
            }, indent=2),
            encoding="utf-8",
        )
        an_scenes_path.write_text(
            json.dumps({
                "video_fingerprint": scenes_payload.get("video_fingerprint", ""),
                "duration": scenes_payload.get("duration", 0.0),
                "has_video": scenes_payload.get("has_video", True),
                "pending": scenes_payload.get("pending", False),
                "pending_from": scenes_payload.get("pending_from"),
                "facecam_box": facecam_box,
                "scenes": analyzed_scenes,
            }, indent=2),
            encoding="utf-8",
        )
        summary_payload = {
            "face_detector": detector.name,
            "face_detection_available": detector.available,
            "facecam_box": facecam_box,
            "snapshot_count": len(analyzed_snaps),
            "scene_count": len(analyzed_scenes),
            "layouts": {s["id"]: s.get("layout") for s in analyzed_scenes},
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        return {
            "analyzed_snapshots_json": str(an_snaps_path),
            "analyzed_scenes_json": str(an_scenes_path),
            "visual_summary_json": str(summary_path),
            "facecam_box": facecam_box,
            "face_detection_available": detector.available,
        }
