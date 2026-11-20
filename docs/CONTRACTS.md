# Cat's Eye - Pipeline Contracts Specification

This document defines the typed data schemas (`studio/contracts.py`) shared across pipeline stages, engines, and storage artifacts in **Cat's Eye**.

All contracts are implemented using Pydantic v2 models and enforce strict validation rules:
- **Non-negative timestamps**: `start >= 0.0`, `end >= 0.0`, `timestamp >= 0.0`
- **Temporal order**: `start < end` for all time intervals
- **Confidence bounds**: `0.0 <= confidence <= 1.0`
- **Crop boundaries**: Required keys `x`, `y`, `w`, `h` with non-negative coordinates and positive width/height
- **Referenced file checks**: Optional/explicit `validate(check_files=True)` verifying local file existence

---

## 1. Scene (`Scene`)
Represents visual scene cuts and boundaries identified during video analysis.

### Schema Fields
- `id` (`str`): Deterministic identifier derived from video fingerprint + start time (e.g. `"sc_a1b2c3d4e5f60708"`).
- `start` (`float`): Start time in source video seconds ($t \ge 0$).
- `end` (`float`): End time in source video seconds ($start < end$).
- `method` (`Literal["cut", "soft"]`): `"cut"` for visual scene change, `"soft"` for silence/time window split.
- `confidence` (`float`): Cut/split confidence ($0.0 \dots 1.0$).
- `layout` (`Optional[str]`): Visual scene layout (`one_face_closeup`, `two_faces`, `facecam_plus_content`, `slides_or_screen`, `wide_scene`, `black_or_transition`).
- `layout_confidence` (`float`): Layout classification confidence ($0.0 \dots 1.0$).
- `facecam_box` (`Optional[list[float]]`): Normalized `[x, y, w, h]` facecam crop box when detected.
- `pending` (`bool`): True when scene sits at the growing tail of a live stream.
- `dominant_color` (`Optional[str]`): Hex color code of the dominant frame palette (e.g. `"#1a1a24"`).
- `motion_score` (`float`): Estimated frame motion / visual energy metric ($0.0 \dots 1.0$).
- `shot_type` (`Optional[str]`): Framing classification (`"wide"`, `"medium"`, `"close"`).

### JSON Example
```json
{
  "id": "scene_001",
  "start": 12.4,
  "end": 18.2,
  "dominant_color": "#282a36",
  "motion_score": 0.74,
  "shot_type": "medium"
}
```

---

## 2. Snapshot (`Snapshot`)
Visual frame extracted for multimodal AI inspection and face tracking.

### Schema Fields
- `id` (`str`): Deterministic snapshot identifier derived from video fingerprint + timestamp (e.g. `"sn_0042..."`).
- `timestamp` (`float`): Source video timestamp in seconds ($t \ge 0$).
- `path` (`str`): File path to analysis JPEG image on disk.
- `thumb_path` (`Optional[str]`): Relative path inside artifact pack to 320px WebP thumbnail.
- `analysis_path` (`Optional[str]`): Relative path inside artifact pack to 640px JPEG analysis frame.
- `width` (`int`): Pixel width ($> 0$).
- `height` (`int`): Pixel height ($> 0$).
- `dhash` (`str`): 64-bit perceptual difference hash (16 hex chars).
- `is_scene_boundary` (`bool`): Whether snapshot anchors a scene start/end boundary.
- `scene_id` (`Optional[str]`): Containing scene ID.
- `brightness` (`Optional[float]`): Mean grayscale luma ($0 \dots 255$).
- `blur` (`Optional[float]`): Raw Laplacian variance sharpness metric.
- `blurriness` (`float`): Normalized blur metric ($0.0 \dots 1.0$).
- `edge_density` (`Optional[float]`): Fraction of high-gradient edge pixels ($0.0 \dots 1.0$).
- `motion` (`Optional[float]`): Mean frame delta vs previous kept snapshot ($0.0 \dots 1.0$).
- `faces` (`Optional[int]`): Detected face count, or `null` when face detector unavailable.
- `face_count` (`Optional[int]`): Legacy alias for face count (or `null` when unavailable).
- `face_boxes` (`Optional[list[list[float]]]`): Detected normalized `[x, y, w, h]` face rectangles.
- `largest_face_box` (`Optional[list[float]]`): Normalized `[x, y, w, h]` of largest detected face.
- `primary_face_bbox` (`Optional[list[float]]`): Primary speaker bounding box `[x, y, w, h]`.
- `caption` (`Optional[dict]`): Optional validated AI contact-sheet caption payload.

### JSON Example
```json
{
  "id": "snap_0042",
  "timestamp": 14.5,
  "path": "/data/artifacts/snapshots/ab/cd1234/frame_0042.jpg",
  "width": 1920,
  "height": 1080,
  "blurriness": 0.08,
  "face_count": 1,
  "primary_face_bbox": [0.42, 0.25, 0.16, 0.28]
}
```

---

## 3. Word (`Word`)
Word-level transcription unit with speech timestamps and model confidence.

### Schema Fields
- `word` (`str`): Transcribed word text (e.g. `"incredible"`).
- `start` (`float`): Word start time in seconds ($t \ge 0$).
- `end` (`float`): Word end time in seconds ($start < end$).
- `confidence` (`float`): ASR model confidence score ($0.0 \le c \le 1.0$).

### JSON Example
```json
{
  "word": "incredible",
  "start": 14.12,
  "end": 14.68,
  "confidence": 0.98
}
```

---

## 4. TranscriptSegment (`TranscriptSegment`)
A continuous sentence or clause chunk of transcribed speech containing word-level timestamps.

### Schema Fields
- `start` (`float`): Segment start time in seconds ($t \ge 0$).
- `end` (`float`): Segment end time in seconds ($start < end$).
- `text` (`str`): Full text string for the phrase.
- `words` (`list[Word]`): Array of constituent `Word` models.

### JSON Example
```json
{
  "start": 12.4,
  "end": 16.8,
  "text": "This is an incredible moment.",
  "words": [
    {"word": "This", "start": 12.4, "end": 12.65, "confidence": 0.99},
    {"word": "is", "start": 12.65, "end": 12.8, "confidence": 0.99},
    {"word": "an", "start": 12.8, "end": 12.95, "confidence": 0.95},
    {"word": "incredible", "start": 12.95, "end": 13.5, "confidence": 0.98},
    {"word": "moment.", "start": 13.5, "end": 14.1, "confidence": 0.97}
  ]
}
```

---

## 5. Moment (`Moment`)
Candidate highlight moment selected for conversion into a short-form clip.

### Schema Fields
- `id` (`str`): Deterministic moment ID (e.g. `"mo_01..."`).
- `start` (`float`): Start time in seconds ($t \ge 0$).
- `end` (`float`): End time in seconds ($start < end$).
- `score` (`float`): Viral / importance score ($0 \dots 100$).
- `reason` (`str`): Justification generated by AI or loudness model.
- `transcript_segment_ids` (`list[str]`): Joined transcript segment IDs inside this half-open `[start, end)` window.
- `snapshot_ids` (`list[str]`): Snapshot IDs whose timestamps fall in `[start, end)`.
- `scene_ids` (`list[str]`): Scene IDs overlapping `[start, end)`.
- `text` (`str`): Combined transcript text for the moment window.
- `pending` (`bool`): True when moment is at a live growing stream tail.
- `hook_text` (`str`): Attention-grabbing hook caption.
- `suggested_title` (`str`): Title recommendation for social posting.

### JSON Example
```json
{
  "id": "moment_01",
  "start": 12.0,
  "end": 42.5,
  "score": 94.2,
  "reason": "High emotional payoff with rapid pacing and immediate hook.",
  "hook_text": "YOU WON'T BELIEVE THIS",
  "suggested_title": "The Secret Nobody Told You"
}
```

---

## 6. Shot (`Shot`)
Individual camera cut or framing angle within a moment for vertical 9:16 reframing.

### Schema Fields
- `id` (`str`): Shot identifier (e.g. `"shot_01"`).
- `start` (`float`): Cut start time relative to source video ($t \ge 0$).
- `end` (`float`): Cut end time relative to source video ($start < end$).
- `crop` (`dict[str, float]`): Normalized or pixel coordinates `{"x": float, "y": float, "w": float, "h": float}`.
- `transition_in` (`Optional[str]`): Transition effect entering shot (`"cut"`, `"fade"`, `"whip"`).
- `transition_out` (`Optional[str]`): Transition effect exiting shot.

### JSON Example
```json
{
  "id": "shot_01",
  "start": 12.0,
  "end": 16.5,
  "crop": {
    "x": 420.0,
    "y": 0.0,
    "w": 607.5,
    "h": 1080.0
  },
  "transition_in": "fade",
  "transition_out": "cut"
}
```

---

## 7. ShotPlan (`ShotPlan`)
Composite multi-camera shot structure composing a full clip.

### Schema Fields
- `moment_id` (`str`): Target moment ID.
- `shots` (`list[Shot]`): Non-empty, contiguous sequence of `Shot` models.
- `audio_track` (`Optional[str]`): Optional secondary audio track (e.g. background music).
- `overlay_track` (`Optional[str]`): Optional visual graphics / logo overlay path.

### JSON Example
```json
{
  "moment_id": "moment_01",
  "shots": [
    {
      "id": "shot_01",
      "start": 12.0,
      "end": 16.5,
      "crop": {"x": 420.0, "y": 0.0, "w": 607.5, "h": 1080.0},
      "transition_in": "cut",
      "transition_out": "cut"
    },
    {
      "id": "shot_02",
      "start": 16.5,
      "end": 22.0,
      "crop": {"x": 550.0, "y": 50.0, "w": 500.0, "h": 888.8},
      "transition_in": "cut",
      "transition_out": "cut"
    }
  ],
  "audio_track": "/data/assets/audio/energetic_beat.mp3",
  "overlay_track": null
}
```

---

## 8. Asset (`Asset`)
Reusable audio/video media asset stored in the creator's asset library.

### Schema Fields
- `id` (`str`): Unique asset identifier (e.g. `"ast_broll_01"`).
- `name` (`str`): Human-readable name.
- `type` (`str`): Allowed types: `"b-roll"`, `"music"`, `"sound_effect"`, `"overlay"`, `"font"`.
- `path` (`str`): Absolute path to file on disk.
- `duration` (`Optional[float]`): Asset length in seconds (for audio/video).
- `tags` (`list[str]`): Search tags (e.g. `["cinematic", "drone", "nature"]`).
- `source` (`str`): Attribution or origin URL.

### JSON Example
```json
{
  "id": "ast_broll_01",
  "name": "Cinematic Forest Aerial",
  "type": "b-roll",
  "path": "/data/assets/broll/forest_aerial.mp4",
  "duration": 14.8,
  "tags": ["nature", "aerial", "calm"],
  "source": "Pexels License"
}
```
