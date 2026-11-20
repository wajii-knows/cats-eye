"""Typed contracts for data shared between pipeline stages and engines in Cat's Eye.

All models use Pydantic v2 and enforce strict validation:
- Non-negative timestamps (start >= 0, end >= 0)
- Temporal consistency (start < end)
- Confidence bounded in [0.0, 1.0]
- Bounded crop coordinates
- Explicit .validate() method supporting referenced file existence checks
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def make_scene_id(video_fp: str, start: float) -> str:
    """Deterministic Scene ID derived from video fingerprint + start timestamp."""
    raw = f"{video_fp}:scene:{round(float(start), 3):.3f}".encode("utf-8")
    return "sc_" + hashlib.sha256(raw).hexdigest()[:16]


def make_snapshot_id(video_fp: str, timestamp: float) -> str:
    """Deterministic Snapshot ID derived from video fingerprint + timestamp."""
    raw = f"{video_fp}:snap:{round(float(timestamp), 3):.3f}".encode("utf-8")
    return "sn_" + hashlib.sha256(raw).hexdigest()[:16]


def make_segment_id(video_fp: str, start: float, end: float) -> str:
    """Deterministic TranscriptSegment ID derived from video fingerprint + span."""
    raw = f"{video_fp}:seg:{round(float(start), 3):.3f}:{round(float(end), 3):.3f}".encode("utf-8")
    return "seg_" + hashlib.sha256(raw).hexdigest()[:16]


def make_moment_id(video_fp: str, start: float, end: float) -> str:
    """Deterministic Moment ID derived from video fingerprint + span."""
    raw = f"{video_fp}:moment:{round(float(start), 3):.3f}:{round(float(end), 3):.3f}".encode("utf-8")
    return "mo_" + hashlib.sha256(raw).hexdigest()[:16]


class ContractModel(BaseModel):
    """Base model providing explicit .validate() method."""

    def validate_contract(self, check_files: bool = True) -> "ContractModel":
        """Explicit validation method. Enforces contract rules and optional file existence."""
        return self


class Scene(ContractModel):
    """Visual scene boundary identified during video analysis."""

    id: str
    start: float
    end: float
    dominant_color: Optional[str] = None
    motion_score: float = 0.0
    shot_type: Optional[str] = None
    method: Literal["cut", "soft"] = "cut"
    confidence: float = 1.0
    layout: Optional[str] = None
    layout_confidence: float = 0.0
    facecam_box: Optional[List[float]] = None
    pending: bool = False

    @model_validator(mode="after")
    def _validate_timing(self) -> "Scene":
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError(f"Timestamps must be non-negative: start={self.start}, end={self.end}")
        if self.start >= self.end:
            raise ValueError(f"start must be strictly less than end: {self.start} >= {self.end}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0]: {self.confidence}")
        if not (0.0 <= self.layout_confidence <= 1.0):
            raise ValueError(f"layout_confidence must be in [0.0, 1.0]: {self.layout_confidence}")
        if self.facecam_box is not None:
            if len(self.facecam_box) != 4 or any(v < 0.0 or v > 1.0 for v in self.facecam_box):
                raise ValueError(f"facecam_box must be normalized [x, y, w, h] in [0, 1]: {self.facecam_box}")
        return self

    def validate(self, check_files: bool = True) -> "Scene":
        self._validate_timing()
        return self


class Snapshot(ContractModel):
    """Visual frame snapshot extracted for multimodal analysis."""

    id: str
    timestamp: float
    path: str
    width: int
    height: int
    blurriness: float = 0.0
    face_count: Optional[int] = 0
    faces: Optional[int] = None
    primary_face_bbox: Optional[List[float]] = None  # [x, y, w, h] normalized 0..1 or pixels
    thumb_path: Optional[str] = None
    analysis_path: Optional[str] = None
    dhash: str = ""
    is_scene_boundary: bool = False
    scene_id: Optional[str] = None
    brightness: Optional[float] = None
    blur: Optional[float] = None
    edge_density: Optional[float] = None
    motion: Optional[float] = None
    face_boxes: Optional[List[List[float]]] = None
    largest_face_box: Optional[List[float]] = None
    caption: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "Snapshot":
        if self.timestamp < 0.0:
            raise ValueError(f"Timestamp must be non-negative: timestamp={self.timestamp}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be positive: width={self.width}, height={self.height}")
        if self.face_count is not None and self.face_count < 0:
            raise ValueError(f"face_count must be non-negative: {self.face_count}")
        if self.faces is not None and self.faces < 0:
            raise ValueError(f"faces must be non-negative: {self.faces}")
        return self

    def validate(self, check_files: bool = True) -> "Snapshot":
        self._validate_fields()
        if check_files and not Path(self.path).is_file():
            raise ValueError(f"Referenced snapshot file does not exist: {self.path}")
        return self


class Word(ContractModel):
    """A single transcribed word with exact timestamps and confidence."""

    word: str
    start: float
    end: float
    confidence: float = 1.0

    @model_validator(mode="after")
    def _validate_fields(self) -> "Word":
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError(f"Timestamps must be non-negative: start={self.start}, end={self.end}")
        if self.start >= self.end:
            raise ValueError(f"start must be strictly less than end: {self.start} >= {self.end}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0]: {self.confidence}")
        return self

    def validate(self, check_files: bool = True) -> "Word":
        self._validate_fields()
        return self


class TranscriptSegment(ContractModel):
    """A sentence or phrase chunk of transcribed speech containing word-level timestamps."""

    start: float
    end: float
    text: str
    words: List[Word] = Field(default_factory=list)
    id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_timing(self) -> "TranscriptSegment":
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError(f"Timestamps must be non-negative: start={self.start}, end={self.end}")
        if self.start >= self.end:
            raise ValueError(f"start must be strictly less than end: {self.start} >= {self.end}")
        return self

    def validate(self, check_files: bool = True) -> "TranscriptSegment":
        self._validate_timing()
        for w in self.words:
            w.validate(check_files=check_files)
        return self


class Moment(ContractModel):
    """Candidate viral clip moment or timeline segment joining transcript and visual state."""

    id: str
    start: float
    end: float
    score: float = 0.0
    reason: str = ""
    hook_text: str = ""
    suggested_title: str = ""
    transcript_segment_ids: List[str] = Field(default_factory=list)
    snapshot_ids: List[str] = Field(default_factory=list)
    scene_ids: List[str] = Field(default_factory=list)
    text: str = ""
    pending: bool = False

    @model_validator(mode="after")
    def _validate_timing(self) -> "Moment":
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError(f"Timestamps must be non-negative: start={self.start}, end={self.end}")
        if self.start >= self.end:
            raise ValueError(f"start must be strictly less than end: {self.start} >= {self.end}")
        return self

    def validate(self, check_files: bool = True) -> "Moment":
        self._validate_timing()
        return self


class Shot(ContractModel):
    """Framing cut / crop angle within a moment for vertical 9:16 video."""

    id: str
    start: float
    end: float
    crop: Dict[str, float]  # {"x": ..., "y": ..., "w": ..., "h": ...}
    transition_in: Optional[str] = None
    transition_out: Optional[str] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "Shot":
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError(f"Timestamps must be non-negative: start={self.start}, end={self.end}")
        if self.start >= self.end:
            raise ValueError(f"start must be strictly less than end: {self.start} >= {self.end}")
        req_crop = ("x", "y", "w", "h")
        for k in req_crop:
            if k not in self.crop:
                raise ValueError(f"crop dictionary missing required key '{k}'")
            if self.crop[k] < 0.0:
                raise ValueError(f"crop key '{k}' must be non-negative")
        if self.crop["w"] <= 0.0 or self.crop["h"] <= 0.0:
            raise ValueError("crop width and height must be positive")
        return self

    def validate(self, check_files: bool = True) -> "Shot":
        self._validate_fields()
        return self


class ShotPlan(ContractModel):
    """Ordered shot arrangement composing a vertical 9:16 clip."""

    moment_id: str
    shots: List[Shot]
    audio_track: Optional[str] = None
    overlay_track: Optional[str] = None

    @model_validator(mode="after")
    def _validate_shots(self) -> "ShotPlan":
        if not self.shots:
            raise ValueError("ShotPlan must contain at least one shot")
        for i in range(len(self.shots) - 1):
            if self.shots[i].end > self.shots[i + 1].start + 1e-4:
                raise ValueError(
                    f"Shots overlap: shot[{i}] ends at {self.shots[i].end} but shot[{i+1}] starts at {self.shots[i+1].start}"
                )
        return self

    def validate(self, check_files: bool = True) -> "ShotPlan":
        self._validate_shots()
        for shot in self.shots:
            shot.validate(check_files=check_files)
        if check_files:
            if self.audio_track and not Path(self.audio_track).is_file():
                raise ValueError(f"Referenced audio_track does not exist: {self.audio_track}")
            if self.overlay_track and not Path(self.overlay_track).is_file():
                raise ValueError(f"Referenced overlay_track does not exist: {self.overlay_track}")
        return self


ASSET_TYPES = ("b-roll", "music", "sound_effect", "overlay", "font")


class Asset(ContractModel):
    """Reusable visual or audio media asset stored in the creator's library."""

    id: str
    name: str
    type: Literal["b-roll", "music", "sound_effect", "overlay", "font"]
    path: str
    duration: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    source: str = ""

    @model_validator(mode="after")
    def _validate_fields(self) -> "Asset":
        if self.duration is not None and self.duration < 0.0:
            raise ValueError(f"duration must be non-negative: {self.duration}")
        return self

    def validate(self, check_files: bool = True) -> "Asset":
        self._validate_fields()
        if check_files and not Path(self.path).is_file():
            raise ValueError(f"Referenced asset file does not exist: {self.path}")
        return self
