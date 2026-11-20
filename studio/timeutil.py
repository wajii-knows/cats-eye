"""Shared timestamp, span clamping, and overlap helpers across brain, jobs, and render."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def clamp_span(start: float, end: float, duration: Optional[float] = None) -> Tuple[float, float]:
    """Clamp [start, end] to non-negative bounds and optional max duration."""
    s = max(0.0, float(start))
    e = max(s, float(end))
    if duration is not None and float(duration) > 0:
        d = float(duration)
        s = min(s, d)
        e = min(e, d)
    return round(s, 2), round(e, 2)


def shift_shots(shots: Optional[Iterable[Dict[str, Any]]], offset: float) -> List[Dict[str, Any]]:
    """Shift a list of shot dictionaries by `offset` seconds."""
    if not shots:
        return []
    off = float(offset)
    return [
        dict(s, start=round(float(s["start"]) + off, 2), end=round(float(s["end"]) + off, 2))
        for s in shots
        if isinstance(s, dict) and "start" in s and "end" in s
    ]


def overlaps(s: float, e: float, others: Iterable[Dict[str, Any]], ratio: float = 0.3) -> bool:
    """True when [s, e] overlaps any interval in `others` by more than `ratio` of its length."""
    span = e - s
    if span <= 0:
        return False
    return any(min(e, float(k["end"])) - max(s, float(k["start"])) > ratio * span for k in others)


def sec_to_ass(t: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc."""
    val = max(0.0, float(t))
    h, m = int(val // 3600), int(val % 3600 // 60)
    return f"{h}:{m:02d}:{val % 60:05.2f}"


from studio.pipeline.visual_index import (  # noqa: E402, F401
    VisualTimelineIndex,
    nearest_snapshot,
    scene_at,
    scenes_overlapping,
    segment_at,
    snapshots_in,
    words_in,
)
