"""O(log n) bisect-based visual & transcript timeline helpers for Cat's Eye."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, List, Optional, Sequence


def _snap_t(item: Any) -> float:
    if isinstance(item, dict):
        if "timestamp" in item:
            return float(item["timestamp"])
        return float(item.get("t", 0.0))
    if hasattr(item, "timestamp"):
        return float(item.timestamp)
    return float(getattr(item, "t", 0.0))


def _span_start(item: Any) -> float:
    if isinstance(item, dict):
        if "start" in item:
            return float(item["start"])
        return float(item.get("s", 0.0))
    if hasattr(item, "start"):
        return float(item.start)
    return float(getattr(item, "s", 0.0))


def _span_end(item: Any) -> float:
    if isinstance(item, dict):
        if "end" in item:
            return float(item["end"])
        return float(item.get("e", 0.0))
    if hasattr(item, "end"):
        return float(item.end)
    return float(getattr(item, "e", 0.0))


def _sorted_snapshots(snapshots: Sequence[Any]) -> List[Any]:
    return sorted(snapshots, key=_snap_t)


def _sorted_spans(items: Sequence[Any]) -> List[Any]:
    return sorted(items, key=lambda x: (_span_start(x), _span_end(x)))


def snapshots_in(
    snapshots: Sequence[Any],
    start: float,
    end: float,
    _ts: Optional[Sequence[float]] = None,
) -> List[Any]:
    """Return snapshots whose timestamp t lies in half-open interval [start, end) via O(log n) bisect."""
    if not snapshots or end <= start:
        return []
    ts = _ts if _ts is not None else [_snap_t(x) for x in snapshots]
    lo = bisect_left(ts, float(start))
    hi = bisect_left(ts, float(end))
    return list(snapshots[lo:hi])


def nearest_snapshot(
    snapshots: Sequence[Any],
    t: float,
    _ts: Optional[Sequence[float]] = None,
) -> Optional[Any]:
    """Return the snapshot closest in time to `t` via O(log n) bisect (ties break left)."""
    if not snapshots:
        return None
    ts = _ts if _ts is not None else [_snap_t(x) for x in snapshots]
    idx = bisect_left(ts, float(t))
    if idx == 0:
        return snapshots[0]
    if idx >= len(snapshots):
        return snapshots[-1]
    left_d = float(t) - ts[idx - 1]
    right_d = ts[idx] - float(t)
    return snapshots[idx - 1] if left_d <= right_d else snapshots[idx]


def scene_at(
    scenes: Sequence[Any],
    t: float,
    _starts: Optional[Sequence[float]] = None,
) -> Optional[Any]:
    """Return the scene containing `t` in half-open [start, end) via O(log n) bisect, or None."""
    if not scenes or t < 0.0:
        return None
    starts = _starts if _starts is not None else [_span_start(x) for x in scenes]
    idx = bisect_right(starts, float(t)) - 1
    if idx >= 0:
        sc = scenes[idx]
        if _span_start(sc) <= float(t) < _span_end(sc):
            return sc
    return None


def segment_at(
    segments: Sequence[Any],
    t: float,
    _starts: Optional[Sequence[float]] = None,
) -> Optional[Any]:
    """Return the transcript segment containing `t` in half-open [start, end) via O(log n) bisect, or None."""
    if not segments or t < 0.0:
        return None
    starts = _starts if _starts is not None else [_span_start(x) for x in segments]
    idx = bisect_right(starts, float(t)) - 1
    if idx >= 0:
        seg = segments[idx]
        if _span_start(seg) <= float(t) < _span_end(seg):
            return seg
    return None


def words_in(
    words: Sequence[Any],
    start: float,
    end: float,
    _starts: Optional[Sequence[float]] = None,
) -> List[Any]:
    """Return words whose start falls in half-open [start, end) via O(log n) bisect."""
    if not words or end <= start:
        return []
    starts = _starts if _starts is not None else [_span_start(x) for x in words]
    lo = bisect_left(starts, float(start))
    hi = bisect_left(starts, float(end))
    return list(words[lo:hi])


def scenes_overlapping(
    scenes: Sequence[Any],
    start: float,
    end: float,
    _starts: Optional[Sequence[float]] = None,
) -> List[Any]:
    """Return scenes overlapping half-open interval [start, end) via O(log n) bisect."""
    if not scenes or end <= start:
        return []
    starts = _starts if _starts is not None else [_span_start(x) for x in scenes]
    lo = max(0, bisect_right(starts, float(start)) - 1)
    hi = bisect_left(starts, float(end))
    out = []
    for sc in scenes[lo:hi]:
        if _span_start(sc) < float(end) and _span_end(sc) > float(start):
            out.append(sc)
    return out


class VisualTimelineIndex:
    """Sorted timeline index providing O(log n) bisect lookups across scenes, snapshots, segments, and words."""

    def __init__(
        self,
        scenes: Sequence[Any] = (),
        snapshots: Sequence[Any] = (),
        segments: Sequence[Any] = (),
        words: Sequence[Any] = (),
    ):
        self.scenes = _sorted_spans(scenes)
        self.snapshots = _sorted_snapshots(snapshots)
        self.segments = _sorted_spans(segments)
        self.words = _sorted_spans(words)
        self._scene_starts = [_span_start(s) for s in self.scenes]
        self._snap_ts = [_snap_t(s) for s in self.snapshots]
        self._seg_starts = [_span_start(s) for s in self.segments]
        self._word_starts = [_span_start(w) for w in self.words]

    def snapshots_in(self, start: float, end: float) -> List[Any]:
        return snapshots_in(self.snapshots, start, end, _ts=self._snap_ts)

    def nearest_snapshot(self, t: float) -> Optional[Any]:
        return nearest_snapshot(self.snapshots, t, _ts=self._snap_ts)

    def scene_at(self, t: float) -> Optional[Any]:
        return scene_at(self.scenes, t, _starts=self._scene_starts)

    def segment_at(self, t: float) -> Optional[Any]:
        return segment_at(self.segments, t, _starts=self._seg_starts)

    def words_in(self, start: float, end: float) -> List[Any]:
        return words_in(self.words, start, end, _starts=self._word_starts)

    def scenes_overlapping(self, start: float, end: float) -> List[Any]:
        return scenes_overlapping(self.scenes, start, end, _starts=self._scene_starts)
