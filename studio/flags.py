"""Feature flags for Cat's Eye.

All flags default to OFF (False) until their respective phase is merged and verified.
Flags can be set via environment variables (e.g. in .env) or overridden programmatically in tests.
"""
import os
from typing import Optional, Dict

FLAG_DEFAULTS: Dict[str, bool] = {
    "PEAKCUT_SNAPSHOTS": False,
    "PEAKCUT_VIRAL_V2": False,
    "PEAKCUT_ASSETS": False,
    "PEAKCUT_PRESETS": False,
    "PEAKCUT_MULTISHOT": False,
    "PEAKCUT_MATCHING": False,
    "PEAKCUT_HWACCEL": False,
    "PEAKCUT_WORKSPACE": False,
    "PEAKCUT_QC": False,
    "PEAKCUT_LEARNING": False,
}

FLAG_DESCRIPTIONS: Dict[str, str] = {
    "PEAKCUT_SNAPSHOTS": "Periodic visual frame extraction for multimodal AI moment detection.",
    "PEAKCUT_VIRAL_V2": "Multimodal viral scoring model (pacing, speech energy, facial expressions).",
    "PEAKCUT_ASSETS": "Asset library management (b-roll, music, sound effects, overlays, fonts).",
    "PEAKCUT_PRESETS": "Creator preset engine for typography, color grading, and layouts.",
    "PEAKCUT_MULTISHOT": "Dynamic shot planning (wide, close-up, crop, zoom) for 9:16 reframing.",
    "PEAKCUT_MATCHING": "Semantic asset matching engine (connecting b-roll to transcript).",
    "PEAKCUT_HWACCEL": "Hardware-accelerated encoding (NVENC, QSV, Apple VideoToolbox).",
    "PEAKCUT_WORKSPACE": "Multi-user creator workspaces, teams, and role-based permissions.",
    "PEAKCUT_QC": "Automated quality control checks (caption sync, audio loudness, black frames).",
    "PEAKCUT_LEARNING": "Feedback learning loop to refine hook detection and viral scoring.",
}

_overrides: Dict[str, bool] = {}


def is_on(name: str) -> bool:
    """Return whether a feature flag is enabled.

    Overrides take precedence, followed by environment variables, and finally default (False).
    """
    if name in _overrides:
        return _overrides[name]
    val = os.environ.get(name)
    if val is not None:
        return val.strip().lower() in ("1", "true", "yes", "on")
    return FLAG_DEFAULTS.get(name, False)


def all_flags() -> Dict[str, bool]:
    """Return a dictionary of all registered feature flags and their current state."""
    return {name: is_on(name) for name in FLAG_DEFAULTS}


def override(name: str, value: Optional[bool]) -> None:
    """Override a flag value in memory (e.g. for testing).

    Pass value=None to remove an override.
    """
    if value is None:
        _overrides.pop(name, None)
    else:
        _overrides[name] = bool(value)


def reset_overrides() -> None:
    """Reset all in-memory flag overrides."""
    _overrides.clear()
