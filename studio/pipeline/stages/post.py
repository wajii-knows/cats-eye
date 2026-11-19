"""Post stage: schedules or publishes rendered clips to connected social accounts."""
from __future__ import annotations

import logging
from typing import Any, Dict

from studio.pipeline.stage import Stage, StageContext

logger = logging.getLogger("studio.pipeline.post")


class PostStage(Stage):
    name = "post"
    version = 1
    inputs = ("rendered_video",)
    outputs = ("post_results",)
    cacheable = False  # Publishing is an action, not cached

    def run(self, ctx: StageContext) -> Dict[str, Any]:
        # Optional post action; returns empty or schedule status if posting configured
        clip = ctx.inputs.get("current_clip", {})
        accounts = ctx.job.get("options", {}).get("accounts", [])
        return {
            "post_results": {
                "clip_id": clip.get("id"),
                "accounts": accounts,
                "status": "ready" if accounts else "skipped",
            }
        }
