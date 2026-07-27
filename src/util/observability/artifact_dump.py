"""Incremental, opt-in intermediate-artifact dumping for orchestration stages.

Problem: existing tracing (llm_trace.py, run_rschema_batch.py's trace_dir) only
persists artifacts AFTER a stage returns successfully. If a stage crashes
internally, everything computed up to that point (e.g. a merged conceptual
model, right before the step that crashes) is lost, forcing a full re-run --
expensive when stages make live LLM calls.

This writes each named artifact to disk THE MOMENT it is produced, so a
mid-pipeline crash still leaves every prior step inspectable without re-running
anything. No-ops entirely when artifact_dir is None -- zero cost/behavior
change for existing callers that don't opt in.
"""

import json
from pathlib import Path
from typing import Any, Optional


def dump_artifact(artifact_dir: Optional[Path], name: str, obj: Any) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{name}.json"

    if hasattr(obj, "model_dump"):
        payload = obj.model_dump(mode="json")
    elif isinstance(obj, (list, tuple)):
        payload = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in obj
        ]
    else:
        payload = obj

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
