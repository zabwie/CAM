"""Evidence-package export for reviewed incidents."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from traffic_intel.ops.store import IncidentStore


def build_evidence_package(
    store: IncidentStore,
    event_id: str,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Create a portable ZIP containing incident metadata, review history, and evidence files."""
    incident = store.get_incident(event_id)
    if incident is None:
        raise KeyError(f"unknown event_id: {event_id}")
    reviews = store.reviews_for_incident(event_id)

    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="traffic-intel-evidence-"))
    root.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch for ch in event_id if ch.isalnum() or ch in {"-", "_"}) or "incident"
    output = root / f"evidence_{safe_id}.zip"

    portable_incident = dict(incident)
    portable_incident["clip_path"] = "evidence/clip.mp4" if incident.get("clip_path") else ""
    portable_incident["metadata_path"] = "evidence/event.json" if incident.get("metadata_path") else ""
    manifest: dict[str, Any] = {
        "schema": "traffic-intel-evidence/v1",
        "incident": portable_incident,
        "reviews": reviews,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for key in ("clip_path", "metadata_path"):
            value = str(incident.get(key) or "")
            if not value:
                continue
            path = Path(value)
            if path.exists() and path.is_file():
                archive.write(path, arcname=f"evidence/{path.name}")
                if key == "metadata_path":
                    for sibling_name in ("telemetry.csv", "clip.sha256"):
                        sibling = path.parent / sibling_name
                        if sibling.exists() and sibling.is_file():
                            archive.write(sibling, arcname=f"evidence/{sibling.name}")
    return output
