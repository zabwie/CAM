from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from traffic_intel.ops.feedback import FeedbackModel
from traffic_intel.ops.store import IncidentStore


def _incident(event_id: str, score: float = 0.8, *, common_braking: float = 0.0, relative_dv: float = 0.2) -> dict:
    return {
        "event_id": event_id,
        "detected_at": time.time(),
        "trigger_type": "auto_collision",
        "title": "Possible collision",
        "municipality": "Example City",
        "location": "Main & 1st",
        "camera": "CAM-01",
        "detector_score": score,
        "involved_tracks": [1, 2],
        "description": "test",
        "evidence": {
            "contact": 0.9,
            "interaction_risk": 0.8,
            "kinematic": 0.7,
            "pair_relative_dv": relative_dv,
            "common_braking": common_braking,
        },
    }


def test_incident_review_search_and_feedback_export(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "ops.db")
    store.upsert_incident(_incident("evt-1"))
    assert store.get_incident("evt-1")["status"] == "pending"
    assert store.list_incidents(query="Main")

    reviewed = store.review_incident(
        "evt-1", decision="approve", reviewer="Alex", notes="Confirmed", corrected_type="collision"
    )
    assert reviewed["status"] == "approved"
    reviews = store.reviews_for_incident("evt-1")
    assert reviews[0]["reviewer"] == "Alex"
    assert reviews[0]["decision"] == "approve"

    output = store.export_feedback_jsonl(tmp_path / "feedback.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["decision"] == "approve"
    assert rows[0]["evidence"]["contact"] == 0.9


def test_feedback_model_learns_repeated_hard_braking_pattern(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "ops.db")
    # Balanced reviewed corpus: true impacts show relative motion change;
    # recurring common-braking cases are dismissed.
    for idx in range(12):
        event_id = f"true-{idx}"
        store.upsert_incident(_incident(event_id, 0.82, common_braking=0.0, relative_dv=0.24))
        store.review_incident(event_id, decision="approve", reviewer="reviewer")
    for idx in range(12):
        event_id = f"false-{idx}"
        store.upsert_incident(_incident(event_id, 0.82, common_braking=1.0, relative_dv=0.02))
        store.review_incident(event_id, decision="dismiss", reviewer="reviewer")

    model = store.feedback_model()
    assert model.active_ is True
    true_pred = model.predict(
        detector_score=0.82,
        evidence={"contact": 0.9, "interaction_risk": 0.8, "kinematic": 0.7, "pair_relative_dv": 0.24, "common_braking": 0.0},
        trigger_type="auto_collision",
    )
    false_pred = model.predict(
        detector_score=0.82,
        evidence={"contact": 0.9, "interaction_risk": 0.8, "kinematic": 0.7, "pair_relative_dv": 0.02, "common_braking": 1.0},
        trigger_type="auto_collision",
    )
    assert true_pred.active is True
    assert true_pred.probability > false_pred.probability
    assert true_pred.probability > 0.70
    assert false_pred.probability < 0.55


def test_speed_analytics_hotspots_and_camera_coordinates(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "ops.db")
    store.upsert_camera(
        camera_name="CAM-01", municipality="Example City", location="Main & 1st",
        latitude=18.4655, longitude=-66.1057, speed_limit_mph=35,
    )

    @dataclass
    class Det:
        track_id: int
        speed: float
        speed_valid: bool = True
        measurement_confidence: float = 0.9
        class_name: str = "car"

    base = time.time()
    for offset, speed in enumerate([30.0, 42.0, 48.0, 33.0]):
        store.record_speed_observations(
            [Det(offset + 1, speed)], observed_at=base + offset,
            municipality="Example City", location="Main & 1st", camera="CAM-01",
            speed_limit_mph=35,
        )
    store.upsert_incident(_incident("evt-hotspot"))
    store.review_incident("evt-hotspot", decision="approve", reviewer="reviewer")

    summary = store.analytics_summary(start_time=base - 10, end_time=base + 10)
    assert summary["speed_samples"] == 4
    assert summary["speeding_samples"] == 2
    assert summary["max_speed_mph"] == 48.0

    hotspots = store.hotspots(days=1)
    assert hotspots[0]["location"] == "Main & 1st"
    assert hotspots[0]["latitude"] == 18.4655
    assert hotspots[0]["longitude"] == -66.1057
    assert hotspots[0]["risk_index"] > 0


def test_notification_outbox_and_event_ingestion(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "ops.db")
    store.upsert_incident(_incident("evt-notify"))
    notification_id = store.queue_notification(
        "evt-notify", channel="in_app", payload={"title": "Possible collision"}
    )
    queued = store.queued_notifications()
    assert queued[0]["notification_id"] == notification_id
    assert queued[0]["payload"]["title"] == "Possible collision"
    store.mark_notification(notification_id, status="acknowledged")
    assert store.queued_notifications() == []

    event_dir = tmp_path / "events" / "package"
    event_dir.mkdir(parents=True)
    (event_dir / "clip.mp4").write_bytes(b"clip")
    (event_dir / "event.json").write_text(json.dumps({
        "event_id": "evt-imported",
        "trigger_time_unix": time.time(),
        "trigger_type": "auto_collision",
        "camera": "CAM-02",
        "location": "Second & Pine",
        "score": 0.77,
        "evidence": {"contact": 0.8},
    }))
    assert store.ingest_event_directory(tmp_path / "events") == 1
    imported = store.get_incident("evt-imported")
    assert imported is not None
    assert imported["clip_path"].endswith("clip.mp4")


def test_build_evidence_package_contains_manifest_and_files(tmp_path: Path) -> None:
    import zipfile

    from traffic_intel.ops.evidence import build_evidence_package

    store = IncidentStore(tmp_path / "ops.db")
    event_dir = tmp_path / "events" / "evt-package"
    event_dir.mkdir(parents=True)
    clip = event_dir / "clip.mp4"
    metadata = event_dir / "event.json"
    telemetry = event_dir / "telemetry.csv"
    checksum = event_dir / "clip.sha256"
    clip.write_bytes(b"video")
    metadata.write_text(json.dumps({"event_id": "evt-package"}))
    telemetry.write_text("frame,track_id\n1,7\n")
    checksum.write_text("abc\n")
    store.upsert_incident({
        **_incident("evt-package"),
        "clip_path": str(clip),
        "metadata_path": str(metadata),
    })
    store.review_incident("evt-package", decision="approve", reviewer="Dana", notes="confirmed")

    package = build_evidence_package(store, "evt-package", output_dir=tmp_path / "exports")
    assert package.exists()
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "evidence/clip.mp4" in names
        assert "evidence/event.json" in names
        assert "evidence/telemetry.csv" in names
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["incident"]["event_id"] == "evt-package"
        assert manifest["reviews"][0]["reviewer"] == "Dana"


def test_webhook_dispatcher_delivers_and_marks_sent(tmp_path: Path) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from traffic_intel.ops.notifications import WebhookDispatcher

    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            received.append(json.loads(self.rfile.read(length)))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A003 - stdlib API
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        store = IncidentStore(tmp_path / "ops.db")
        store.upsert_incident(_incident("evt-webhook"))
        store.queue_notification(
            "evt-webhook",
            channel="webhook",
            destination=f"http://127.0.0.1:{server.server_port}/incident",
            payload={"event_id": "evt-webhook", "priority": "high"},
        )
        results = WebhookDispatcher(store, timeout_seconds=2).dispatch_queued()
        thread.join(timeout=2)
        assert results[0].status == "sent"
        assert received == [{"event_id": "evt-webhook", "priority": "high"}]
        assert store.queued_notifications() == []
    finally:
        server.server_close()


def test_speed_analytics_use_configured_agency_timezone(tmp_path: Path) -> None:
    import datetime as dt

    store = IncidentStore(tmp_path / "ops.db")

    @dataclass
    class Det:
        track_id: int = 1
        speed: float = 50.0
        speed_valid: bool = True
        measurement_confidence: float = 0.9
        class_name: str = "car"

    # 2026-07-12 03:00 UTC is 2026-07-11 23:00 in Puerto Rico.
    observed_at = dt.datetime(2026, 7, 12, 3, 0, tzinfo=dt.timezone.utc).timestamp()
    store.record_speed_observations(
        [Det()], observed_at=observed_at, location="Main", camera="CAM-TZ",
        speed_limit_mph=35, timezone_name="America/Puerto_Rico",
    )
    days = store.dangerous_days(days=3650)
    hours = store.dangerous_hours(days=3650)
    assert days[0]["weekday"] == "Saturday"
    assert hours[0]["hour"] == 23
