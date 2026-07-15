"""SQLite-backed operational store for incidents, reviews, telemetry, and alerts."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
import datetime as dt
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from traffic_intel.ops.feedback import FeedbackModel


SCHEMA_VERSION = 3


class IncidentStore:
    """Durable local repository suitable for a single-node pilot deployment."""

    def __init__(self, path: str | Path = "data/traffic_intel.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS incidents (
                    event_id TEXT PRIMARY KEY,
                    detected_at REAL NOT NULL,
                    trigger_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    municipality TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    camera TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    detector_score REAL,
                    review_score REAL,
                    priority TEXT NOT NULL DEFAULT 'normal',
                    involved_tracks_json TEXT NOT NULL DEFAULT '[]',
                    description TEXT NOT NULL DEFAULT '',
                    clip_path TEXT NOT NULL DEFAULT '',
                    metadata_path TEXT NOT NULL DEFAULT '',
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_incidents_detected_at ON incidents(detected_at DESC);
                CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
                CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents(location);
                CREATE INDEX IF NOT EXISTS idx_incidents_camera ON incidents(camera);

                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    municipality TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    camera_name TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    latitude REAL,
                    longitude REAL,
                    speed_limit_mph REAL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cameras_location ON cameras(location, camera_name);

                CREATE TABLE IF NOT EXISTS reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL REFERENCES incidents(event_id) ON DELETE CASCADE,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    corrected_type TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reviews_event ON reviews(event_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS speed_observations (
                    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at REAL NOT NULL,
                    municipality TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    camera TEXT NOT NULL DEFAULT '',
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL DEFAULT '',
                    speed_mph REAL NOT NULL,
                    speed_limit_mph REAL,
                    measurement_confidence REAL NOT NULL DEFAULT 0,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    local_weekday INTEGER,
                    local_hour INTEGER,
                    UNIQUE(camera, track_id, observed_at)
                );
                CREATE INDEX IF NOT EXISTS idx_speed_time ON speed_observations(observed_at);
                CREATE INDEX IF NOT EXISTS idx_speed_location ON speed_observations(location, observed_at);

                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL REFERENCES incidents(event_id) ON DELETE CASCADE,
                    channel TEXT NOT NULL,
                    destination TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    sent_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status, created_at);
                """
            )
            # Lightweight forward migrations for pilot databases created by earlier releases.
            migrations = {
                "cameras": {"timezone": "TEXT NOT NULL DEFAULT 'UTC'"},
                "speed_observations": {
                    "timezone": "TEXT NOT NULL DEFAULT 'UTC'",
                    "local_weekday": "INTEGER",
                    "local_hour": "INTEGER",
                },
            }
            for table, columns in migrations.items():
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                for column, declaration in columns.items():
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decode_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        for source, target, default in (
            ("involved_tracks_json", "involved_tracks", []),
            ("evidence_json", "evidence", {}),
            ("payload_json", "payload", {}),
        ):
            if source in data:
                try:
                    data[target] = json.loads(data.pop(source))
                except (TypeError, json.JSONDecodeError):
                    data[target] = default
        return data

    def upsert_camera(
        self,
        *,
        camera_name: str,
        municipality: str = "",
        location: str = "",
        source: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
        speed_limit_mph: float | None = None,
        timezone_name: str = "UTC",
        active: bool = True,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        name = str(camera_name).strip() or "Unspecified camera"
        identifier = str(camera_id or name).strip()
        timezone_name = str(timezone_name or "UTC").strip()
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {timezone_name}") from exc
        now = time.time()
        lat = float(latitude) if latitude is not None else None
        lon = float(longitude) if longitude is not None else None
        if lat is not None and not (-90.0 <= lat <= 90.0):
            raise ValueError("latitude must be between -90 and 90")
        if lon is not None and not (-180.0 <= lon <= 180.0):
            raise ValueError("longitude must be between -180 and 180")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO cameras(camera_id, municipality, location, camera_name, source, latitude, longitude, speed_limit_mph, timezone, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    municipality=excluded.municipality, location=excluded.location,
                    camera_name=excluded.camera_name, source=excluded.source,
                    latitude=COALESCE(excluded.latitude, cameras.latitude),
                    longitude=COALESCE(excluded.longitude, cameras.longitude),
                    speed_limit_mph=COALESCE(excluded.speed_limit_mph, cameras.speed_limit_mph),
                    timezone=excluded.timezone, active=excluded.active, updated_at=excluded.updated_at
                """,
                (identifier, str(municipality), str(location), name, str(source), lat, lon,
                 float(speed_limit_mph) if speed_limit_mph is not None else None, timezone_name,
                 int(bool(active)), now, now),
            )
            row = conn.execute("SELECT * FROM cameras WHERE camera_id = ?", (identifier,)).fetchone()
        assert row is not None
        return dict(row)

    def list_cameras(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM cameras" + (" WHERE active = 1" if active_only else "") + " ORDER BY location, camera_name"
        with self.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def upsert_incident(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        event_id = str(payload.get("event_id") or uuid.uuid4())
        detector_score = payload.get("detector_score", payload.get("score"))
        evidence = payload.get("evidence") or {}
        trigger_type = str(payload.get("trigger_type") or "event")
        title = str(payload.get("title") or (
            "Possible collision" if trigger_type.startswith("auto_") else "Manual capture"
        ))
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO incidents (
                    event_id, detected_at, trigger_type, title, municipality, location,
                    camera, source, status, detector_score, review_score, priority,
                    involved_tracks_json, description, clip_path, metadata_path,
                    evidence_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    detected_at=excluded.detected_at,
                    trigger_type=excluded.trigger_type,
                    title=excluded.title,
                    municipality=excluded.municipality,
                    location=excluded.location,
                    camera=excluded.camera,
                    source=excluded.source,
                    detector_score=COALESCE(excluded.detector_score, incidents.detector_score),
                    review_score=COALESCE(excluded.review_score, incidents.review_score),
                    priority=excluded.priority,
                    involved_tracks_json=excluded.involved_tracks_json,
                    description=excluded.description,
                    clip_path=excluded.clip_path,
                    metadata_path=excluded.metadata_path,
                    evidence_json=excluded.evidence_json,
                    updated_at=excluded.updated_at
                """,
                (
                    event_id,
                    float(payload.get("detected_at", payload.get("trigger_time_unix", now))),
                    trigger_type,
                    title,
                    str(payload.get("municipality") or ""),
                    str(payload.get("location") or ""),
                    str(payload.get("camera") or payload.get("camera_id") or ""),
                    str(payload.get("source") or ""),
                    str(payload.get("status") or "pending"),
                    float(detector_score) if isinstance(detector_score, (int, float)) else None,
                    float(payload["review_score"]) if isinstance(payload.get("review_score"), (int, float)) else None,
                    str(payload.get("priority") or "normal"),
                    self._json(payload.get("involved_tracks") or []),
                    str(payload.get("description") or ""),
                    str(payload.get("clip_path") or ""),
                    str(payload.get("metadata_path") or ""),
                    self._json(evidence),
                    now,
                    now,
                ),
            )
        incident = self.get_incident(event_id)
        assert incident is not None
        return incident

    def ingest_event_directory(self, event_dir: str | Path) -> int:
        """Import finalized recorder packages into the operations database idempotently."""
        root = Path(event_dir)
        if not root.exists():
            return 0
        imported = 0
        for metadata_path in root.rglob("event.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            event_id = str(metadata.get("event_id") or metadata_path.parent.name)
            clip_path = metadata_path.parent / "clip.mp4"
            trigger_type = str(metadata.get("trigger_type") or "event")
            self.upsert_incident({
                **metadata,
                "event_id": event_id,
                "detected_at": float(metadata.get("trigger_time_unix") or metadata_path.stat().st_mtime),
                "title": "Possible collision" if trigger_type.startswith("auto_") else "Manual capture",
                "detector_score": metadata.get("score"),
                "review_score": metadata.get("review_score"),
                "clip_path": str(clip_path) if clip_path.exists() else "",
                "metadata_path": str(metadata_path),
                "status": "pending",
            })
            imported += 1
        return imported

    def get_incident(self, event_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM incidents WHERE event_id = ?", (event_id,)).fetchone()
        return self._decode_row(row) if row else None

    def list_incidents(
        self,
        *,
        status: str | None = None,
        query: str = "",
        location: str | None = None,
        camera: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if location:
            clauses.append("location = ?")
            params.append(location)
        if camera:
            clauses.append("camera = ?")
            params.append(camera)
        if start_time is not None:
            clauses.append("detected_at >= ?")
            params.append(float(start_time))
        if end_time is not None:
            clauses.append("detected_at <= ?")
            params.append(float(end_time))
        if query.strip():
            term = f"%{query.strip()}%"
            clauses.append(
                "(event_id LIKE ? OR title LIKE ? OR description LIKE ? OR location LIKE ? OR camera LIKE ?)"
            )
            params.extend([term] * 5)
        params.append(max(1, min(int(limit), 5000)))
        sql = f"SELECT * FROM incidents WHERE {' AND '.join(clauses)} ORDER BY detected_at DESC LIMIT ?"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def review_incident(
        self,
        event_id: str,
        *,
        decision: str,
        reviewer: str = "",
        notes: str = "",
        corrected_type: str = "",
    ) -> dict[str, Any]:
        decision = str(decision).lower().strip()
        status_map = {"approve": "approved", "dismiss": "dismissed", "needs_info": "needs_info"}
        if decision not in status_map:
            raise ValueError("decision must be approve, dismiss, or needs_info")
        now = time.time()
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM incidents WHERE event_id = ?", (event_id,)).fetchone()
            if not exists:
                raise KeyError(f"unknown event_id: {event_id}")
            conn.execute(
                "INSERT INTO reviews(event_id, decision, reviewer, notes, corrected_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, decision, reviewer.strip(), notes.strip(), corrected_type.strip(), now),
            )
            conn.execute(
                "UPDATE incidents SET status = ?, updated_at = ? WHERE event_id = ?",
                (status_map[decision], now, event_id),
            )
        incident = self.get_incident(event_id)
        assert incident is not None
        return incident

    def reviews_for_incident(self, event_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE event_id = ? ORDER BY created_at DESC", (event_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def feedback_training_rows(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT i.event_id, i.detector_score, i.evidence_json AS evidence,
                       i.trigger_type, r.decision, r.created_at
                FROM incidents i
                JOIN reviews r ON r.review_id = (
                    SELECT r2.review_id FROM reviews r2
                    WHERE r2.event_id = i.event_id AND r2.decision IN ('approve','dismiss')
                    ORDER BY r2.created_at DESC LIMIT 1
                )
                ORDER BY r.created_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def feedback_model(self) -> FeedbackModel:
        return FeedbackModel().fit(self.feedback_training_rows())

    def refresh_review_scores(self) -> int:
        model = self.feedback_model()
        incidents = self.list_incidents(limit=5000)
        now = time.time()
        updated = 0
        with self.connect() as conn:
            for item in incidents:
                prediction = model.predict(
                    detector_score=item.get("detector_score"),
                    evidence=item.get("evidence"),
                    trigger_type=item.get("trigger_type", ""),
                )
                score = prediction.probability
                priority = "high" if score >= 0.80 else "normal" if score >= 0.50 else "low"
                conn.execute(
                    "UPDATE incidents SET review_score = ?, priority = ?, updated_at = ? WHERE event_id = ?",
                    (score, priority, now, item["event_id"]),
                )
                updated += 1
        return updated

    def record_speed_observations(
        self,
        detections: Iterable[Any],
        *,
        observed_at: float,
        municipality: str = "",
        location: str = "",
        camera: str = "",
        speed_limit_mph: float | None = None,
        min_confidence: float = 0.35,
        timezone_name: str = "UTC",
    ) -> int:
        timezone_name = str(timezone_name or "UTC").strip()
        try:
            zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {timezone_name}") from exc
        local_dt = dt.datetime.fromtimestamp(float(observed_at), tz=zone)
        # Python Monday=0; store SQLite-compatible Sunday=0 for stable labels.
        local_weekday = (local_dt.weekday() + 1) % 7
        local_hour = local_dt.hour
        rows = []
        for det in detections:
            if not bool(getattr(det, "speed_valid", False)):
                continue
            confidence = float(getattr(det, "measurement_confidence", 0.0))
            if confidence < min_confidence:
                continue
            speed = float(getattr(det, "speed", 0.0))
            if speed < 0 or speed > 220:
                continue
            rows.append((
                float(observed_at), municipality, location, camera,
                int(getattr(det, "track_id")), str(getattr(det, "class_name", "vehicle")),
                speed, float(speed_limit_mph) if speed_limit_mph is not None else None,
                confidence, timezone_name, local_weekday, local_hour,
            ))
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO speed_observations(
                    observed_at, municipality, location, camera, track_id, class_name,
                    speed_mph, speed_limit_mph, measurement_confidence, timezone, local_weekday, local_hour
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def analytics_summary(
        self,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        speeding_margin_mph: float = 5.0,
    ) -> dict[str, Any]:
        start = float(start_time) if start_time is not None else 0.0
        end = float(end_time) if end_time is not None else time.time()
        margin = float(speeding_margin_mph)
        with self.connect() as conn:
            incident = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                       SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) AS approved,
                       SUM(CASE WHEN status='dismissed' THEN 1 ELSE 0 END) AS dismissed
                FROM incidents WHERE detected_at BETWEEN ? AND ?
                """,
                (start, end),
            ).fetchone()
            speed = conn.execute(
                """
                SELECT COUNT(*) AS samples, AVG(speed_mph) AS avg_speed, MAX(speed_mph) AS max_speed,
                       SUM(CASE WHEN speed_limit_mph IS NOT NULL AND speed_mph > speed_limit_mph + ? THEN 1 ELSE 0 END) AS speeding
                FROM speed_observations WHERE observed_at BETWEEN ? AND ?
                """,
                (margin, start, end),
            ).fetchone()
        speed_samples = int(speed["samples"] or 0)
        speeding = int(speed["speeding"] or 0)
        return {
            "incidents_total": int(incident["total"] or 0),
            "incidents_pending": int(incident["pending"] or 0),
            "incidents_approved": int(incident["approved"] or 0),
            "incidents_dismissed": int(incident["dismissed"] or 0),
            "speed_samples": speed_samples,
            "avg_speed_mph": float(speed["avg_speed"] or 0.0),
            "max_speed_mph": float(speed["max_speed"] or 0.0),
            "speeding_samples": speeding,
            "speeding_rate": (speeding / speed_samples) if speed_samples else 0.0,
        }

    def dangerous_days(self, *, days: int = 90, speeding_margin_mph: float = 5.0) -> list[dict[str, Any]]:
        cutoff = time.time() - max(1, int(days)) * 86400
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(local_weekday, CAST(strftime('%w', observed_at, 'unixepoch') AS INTEGER)) AS weekday,
                       COUNT(*) AS samples,
                       AVG(speed_mph) AS avg_speed,
                       MAX(speed_mph) AS max_speed,
                       SUM(CASE WHEN speed_limit_mph IS NOT NULL AND speed_mph > speed_limit_mph + ? THEN 1 ELSE 0 END) AS speeding
                FROM speed_observations
                WHERE observed_at >= ?
                GROUP BY weekday
                """,
                (float(speeding_margin_mph), cutoff),
            ).fetchall()
        names = {"0":"Sunday","1":"Monday","2":"Tuesday","3":"Wednesday","4":"Thursday","5":"Friday","6":"Saturday"}
        result = []
        for row in rows:
            samples = int(row["samples"] or 0)
            speeding = int(row["speeding"] or 0)
            result.append({
                "weekday": names.get(str(row["weekday"]), str(row["weekday"])),
                "samples": samples,
                "avg_speed_mph": float(row["avg_speed"] or 0.0),
                "max_speed_mph": float(row["max_speed"] or 0.0),
                "speeding_rate": speeding / samples if samples else 0.0,
            })
        return sorted(result, key=lambda row: (row["speeding_rate"], row["avg_speed_mph"]), reverse=True)

    def dangerous_hours(self, *, days: int = 30, speeding_margin_mph: float = 5.0) -> list[dict[str, Any]]:
        cutoff = time.time() - max(1, int(days)) * 86400
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(local_hour, CAST(strftime('%H', observed_at, 'unixepoch') AS INTEGER)) AS hour,
                       COUNT(*) AS samples, AVG(speed_mph) AS avg_speed, MAX(speed_mph) AS max_speed,
                       SUM(CASE WHEN speed_limit_mph IS NOT NULL AND speed_mph > speed_limit_mph + ? THEN 1 ELSE 0 END) AS speeding
                FROM speed_observations WHERE observed_at >= ? GROUP BY hour ORDER BY hour
                """,
                (float(speeding_margin_mph), cutoff),
            ).fetchall()
        result = []
        for row in rows:
            samples = int(row["samples"] or 0)
            speeding = int(row["speeding"] or 0)
            result.append({
                "hour": int(row["hour"]), "samples": samples,
                "avg_speed_mph": float(row["avg_speed"] or 0.0),
                "max_speed_mph": float(row["max_speed"] or 0.0),
                "speeding_rate": speeding / samples if samples else 0.0,
            })
        return result

    def hotspots(self, *, days: int = 90, speeding_margin_mph: float = 5.0) -> list[dict[str, Any]]:
        cutoff = time.time() - max(1, int(days)) * 86400
        with self.connect() as conn:
            incident_rows = conn.execute(
                """
                SELECT location, camera, COUNT(*) AS incidents,
                       SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) AS approved_incidents
                FROM incidents WHERE detected_at >= ?
                GROUP BY location, camera
                """,
                (cutoff,),
            ).fetchall()
            speed_rows = conn.execute(
                """
                SELECT location, camera, COUNT(*) AS samples, AVG(speed_mph) AS avg_speed,
                       SUM(CASE WHEN speed_limit_mph IS NOT NULL AND speed_mph > speed_limit_mph + ? THEN 1 ELSE 0 END) AS speeding
                FROM speed_observations WHERE observed_at >= ?
                GROUP BY location, camera
                """,
                (float(speeding_margin_mph), cutoff),
            ).fetchall()
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in incident_rows:
            key = (str(row["location"] or "Unknown"), str(row["camera"] or "Unknown"))
            by_key[key] = {
                "location": key[0], "camera": key[1],
                "incidents": int(row["incidents"] or 0),
                "approved_incidents": int(row["approved_incidents"] or 0),
                "speed_samples": 0, "avg_speed_mph": 0.0, "speeding_rate": 0.0,
            }
        for row in speed_rows:
            key = (str(row["location"] or "Unknown"), str(row["camera"] or "Unknown"))
            item = by_key.setdefault(key, {
                "location": key[0], "camera": key[1], "incidents": 0,
                "approved_incidents": 0, "speed_samples": 0,
                "avg_speed_mph": 0.0, "speeding_rate": 0.0,
            })
            samples = int(row["samples"] or 0)
            item["speed_samples"] = samples
            item["avg_speed_mph"] = float(row["avg_speed"] or 0.0)
            item["speeding_rate"] = int(row["speeding"] or 0) / samples if samples else 0.0
        camera_meta = {
            (str(row.get("location") or "Unknown"), str(row.get("camera_name") or "Unknown")): row
            for row in self.list_cameras()
        }
        for item in by_key.values():
            # Transparent operational risk index for ranking, not a claim of crash probability.
            item["risk_index"] = round(
                3.0 * item["approved_incidents"] + 1.0 * item["incidents"] + 10.0 * item["speeding_rate"], 3
            )
            meta = camera_meta.get((str(item["location"]), str(item["camera"]))) or {}
            item["latitude"] = meta.get("latitude")
            item["longitude"] = meta.get("longitude")
        return sorted(by_key.values(), key=lambda item: item["risk_index"], reverse=True)

    def queue_notification(
        self,
        event_id: str,
        *,
        channel: str,
        destination: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        now = time.time()
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO notifications(event_id, channel, destination, status, payload_json, created_at) VALUES (?, ?, ?, 'queued', ?, ?)",
                (event_id, channel, destination, self._json(payload or {}), now),
            )
            return int(cursor.lastrowid)

    def queued_notifications(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications WHERE status='queued' ORDER BY created_at ASC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def mark_notification(
        self, notification_id: int, *, status: str, sent_at: float | None = None
    ) -> None:
        allowed = {"queued", "sent", "failed", "acknowledged"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        with self.connect() as conn:
            conn.execute(
                "UPDATE notifications SET status = ?, sent_at = ? WHERE notification_id = ?",
                (status, float(sent_at) if sent_at is not None else (time.time() if status == "sent" else None), int(notification_id)),
            )

    def export_feedback_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.feedback_training_rows()
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = dict(row)
                if isinstance(payload.get("evidence"), str):
                    try:
                        payload["evidence"] = json.loads(payload["evidence"])
                    except json.JSONDecodeError:
                        pass
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return output
