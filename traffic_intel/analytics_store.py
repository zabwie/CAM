"""SQLite persistence and report queries for pilot traffic analytics."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from .analytics import CameraHealthRecord, VehiclePassage


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SiteSummary:
    camera_id: str
    vehicle_count: int
    valid_speed_count: int
    average_speed_mph: float | None
    median_speed_mph: float | None
    percentile_85_speed_mph: float | None
    speeding_count: int
    speeding_rate: float | None


class AnalyticsStore:
    """Small, durable SQLite ledger suitable for a two-camera pilot."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> "AnalyticsStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._connection.commit()
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vehicle_passages (
                passage_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                municipality TEXT NOT NULL,
                location_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                canonical_track_id INTEGER NOT NULL,
                vehicle_class TEXT NOT NULL,
                first_seen_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                observed_seconds REAL NOT NULL,
                valid_speed_samples INTEGER NOT NULL,
                representative_speed_mph REAL,
                max_speed_mph REAL,
                measurement_confidence REAL NOT NULL,
                speed_limit_mph REAL,
                speeding INTEGER,
                vision_state TEXT NOT NULL,
                calibration_id TEXT NOT NULL,
                software_version TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (unixepoch())
            );

            CREATE INDEX IF NOT EXISTS idx_passages_camera_time
                ON vehicle_passages(camera_id, first_seen_at);
            CREATE INDEX IF NOT EXISTS idx_passages_location_time
                ON vehicle_passages(location_id, first_seen_at);
            CREATE INDEX IF NOT EXISTS idx_passages_speeding
                ON vehicle_passages(camera_id, speeding, first_seen_at);

            CREATE TABLE IF NOT EXISTS camera_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                bucket_start REAL NOT NULL,
                bucket_end REAL NOT NULL,
                frames_received INTEGER NOT NULL,
                frames_analyzed INTEGER NOT NULL,
                frame_gap_count INTEGER NOT NULL,
                analysis_fps REAL NOT NULL,
                detections INTEGER NOT NULL,
                valid_speed_detections INTEGER NOT NULL,
                valid_speed_rate REAL NOT NULL,
                median_brightness REAL NOT NULL,
                dark_fraction REAL NOT NULL,
                clipped_highlight_fraction REAL NOT NULL,
                median_saturation REAL NOT NULL,
                blur_score REAL NOT NULL,
                vision_state TEXT NOT NULL,
                UNIQUE(camera_id, bucket_start)
            );

            CREATE INDEX IF NOT EXISTS idx_health_camera_time
                ON camera_health(camera_id, bucket_start);

            CREATE TABLE IF NOT EXISTS incidents (
                event_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                incident_type TEXT NOT NULL,
                score REAL,
                participant_count INTEGER NOT NULL,
                involved_tracks_json TEXT NOT NULL,
                evidence_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._connection.commit()

    def write_passages(self, passages: Iterable[VehiclePassage]) -> int:
        rows = list(passages)
        if not rows:
            return 0
        self._connection.executemany(
            """
            INSERT OR REPLACE INTO vehicle_passages (
                passage_id, session_id, municipality, location_id, camera_id,
                canonical_track_id, vehicle_class, first_seen_at, last_seen_at,
                observed_seconds, valid_speed_samples,
                representative_speed_mph, max_speed_mph,
                measurement_confidence, speed_limit_mph, speeding,
                vision_state, calibration_id, software_version
            ) VALUES (
                :passage_id, :session_id, :municipality, :location_id, :camera_id,
                :canonical_track_id, :vehicle_class, :first_seen_at, :last_seen_at,
                :observed_seconds, :valid_speed_samples,
                :representative_speed_mph, :max_speed_mph,
                :measurement_confidence, :speed_limit_mph, :speeding,
                :vision_state, :calibration_id, :software_version
            )
            """,
            [
                {
                    **asdict(row),
                    "speeding": (
                        None if row.speeding is None else int(row.speeding)
                    ),
                }
                for row in rows
            ],
        )
        self._connection.commit()
        return len(rows)

    def write_camera_health(self, records: Iterable[CameraHealthRecord]) -> int:
        rows = list(records)
        if not rows:
            return 0
        self._connection.executemany(
            """
            INSERT OR REPLACE INTO camera_health (
                camera_id, bucket_start, bucket_end, frames_received,
                frames_analyzed, frame_gap_count, analysis_fps, detections,
                valid_speed_detections, valid_speed_rate, median_brightness,
                dark_fraction, clipped_highlight_fraction, median_saturation,
                blur_score, vision_state
            ) VALUES (
                :camera_id, :bucket_start, :bucket_end, :frames_received,
                :frames_analyzed, :frame_gap_count, :analysis_fps, :detections,
                :valid_speed_detections, :valid_speed_rate, :median_brightness,
                :dark_fraction, :clipped_highlight_fraction, :median_saturation,
                :blur_score, :vision_state
            )
            """,
            [asdict(row) for row in rows],
        )
        self._connection.commit()
        return len(rows)

    def write_incident(
        self,
        *,
        event_id: str,
        camera_id: str,
        occurred_at: float,
        incident_type: str,
        score: float | None,
        involved_tracks: Sequence[int],
        evidence_path: str = "",
        metadata: dict | None = None,
    ) -> None:
        tracks = [int(track) for track in involved_tracks]
        self._connection.execute(
            """
            INSERT OR REPLACE INTO incidents (
                event_id, camera_id, occurred_at, incident_type, score,
                participant_count, involved_tracks_json, evidence_path,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event_id),
                str(camera_id),
                float(occurred_at),
                str(incident_type),
                None if score is None else float(score),
                len(tracks),
                json.dumps(tracks, separators=(",", ":")),
                str(evidence_path),
                json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True),
            ),
        )
        self._connection.commit()

    def site_summary(
        self,
        camera_id: str,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
    ) -> SiteSummary:
        where, params = self._passage_filter(camera_id, start_at, end_at)
        rows = self._connection.execute(
            f"""
            SELECT representative_speed_mph, speeding
            FROM vehicle_passages
            WHERE {where}
            """,
            params,
        ).fetchall()
        speeds = np.asarray(
            [row["representative_speed_mph"] for row in rows if row["representative_speed_mph"] is not None],
            dtype=np.float64,
        )
        speeding_rows = [row for row in rows if row["speeding"] is not None]
        speeding_count = sum(int(row["speeding"]) for row in speeding_rows)
        return SiteSummary(
            camera_id=str(camera_id),
            vehicle_count=len(rows),
            valid_speed_count=int(speeds.size),
            average_speed_mph=(float(np.mean(speeds)) if speeds.size else None),
            median_speed_mph=(float(np.median(speeds)) if speeds.size else None),
            percentile_85_speed_mph=(
                float(np.percentile(speeds, 85)) if speeds.size else None
            ),
            speeding_count=speeding_count,
            speeding_rate=(
                speeding_count / len(speeding_rows) if speeding_rows else None
            ),
        )

    def camera_ids(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT DISTINCT camera_id FROM vehicle_passages ORDER BY camera_id"
        ).fetchall()
        return [str(row[0]) for row in rows]

    def speeding_by_weekday(
        self,
        camera_id: str,
        *,
        timezone: str = "America/Puerto_Rico",
        start_at: float | None = None,
        end_at: float | None = None,
    ) -> list[dict]:
        return self._speeding_group(
            camera_id,
            timezone=timezone,
            group="weekday",
            start_at=start_at,
            end_at=end_at,
        )

    def speeding_by_hour(
        self,
        camera_id: str,
        *,
        timezone: str = "America/Puerto_Rico",
        start_at: float | None = None,
        end_at: float | None = None,
    ) -> list[dict]:
        return self._speeding_group(
            camera_id,
            timezone=timezone,
            group="hour",
            start_at=start_at,
            end_at=end_at,
        )

    def _speeding_group(
        self,
        camera_id: str,
        *,
        timezone: str,
        group: str,
        start_at: float | None,
        end_at: float | None,
    ) -> list[dict]:
        where, params = self._passage_filter(camera_id, start_at, end_at)
        rows = self._connection.execute(
            f"""
            SELECT first_seen_at, speeding
            FROM vehicle_passages
            WHERE {where} AND speeding IS NOT NULL
            """,
            params,
        ).fetchall()
        zone = ZoneInfo(timezone)
        grouped: dict[int, list[int]] = {}
        for row in rows:
            dt = datetime.fromtimestamp(float(row["first_seen_at"]), tz=zone)
            key = dt.weekday() if group == "weekday" else dt.hour
            grouped.setdefault(key, []).append(int(row["speeding"]))

        result = []
        for key, values in sorted(grouped.items()):
            result.append(
                {
                    group: key,
                    "qualifying_vehicles": len(values),
                    "speeding_vehicles": sum(values),
                    "speeding_rate": sum(values) / len(values),
                }
            )
        return result

    @staticmethod
    def _passage_filter(
        camera_id: str,
        start_at: float | None,
        end_at: float | None,
    ) -> tuple[str, list[object]]:
        clauses = ["camera_id = ?"]
        params: list[object] = [str(camera_id)]
        if start_at is not None:
            clauses.append("first_seen_at >= ?")
            params.append(float(start_at))
        if end_at is not None:
            clauses.append("first_seen_at < ?")
            params.append(float(end_at))
        return " AND ".join(clauses), params


__all__ = ["AnalyticsStore", "SCHEMA_VERSION", "SiteSummary"]
