from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from traffic_intel.analytics import (
    VehiclePassageAggregator,
    VehiclePassageAggregatorConfig,
)
from traffic_intel.analytics_store import AnalyticsStore
from traffic_intel.domain import Detection
from traffic_intel.report import build_pilot_report


def _det(
    track_id: int,
    timestamp: float,
    speed: float,
    *,
    confidence: float = 0.9,
    vision_state: str = "DAY",
) -> Detection:
    return Detection(
        frame=int(timestamp * 30) + 1,
        track_id=track_id,
        class_name="car",
        confidence=0.9,
        bbox=(0, 0, 80, 40),
        speed=speed,
        speed_valid=True,
        measurement_confidence=confidence,
        capture_timestamp=timestamp,
        monotonic_timestamp=timestamp,
        vision_state=vision_state,
    )


def test_one_vehicle_counts_once_regardless_of_frame_count() -> None:
    aggregator = VehiclePassageAggregator(
        camera_id="urban",
        speed_limit_mph=40,
        config=VehiclePassageAggregatorConfig(
            finalization_gap_seconds=1.0,
            min_valid_speed_samples=2,
            min_measurement_confidence=0.5,
        ),
    )

    for index in range(20):
        timestamp = index * 0.1
        aggregator.update(
            [_det(1, timestamp, 30.0 + (index % 3))],
            capture_timestamp=timestamp,
            monotonic_timestamp=timestamp,
            vision_state="DAY",
        )
    for index in range(3):
        timestamp = 0.5 + index * 0.1
        aggregator.update(
            [_det(2, timestamp, 60.0)],
            capture_timestamp=timestamp,
            monotonic_timestamp=timestamp,
            vision_state="DAY",
        )

    passages = aggregator.flush()
    assert len(passages) == 2
    by_track = {passage.canonical_track_id: passage for passage in passages}
    assert by_track[1].representative_speed_mph == pytest.approx(31.0)
    assert by_track[2].representative_speed_mph == pytest.approx(60.0)
    assert by_track[1].speeding is False
    assert by_track[2].speeding is True


def test_store_reports_normalized_speeding_rates_and_p85(tmp_path: Path) -> None:
    zone = ZoneInfo("America/Puerto_Rico")
    monday = datetime(2026, 7, 13, 9, tzinfo=zone).timestamp()
    tuesday = datetime(2026, 7, 14, 9, tzinfo=zone).timestamp()

    aggregator = VehiclePassageAggregator(
        camera_id="rural",
        municipality="Demo",
        location_id="PR-1 KM 10",
        speed_limit_mph=40,
        config=VehiclePassageAggregatorConfig(min_valid_speed_samples=1),
    )
    passages = []
    for track_id, timestamp, speed in [
        (1, monday, 30.0),
        (2, monday + 1, 50.0),
        (3, tuesday, 55.0),
        (4, tuesday + 1, 60.0),
    ]:
        aggregator.update(
            [_det(track_id, timestamp, speed)],
            capture_timestamp=timestamp,
            monotonic_timestamp=timestamp,
        )
        passages.extend(aggregator.flush())

    db = tmp_path / "analytics.db"
    with AnalyticsStore(db) as store:
        store.write_passages(passages)
        summary = store.site_summary("rural")
        weekdays = store.speeding_by_weekday("rural")
        report = build_pilot_report(store)

    assert summary.vehicle_count == 4
    assert summary.valid_speed_count == 4
    assert summary.average_speed_mph == pytest.approx(48.75)
    assert summary.speeding_rate == pytest.approx(0.75)
    assert summary.percentile_85_speed_mph is not None
    by_day = {row["weekday"]: row for row in weekdays}
    assert by_day[0]["speeding_rate"] == pytest.approx(0.5)
    assert by_day[1]["speeding_rate"] == pytest.approx(1.0)
    assert report["highest_speeding_weekday"]["weekday_name"] == "Tuesday"
    assert report["highest_speeding_site"]["camera_id"] == "rural"
