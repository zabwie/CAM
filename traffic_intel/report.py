"""Generate a municipality-readable pilot summary from the SQLite ledger."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .analytics_store import AnalyticsStore

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def build_pilot_report(
    store: AnalyticsStore,
    *,
    timezone: str = "America/Puerto_Rico",
) -> dict:
    sites = []
    weekday_totals: dict[int, dict[str, int]] = {}
    hour_totals: dict[int, dict[str, int]] = {}

    for camera_id in store.camera_ids():
        summary = asdict(store.site_summary(camera_id))
        weekdays = store.speeding_by_weekday(camera_id, timezone=timezone)
        hours = store.speeding_by_hour(camera_id, timezone=timezone)
        summary["speeding_by_weekday"] = [
            {**row, "weekday_name": WEEKDAYS[int(row["weekday"])]}
            for row in weekdays
        ]
        summary["speeding_by_hour"] = hours
        sites.append(summary)

        for row in weekdays:
            bucket = weekday_totals.setdefault(
                int(row["weekday"]),
                {"qualifying_vehicles": 0, "speeding_vehicles": 0},
            )
            bucket["qualifying_vehicles"] += int(row["qualifying_vehicles"])
            bucket["speeding_vehicles"] += int(row["speeding_vehicles"])
        for row in hours:
            bucket = hour_totals.setdefault(
                int(row["hour"]),
                {"qualifying_vehicles": 0, "speeding_vehicles": 0},
            )
            bucket["qualifying_vehicles"] += int(row["qualifying_vehicles"])
            bucket["speeding_vehicles"] += int(row["speeding_vehicles"])

    eligible_sites = [site for site in sites if site["speeding_rate"] is not None]
    hottest_site = (
        max(eligible_sites, key=lambda site: site["speeding_rate"])
        if eligible_sites
        else None
    )

    def hottest_bucket(totals: dict[int, dict[str, int]], key_name: str) -> dict | None:
        rows = []
        for key, counts in totals.items():
            qualifying = counts["qualifying_vehicles"]
            rows.append(
                {
                    key_name: key,
                    **counts,
                    "speeding_rate": (
                        counts["speeding_vehicles"] / qualifying
                        if qualifying
                        else None
                    ),
                }
            )
        valid = [row for row in rows if row["speeding_rate"] is not None]
        return max(valid, key=lambda row: row["speeding_rate"]) if valid else None

    hottest_weekday = hottest_bucket(weekday_totals, "weekday")
    if hottest_weekday is not None:
        hottest_weekday["weekday_name"] = WEEKDAYS[hottest_weekday["weekday"]]

    return {
        "scope_note": "Comparisons apply only to monitored pilot locations.",
        "timezone": timezone,
        "sites": sites,
        "highest_speeding_site": hottest_site,
        "highest_speeding_weekday": hottest_weekday,
        "highest_speeding_hour": hottest_bucket(hour_totals, "hour"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CAM pilot analytics report")
    parser.add_argument("--db", default="analytics.db")
    parser.add_argument("--timezone", default="America/Puerto_Rico")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    with AnalyticsStore(args.db) as store:
        report = build_pilot_report(store, timezone=args.timezone)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
