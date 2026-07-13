"""Notification delivery adapters.

The core system persists notifications before delivery.  This module provides a
small webhook dispatcher; failed delivery never deletes the incident or evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from traffic_intel.ops.store import IncidentStore


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    notification_id: int
    status: str
    detail: str = ""


class WebhookDispatcher:
    def __init__(self, store: IncidentStore, *, timeout_seconds: float = 5.0) -> None:
        self.store = store
        self.timeout_seconds = max(0.5, float(timeout_seconds))

    def dispatch_queued(self, *, limit: int = 50) -> list[DeliveryResult]:
        results: list[DeliveryResult] = []
        for item in self.store.queued_notifications(limit=limit):
            if item.get("channel") != "webhook":
                continue
            notification_id = int(item["notification_id"])
            destination = str(item.get("destination") or "").strip()
            if not destination.lower().startswith(("https://", "http://")):
                self.store.mark_notification(notification_id, status="failed")
                results.append(DeliveryResult(notification_id, "failed", "invalid webhook destination"))
                continue
            body = json.dumps(item.get("payload") or {}).encode("utf-8")
            req = request.Request(
                destination,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "traffic-intel/0.12"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    code = int(getattr(response, "status", 200))
                if 200 <= code < 300:
                    self.store.mark_notification(notification_id, status="sent")
                    results.append(DeliveryResult(notification_id, "sent", f"HTTP {code}"))
                else:
                    self.store.mark_notification(notification_id, status="failed")
                    results.append(DeliveryResult(notification_id, "failed", f"HTTP {code}"))
            except (error.URLError, TimeoutError, OSError) as exc:
                self.store.mark_notification(notification_id, status="failed")
                results.append(DeliveryResult(notification_id, "failed", str(exc)))
        return results
