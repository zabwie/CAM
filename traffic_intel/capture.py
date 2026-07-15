"""Live capture adapters that decouple RTSP reads from model inference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: np.ndarray
    sequence: int
    capture_timestamp: float
    monotonic_timestamp: float


class LatestFrameCapture:
    """Continuously read a live source and expose only the newest frame.

    YOLO inference can run slower than the camera without making the RTSP socket
    the application's clock. Consumers receive real receipt timestamps and a
    sequence gap indicating how many captured frames were intentionally skipped.
    """

    def __init__(
        self,
        source: str | int,
        *,
        reconnect_seconds: float = 1.0,
    ) -> None:
        self.source = source
        self.reconnect_seconds = max(0.1, float(reconnect_seconds))
        self._condition = threading.Condition()
        self._capture: cv2.VideoCapture | None = None
        self._latest: CapturedFrame | None = None
        self._sequence = 0
        self._closed = False
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="camera-capture",
            daemon=True,
        )
        self._open_capture()
        self._thread.start()

    @property
    def fps(self) -> float:
        cap = self._capture
        return float(cap.get(cv2.CAP_PROP_FPS) or 30.0) if cap is not None else 30.0

    @property
    def width(self) -> int:
        cap = self._capture
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if cap is not None else 0

    @property
    def height(self) -> int:
        cap = self._capture
        return int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if cap is not None else 0

    @property
    def last_error(self) -> str | None:
        return self._error

    def is_opened(self) -> bool:
        return not self._closed and self._capture is not None and self._capture.isOpened()

    def read_packet(
        self,
        *,
        after_sequence: int = 0,
        timeout: float = 3.0,
    ) -> CapturedFrame | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed:
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def release(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        cap = self._capture
        if cap is not None:
            cap.release()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "LatestFrameCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    def _open_capture(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera source: {self._safe_source_label()}")
        self._capture = cap
        self._error = None

    def _reopen_capture(self) -> None:
        cap = self._capture
        if cap is not None:
            cap.release()
        self._capture = None
        while not self._closed:
            try:
                self._open_capture()
                return
            except RuntimeError as exc:
                self._error = str(exc)
                time.sleep(self.reconnect_seconds)

    def _safe_source_label(self) -> str:
        if isinstance(self.source, int):
            return str(self.source)
        value = str(self.source)
        if not value.lower().startswith(("rtsp://", "http://", "https://")):
            return value
        parts = urlsplit(value)
        host = parts.hostname or "camera"
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, f"{host}{port}", parts.path, "", ""))

    def _run(self) -> None:
        while not self._closed:
            cap = self._capture
            if cap is None:
                self._reopen_capture()
                continue
            ok, frame = cap.read()
            received_monotonic = time.monotonic()
            received_wall = time.time()
            if not ok or frame is None:
                self._error = "camera read failed; reconnecting"
                self._reopen_capture()
                continue

            with self._condition:
                self._sequence += 1
                self._latest = CapturedFrame(
                    image=frame,
                    sequence=self._sequence,
                    capture_timestamp=received_wall,
                    monotonic_timestamp=received_monotonic,
                )
                self._condition.notify_all()


__all__ = ["CapturedFrame", "LatestFrameCapture"]
