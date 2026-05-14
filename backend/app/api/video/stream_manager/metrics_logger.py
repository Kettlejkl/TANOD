import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Lightweight per-camera metrics logger.
    Accumulates frame-level stats and periodically prints summaries.
    """

    def __init__(self, log_interval: int = 150):
        """
        Args:
            log_interval: Print a summary every N frames logged.
        """
        self.log_interval = log_interval
        self._cameras: dict[str, dict] = {}
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_camera(self, camera_id: str):
        if camera_id not in self._cameras:
            self._cameras[camera_id] = {
                "frame_count":       0,
                "fps_samples":       deque(maxlen=60),
                "detection_samples": deque(maxlen=60),
                "tracking_samples":  deque(maxlen=60),
                "latency_samples":   deque(maxlen=60),
                "confidence_samples":deque(maxlen=60),
                "occupancy_samples": deque(maxlen=60),
                "last_log_time":     time.time(),
            }

    def _avg(self, dq: deque) -> float:
        return sum(dq) / len(dq) if dq else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_frame(
        self,
        camera_id: str,
        fps: float = 0.0,
        detection_count: int = 0,
        tracking_count: int = 0,
        latency: float = 0.0,
        avg_confidence: float = 0.0,
        occupancy: int = 0,
    ):
        """Record metrics for a single processed frame."""
        self._ensure_camera(camera_id)
        c = self._cameras[camera_id]

        c["frame_count"]        += 1
        c["fps_samples"].append(fps)
        c["detection_samples"].append(detection_count)
        c["tracking_samples"].append(tracking_count)
        c["latency_samples"].append(latency * 1000)   # store as ms
        c["confidence_samples"].append(avg_confidence)
        c["occupancy_samples"].append(occupancy)

        if c["frame_count"] % self.log_interval == 0:
            self._print_summary(camera_id)

    def _print_summary(self, camera_id: str):
        c   = self._cameras[camera_id]
        fps = self._avg(c["fps_samples"])
        det = self._avg(c["detection_samples"])
        trk = self._avg(c["tracking_samples"])
        lat = self._avg(c["latency_samples"])
        cof = self._avg(c["confidence_samples"])
        occ = self._avg(c["occupancy_samples"])

        print(
            f"[Metrics] {camera_id} | "
            f"frame={c['frame_count']:,} | "
            f"fps={fps:.1f} | "
            f"det={det:.1f} | "
            f"track={trk:.1f} | "
            f"latency={lat:.1f}ms | "
            f"conf={cof:.2f} | "
            f"occupancy={occ:.1f}"
        )
        c["last_log_time"] = time.time()

    def get_summary(self, camera_id: str) -> dict:
        """Return the latest averaged metrics for a camera."""
        self._ensure_camera(camera_id)
        c = self._cameras[camera_id]
        return {
            "camera_id":       camera_id,
            "frame_count":     c["frame_count"],
            "avg_fps":         self._avg(c["fps_samples"]),
            "avg_detections":  self._avg(c["detection_samples"]),
            "avg_tracking":    self._avg(c["tracking_samples"]),
            "avg_latency_ms":  self._avg(c["latency_samples"]),
            "avg_confidence":  self._avg(c["confidence_samples"]),
            "avg_occupancy":   self._avg(c["occupancy_samples"]),
        }

    def get_all_summaries(self) -> list[dict]:
        return [self.get_summary(cid) for cid in self._cameras]

    def finalize(self):
        """Print a final summary for all cameras on shutdown."""
        elapsed = time.time() - self._start_time
        print(f"\n[Metrics] ── Final report (uptime {elapsed:.0f}s) ──")
        for camera_id in self._cameras:
            self._print_summary(camera_id)
        print("[Metrics] ─────────────────────────────────────────\n")

    def reset(self, camera_id: str | None = None):
        if camera_id is None:
            self._cameras.clear()
        elif camera_id in self._cameras:
            del self._cameras[camera_id]