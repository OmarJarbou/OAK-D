"""
lidar_scanner.py — Thin thread-safe wrapper around RPLIDAR C1.

The background thread continuously reads scans; the main loop calls
get_scan() to get a non-blocking snapshot of the latest full rotation.

Angle convention from the sensor:
  0° = direction of the USB/power connector
  Angles increase clockwise when viewed from above.
  Use FRONT_HEADING in navigator.py to compensate for mounting direction.
"""

import threading
from rplidar import RPLidar

# ── Hardware defaults ──────────────────────────────────────────────────
DEFAULT_PORT     = '/dev/ttyUSB0'   # RPLIDAR C1 on Pi  (adjust if needed)
DEFAULT_BAUD     = 460800           # C1 factory default
MIN_QUALITY      = 10               # 0-15; discard low-confidence points
MIN_DIST_MM      = 100              # ignore returns < 10 cm (self-reflection)


class LidarScanner:
    """Start with .start(), read with .get_scan(), stop with .stop()."""

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUD):
        self._port    = port
        self._baud    = baudrate
        self._lidar   = None
        self._scan    = {}           # {angle_int (0-359): distance_mm}
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    # ------------------------------------------------------------------ #

    def start(self):
        self._lidar   = RPLidar(self._port, baudrate=self._baud)
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            for scan in self._lidar.iter_scans(min_len=5):
                if not self._running:
                    break
                fresh = {}
                for quality, angle, distance in scan:
                    if quality >= MIN_QUALITY and distance >= MIN_DIST_MM:
                        fresh[int(angle) % 360] = distance
                with self._lock:
                    self._scan = fresh
        except Exception as exc:
            print(f"[LIDAR] Scan loop error: {exc}")
        finally:
            self._cleanup()

    def get_scan(self) -> dict:
        """Return a snapshot of the latest complete rotation {angle: dist_mm}."""
        with self._lock:
            return dict(self._scan)

    def stop(self):
        self._running = False
        self._cleanup()

    def _cleanup(self):
        try:
            if self._lidar:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
                self._lidar = None
        except Exception:
            pass
