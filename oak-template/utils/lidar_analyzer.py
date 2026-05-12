"""
RPLIDAR-based proximity scan analyzer with RPLIDAR C1 (rplidarc1) + legacy fallback.

Angle convention:
  - 0° is forward
  - positive angles are to the left, negative to the right
  - angles are normalized to [-180, 180]

Backends:
  - ``c1``: SLAMTEC RPLIDAR C1 via ``rplidarc1`` (460800 baud default).
  - ``legacy``: A-series style devices via ``rplidar`` (Roboticia fork / PyPI).
  - ``auto``: try C1 first when ``rplidarc1`` is installed, then legacy.
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import serial


# ── Stdout filter for rplidarc1 sync noise ──────────────────────────────────
# The rplidarc1 library prints sync-failure diagnostics directly to stdout
# (not via Python's logging).  We suppress only the known noisy patterns so
# real errors are never swallowed.
_LIDAR_NOISE_PATTERNS = (
    "S bit verification failed",
    "C bit verification failed",
    "Verification bytes not matching",
    "Angles should not be >",
    "calculated angle",
)


class _FilteredStdout(io.TextIOBase):
    """Proxy that drops known rplidarc1 sync-noise lines."""

    def __init__(self, wrapped: io.TextIOBase) -> None:
        self._wrapped = wrapped

    def write(self, s: str) -> int:  # type: ignore[override]
        if any(pat in s for pat in _LIDAR_NOISE_PATTERNS):
            return len(s)           # silently swallow the noise
        return self._wrapped.write(s)

    def flush(self) -> None:
        self._wrapped.flush()


@contextmanager
def _suppress_lidar_noise():
    """Context manager: replace sys.stdout with a filtered proxy."""
    original = sys.stdout
    sys.stdout = _FilteredStdout(original)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = original

try:
    from rplidarc1.scanner import RPLidar as RPLidarC1
    RPLIDARC1_AVAILABLE = True
except ImportError:
    RPLidarC1 = None  # type: ignore[misc, assignment]
    RPLIDARC1_AVAILABLE = False

try:
    from rplidar import RPLidar as RPLidarLegacy
    RPLIDAR_LEGACY_AVAILABLE = True
except ImportError:
    RPLidarLegacy = None  # type: ignore[misc, assignment]
    RPLIDAR_LEGACY_AVAILABLE = False

# Defaults (overridden per-instance from WalkerConfig in main)
DEFAULT_SAFETY_MM = 600.0
DEFAULT_SIDE_ESCAPE_MM = 800.0
DEFAULT_SCAN_TIMEOUT_S = 0.5
DEFAULT_FRONT_ARC_DEG = 30.0
DEFAULT_SIDE_ARC_START_DEG = 30.0
DEFAULT_SIDE_ARC_END_DEG = 90.0
DEFAULT_C1_BAUD = 460800
DEFAULT_LEGACY_BAUD = 115200


@dataclass
class LidarScan:
    front_clear: bool
    front_min_mm: float
    left_min_mm: float
    right_min_mm: float
    side_escape_left: bool
    side_escape_right: bool
    raw_angles: np.ndarray
    raw_distances: np.ndarray
    timestamp: float


class LidarAnalyzer:
    """
    Continuously reads scans in a background daemon thread.

    - ``mock=True`` or ``LIDAR_PORT=MOCK``: synthetic clear scans every 0.1s.
    - Real hardware: uses ``backend`` (``auto`` / ``c1`` / ``legacy``).
      On total failure, the thread stops and ``latest_scan`` stays stale
      (no silent switch to mock).
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        mock: bool = False,
        *,
        backend: str = "auto",
        c1_baud: Optional[int] = None,
        legacy_baud: Optional[int] = None,
        safety_mm: float = DEFAULT_SAFETY_MM,
        side_escape_mm: float = DEFAULT_SIDE_ESCAPE_MM,
        scan_timeout_s: float = DEFAULT_SCAN_TIMEOUT_S,
        front_arc_deg: float = DEFAULT_FRONT_ARC_DEG,
        side_arc_start_deg: float = DEFAULT_SIDE_ARC_START_DEG,
        side_arc_end_deg: float = DEFAULT_SIDE_ARC_END_DEG,
    ):
        self.port = str(port)
        b = str(backend).lower().strip()
        if b not in ("auto", "c1", "legacy"):
            b = "auto"
        self._backend = b
        self.mock = bool(mock) or self.port.upper() == "MOCK"

        self._baud_c1 = int(c1_baud) if c1_baud is not None else DEFAULT_C1_BAUD
        self._baud_legacy = (
            int(legacy_baud) if legacy_baud is not None else DEFAULT_LEGACY_BAUD
        )

        self._safety_mm = float(safety_mm)
        self._side_escape_mm = float(side_escape_mm)
        self._scan_timeout_s = float(scan_timeout_s)
        fa = float(front_arc_deg)
        s0 = float(side_arc_start_deg)
        s1 = float(side_arc_end_deg)
        self._front_arc = (-fa, fa)
        self._left_arc = (s0, s1)
        self._right_arc = (-s1, -s0)

        self._lock = threading.Lock()
        self._latest: Optional[LidarScan] = None

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None

    @property
    def latest_scan(self) -> Optional[LidarScan]:
        with self._lock:
            scan = self._latest
        if scan is None:
            return None
        if (time.time() - scan.timestamp) > self._scan_timeout_s:
            return None
        return scan

    def _run(self) -> None:
        if self.mock:
            if not RPLIDARC1_AVAILABLE and not RPLIDAR_LEGACY_AVAILABLE:
                print(
                    "[LiDAR] MOCK mode (no rplidarc1/rplidar); "
                    "synthetic scans only."
                )
            self._run_mock()
            return

        ok = False
        if self._backend == "c1":
            ok = self._try_run_c1()
        elif self._backend == "legacy":
            ok = self._try_run_legacy()
        else:
            if RPLIDARC1_AVAILABLE:
                print("[LiDAR] backend=auto: trying rplidarc1 (RPLIDAR C1)...")
                ok = self._try_run_c1()
            # Only try legacy if rplidarc1 is NOT available.
            # Legacy driver uses 115200 baud and will always fail against a C1
            # (which requires 460800), producing 'Descriptor length mismatch'.
            if not ok and not RPLIDARC1_AVAILABLE and RPLIDAR_LEGACY_AVAILABLE:
                print("[LiDAR] C1 driver unavailable; trying legacy rplidar...")
                ok = self._try_run_legacy()

        if not ok:
            if self._backend == "auto" and not RPLIDARC1_AVAILABLE:
                print(
                    "[LiDAR] rplidarc1 not installed. "
                    "Install: pip install rplidarc1  (recommended for RPLIDAR C1)"
                )
            if not RPLIDAR_LEGACY_AVAILABLE:
                print(
                    "[LiDAR] legacy rplidar not installed. "
                    "Install: pip install rplidar-roboticia  (A-series / some devices)"
                )
            print(
                "[LiDAR] No working driver; latest_scan will stay stale. "
                "Set LIDAR_PORT=MOCK to test without hardware."
            )

    def _try_run_c1(self) -> bool:
        if not RPLIDARC1_AVAILABLE or RPLidarC1 is None:
            return False
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f"[LiDAR] rplidarc1: port={self.port!r} baud={self._baud_c1}"
                    + (f" (attempt {attempt}/{max_attempts})" if attempt > 1 else "")
                )
                asyncio.run(self._c1_async_main())
                return True
            except Exception as e:
                print(f"[LiDAR] rplidarc1 error (attempt {attempt}): {e!r}")
                if attempt < max_attempts:
                    print("[LiDAR] Retrying in 2 s...")
                    time.sleep(2.0)
        return False

    async def _c1_async_main(self) -> None:
        assert RPLidarC1 is not None

        # ── Flush stale bytes and wait for motor spin-up ─────────────────
        # On Raspberry Pi the C1 motor takes 2-3 s to reach full speed.
        # We open the port early, drain all stale bytes for the full wait
        # period, then hand off to RPLidarC1.
        SPINUP_S = 3.0
        try:
            with serial.Serial(self.port, self._baud_c1, timeout=0.05) as _ser:
                _ser.reset_input_buffer()
                print(
                    f"[LiDAR] Flushing {self.port!r} and waiting "
                    f"{SPINUP_S:.0f} s for C1 motor spin-up..."
                )
                deadline = time.monotonic() + SPINUP_S
                while time.monotonic() < deadline:
                    _ser.read(256)   # drain any bytes arriving during spin-up
                    await asyncio.sleep(0.05)
        except Exception as _e:
            print(f"[LiDAR] Flush/spinup skipped ({_e!r}) — continuing anyway")
            await asyncio.sleep(SPINUP_S)


        lidar = RPLidarC1(self.port, self._baud_c1)
        scan_coro = lidar.simple_scan()

        async def consume() -> None:
            try:
                while not self._stop_evt.is_set():
                    angles: list[float] = []
                    distances: list[float] = []
                    t_end = time.monotonic() + 0.12
                    while time.monotonic() < t_end and not self._stop_evt.is_set():
                        try:
                            item = await asyncio.wait_for(
                                lidar.output_queue.get(), timeout=0.02
                            )
                        except asyncio.TimeoutError:
                            continue
                        d = item.get("d_mm")
                        if d is None:
                            continue
                        a_deg = float(item["a_deg"])
                        if a_deg > 360.0 or a_deg < -360.0:
                            continue  # Ignore out-of-range angles silently
                        angles.append(a_deg)
                        distances.append(float(d))
                    if angles and distances:
                        a = np.asarray(angles, dtype=np.float32)
                        d = np.asarray(distances, dtype=np.float32)
                        self._publish_scan(self._process_scan(a, d))
            finally:
                lidar.stop_event.set()

        with _suppress_lidar_noise():
            try:
                await asyncio.gather(scan_coro, consume())
            finally:
                try:
                    lidar.shutdown()
                except Exception as e:
                    print(f"[LiDAR] rplidarc1 shutdown: {e!r}")

    def _try_run_legacy(self) -> bool:
        if not RPLIDAR_LEGACY_AVAILABLE or RPLidarLegacy is None:
            return False
        lidar = None
        try:
            print(
                f"[LiDAR] legacy rplidar: port={self.port!r} baud={self._baud_legacy}"
            )
            lidar = RPLidarLegacy(self.port, baudrate=self._baud_legacy)
            for scan in lidar.iter_scans():
                if self._stop_evt.is_set():
                    break
                if not scan:
                    continue
                angles = np.array([m[1] for m in scan], dtype=np.float32)
                distances = np.array([m[2] for m in scan], dtype=np.float32)
                self._publish_scan(self._process_scan(angles, distances))
            return True
        except Exception as e:
            print(f"[LiDAR] legacy rplidar error: {e!r}")
            return False
        finally:
            if lidar is not None:
                try:
                    lidar.stop()
                except Exception:
                    pass
                try:
                    lidar.stop_motor()
                except Exception:
                    pass
                try:
                    lidar.disconnect()
                except Exception:
                    pass

    def _run_mock(self) -> None:
        angles = np.linspace(-180.0, 180.0, 360, endpoint=False, dtype=np.float32)
        distances = np.full_like(angles, 2000.0, dtype=np.float32)
        while not self._stop_evt.is_set():
            self._publish_scan(self._process_scan(angles, distances))
            time.sleep(0.1)

    def _publish_scan(self, scan: LidarScan) -> None:
        with self._lock:
            self._latest = scan

    @staticmethod
    def _angle_in_arc(a: np.ndarray, arc: tuple[float, float]) -> np.ndarray:
        lo, hi = float(arc[0]), float(arc[1])
        return (a >= lo) & (a <= hi)

    def _process_scan(
        self, angles_deg: np.ndarray, distances_mm: np.ndarray
    ) -> LidarScan:
        angles = np.asarray(angles_deg, dtype=np.float32).copy()
        dists = np.asarray(distances_mm, dtype=np.float32).copy()

        angles = np.mod(angles, 360.0)
        angles = np.where(angles > 180.0, angles - 360.0, angles)

        valid = (
            (dists > 50.0)
            & (dists < 8000.0)
            & np.isfinite(dists)
            & np.isfinite(angles)
        )
        angles_v = angles[valid]
        dists_v = dists[valid]

        def arc_min(arc: tuple[float, float]) -> float:
            if angles_v.size == 0:
                return float("inf")
            m = self._angle_in_arc(angles_v, arc)
            if not np.any(m):
                return float("inf")
            return float(np.min(dists_v[m]))

        front_min = arc_min(self._front_arc)
        left_min = arc_min(self._left_arc)
        right_min = arc_min(self._right_arc)

        front_clear = front_min > self._safety_mm
        side_escape_left = left_min > self._side_escape_mm
        side_escape_right = right_min > self._side_escape_mm

        return LidarScan(
            front_clear=bool(front_clear),
            front_min_mm=float(front_min),
            left_min_mm=float(left_min),
            right_min_mm=float(right_min),
            side_escape_left=bool(side_escape_left),
            side_escape_right=bool(side_escape_right),
            raw_angles=angles_v,
            raw_distances=dists_v,
            timestamp=time.time(),
        )
