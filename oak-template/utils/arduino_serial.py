# utils/arduino_serial.py
"""
Bidirectional serial communication with Arduino Mega.

Writer thread: sends CMD:xxx\n commands from a queue.
Reader thread: parses STATUS:* and BANK:* messages, updates shared state.
MOCK mode: simulates serial for testing without hardware.
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Callable

try:
    import serial
except ImportError:
    serial = None  # type: ignore


@dataclass
class ArduinoState:
    """Thread-safe snapshot of Arduino system state."""

    authorized: bool = False
    mode: str = "UNKNOWN"               # FREE / ASSIST / UNKNOWN
    calibrated: bool = True             # Assume calibrated until told otherwise
    sensor_ok: bool = True              # Assume sensor OK unless error
    locked_left: bool = False
    locked_right: bool = False
    ready: bool = False                 # Not ready until authorized + auth sequence done
    last_bank_result: str = ""
    last_status: str = ""
    connected: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self) -> dict:
        """Return a plain dict copy for safe cross-thread reads."""
        with self._lock:
            return {
                "authorized": self.authorized,
                "mode": self.mode,
                "calibrated": self.calibrated,
                "sensor_ok": self.sensor_ok,
                "locked_left": self.locked_left,
                "locked_right": self.locked_right,
                "ready": self.ready,
                "last_bank_result": self.last_bank_result,
                "last_status": self.last_status,
                "connected": self.connected,
            }

    def can_navigate(self) -> bool:
        """Return True if it's safe to send navigation commands."""
        with self._lock:
            return (
                self.authorized
                and self.ready
                and self.calibrated
                and self.sensor_ok
                and self.connected
            )


class ArduinoSerial:
    """
    Manages bidirectional serial communication with Arduino.

    Usage:
        ard = ArduinoSerial(port="COM4", baud=9600)
        ard.start()
        ...
        ard.send_command("GO:CENTER")   # Sends "CMD:GO:CENTER\n"
        state = ard.state.snapshot()
        ...
        ard.stop()
    """

    def __init__(
        self,
        port: str = "MOCK",
        baud: int = 9600,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.port = port
        self.baud = baud
        self.mock = port.upper() == "MOCK"
        self.state = ArduinoState()
        self._send_queue: queue.Queue = queue.Queue()
        self._running = False
        self._ser: Optional[object] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._on_status = on_status   # Optional callback for status changes

    # ── Public API ─────────────────────────────────────────

    def start(self) -> None:
        """Start writer + reader threads."""
        self._running = True

        if self.mock:
            print("[Serial] MOCK mode — no Arduino connected")
            self.state.update(connected=True)
            self._writer_thread = threading.Thread(
                target=self._mock_writer, daemon=True
            )
            self._writer_thread.start()

            # In mock mode, simulate authorization after 3 seconds
            self._mock_auth_thread = threading.Thread(
                target=self._mock_auth_sequence, daemon=True
            )
            self._mock_auth_thread.start()
            return

        if serial is None:
            print("[Serial] ERROR: pyserial not installed. Run: pip install pyserial")
            self.state.update(connected=False)
            return

        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
            time.sleep(0.5)  # Arduino reset delay
            self.state.update(connected=True)
            print(f"[Serial] Connected: {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[Serial] Could not open {self.port}: {e}")
            self.state.update(connected=False)
            return

        self._writer_thread = threading.Thread(
            target=self._serial_writer, daemon=True
        )
        self._reader_thread = threading.Thread(
            target=self._serial_reader, daemon=True
        )
        self._writer_thread.start()
        self._reader_thread.start()

    def stop(self) -> None:
        """Gracefully stop threads."""
        self._running = False
        self._send_queue.put(None)  # Unblock writer
        if self._ser and hasattr(self._ser, "close"):
            try:
                self._ser.close()
            except Exception:
                pass

    def send_command(self, cmd: str) -> None:
        """
        Queue a command to send to Arduino.
        Prepends CMD: if not already present.
        Example: send_command("GO:CENTER") → sends "CMD:GO:CENTER\\n"
        """
        if not cmd:
            return
        full_cmd = cmd if cmd.startswith("CMD:") else f"CMD:{cmd}"
        self._send_queue.put(full_cmd)

    # ── Writer Threads ─────────────────────────────────────

    def _serial_writer(self) -> None:
        """Writes queued commands to the real serial port."""
        while self._running:
            try:
                msg = self._send_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:
                break
            try:
                self._ser.write((msg + "\n").encode("utf-8"))
                print(f"[Serial → Arduino] {msg}")
            except Exception as e:
                print(f"[Serial] Write error: {e}")
                self.state.update(connected=False)

    def _mock_writer(self) -> None:
        """Prints queued commands to console (MOCK mode)."""
        while self._running:
            try:
                msg = self._send_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:
                break
            print(f"[Serial → Arduino] (MOCK) {msg}")

    # ── Reader Thread ──────────────────────────────────────

    def _serial_reader(self) -> None:
        """Reads lines from Arduino and parses STATUS/BANK messages."""
        buffer = ""
        while self._running:
            try:
                if self._ser is None or not self._ser.is_open:
                    time.sleep(0.5)
                    continue
                raw = self._ser.read(256)
                if not raw:
                    continue
                buffer += raw.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip("\r\n \t")
                    if line:
                        self._parse_message(line)
            except Exception as e:
                print(f"[Serial] Read error: {e}")
                time.sleep(0.5)

    def _parse_message(self, msg: str) -> None:
        """Parse a single message from Arduino and update state."""
        msg = msg.strip()
        if not msg:
            return

        print(f"[Arduino → Pi] {msg}")
        self.state.update(last_status=msg)

        # ── STATUS messages ───────────────────────────────
        if msg.startswith("STATUS:"):
            status = msg[7:]  # Strip "STATUS:"

            if status == "AUTHORIZED":
                self.state.update(authorized=True, ready=False)
                # ready becomes True when auth sequence finishes
                # (we'll get ASSIST or FREE after the sequence)

            elif status == "UNAUTHORIZED":
                self.state.update(
                    authorized=False,
                    ready=False,
                    mode="UNKNOWN",
                    locked_left=False,
                    locked_right=False,
                )

            elif status == "FREE":
                self.state.update(mode="FREE", ready=True)

            elif status == "ASSIST":
                self.state.update(mode="ASSIST", ready=True)

            elif status == "REACHED":
                pass  # Informational — steering reached target

            elif status == "AT_TARGET":
                pass  # Already at target position

            elif status == "STOPPED":
                pass  # Emergency stop executed

            elif status == "NOT_READY":
                self.state.update(ready=False)

            elif status == "SENSOR_ERROR":
                self.state.update(sensor_ok=False)

            elif status == "NOT_CALIBRATED":
                self.state.update(calibrated=False)

            elif status == "LOCKED_LEFT":
                self.state.update(locked_left=True)

            elif status == "LOCKED_RIGHT":
                self.state.update(locked_right=True)

            elif status == "UNLOCKED":
                self.state.update(locked_left=False, locked_right=False)

            elif status.startswith("MOVING:"):
                pass  # Informational — steering in progress

            else:
                print(f"[Serial] Unknown STATUS: {status}")

        # ── BANK messages ─────────────────────────────────
        elif msg.startswith("BANK:"):
            result = msg[5:]
            if result == "REMOVED":
                self.state.update(last_bank_result="")
            else:
                self.state.update(last_bank_result=result)
                print(f"[Banknote] Detected: {result}")

        if self._on_status:
            try:
                self._on_status(msg)
            except Exception:
                pass

    # ── Mock Helpers ───────────────────────────────────────

    def _mock_auth_sequence(self) -> None:
        """Simulate Arduino authorization after a delay for MOCK testing."""
        time.sleep(3.0)
        if not self._running:
            return
        print("[MOCK] Simulating STATUS:AUTHORIZED")
        self._parse_message("STATUS:AUTHORIZED")
        time.sleep(1.0)
        print("[MOCK] Simulating STATUS:FREE (auth sequence complete)")
        self._parse_message("STATUS:FREE")
