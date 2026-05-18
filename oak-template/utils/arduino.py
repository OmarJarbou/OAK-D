# utils/arduino.py
"""
Smart Walker v4.0 — Arduino Serial
الإصلاح الرئيسي: نتابع حالة is_moving.
ما نبعث GO جديد حتى نستقبل REACHED / AT_TARGET / FREE.
"""

import threading
import time
from typing import Optional


class ArduinoState:
    """الحالة الحالية للأردوينو — thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.authorized: bool = False
        self.ready: bool = False
        self.mode: str = "FREE"
        self.is_moving: bool = False      # ← هل الـ stepper شغال الحين؟
        self._move_start: float = 0.0
        self.MOVE_TIMEOUT_S: float = 4.0  # حماية: إذا ما رد الأردوينو بعد 4 ثواني

    def update(self, msg: str):
        with self._lock:
            if msg == "STATUS:AUTHORIZED":
                self.authorized = True
                self.ready = False
                self.is_moving = False
                print("[Arduino] Authorized — waiting for auth sequence...")

            elif msg == "STATUS:UNAUTHORIZED":
                self.authorized = False
                self.ready = False
                self.is_moving = False
                print("[Arduino] Unauthorized")

            elif msg == "STATUS:FREE":
                self.mode = "FREE"
                self.ready = True
                self.is_moving = False    # ← الـ stepper وقف، عاد لـ FREE

            elif msg == "STATUS:ASSIST":
                self.mode = "ASSIST"
                # is_moving بيتعدل لما يبدأ STATUS:MOVING

            elif msg.startswith("STATUS:MOVING:"):
                self.is_moving = True     # ← الـ stepper بدأ
                self._move_start = time.time()
                pos = msg.replace("STATUS:MOVING:", "")
                print(f"[Arduino] Moving → {pos}")

            elif msg in ("STATUS:REACHED", "STATUS:AT_TARGET"):
                self.is_moving = False    # ← وصل الهدف
                print(f"[Arduino] {msg}")

            elif msg == "STATUS:STOPPED":
                self.is_moving = False
                print("[Arduino] Stopped")

            elif msg.startswith("STATUS:LOCKED"):
                self.is_moving = False    # ← وصل الحد الميكانيكي
                print(f"[Arduino] ⚠️ {msg}")

            elif msg == "STATUS:SENSOR_ERROR":
                self.is_moving = False
                print("[Arduino] ⚠️ Sensor error!")

            elif msg.startswith("BANK:"):
                print(f"[Arduino] Banknote: {msg}")

            else:
                print(f"[Arduino] {msg}")

    def check_move_timeout(self):
        """استدعيها من الـ main loop — تحرر is_moving لو انتهى الـ timeout."""
        with self._lock:
            if (
                self.is_moving
                and (time.time() - self._move_start) > self.MOVE_TIMEOUT_S
            ):
                print("[Arduino] ⚠️ Move timeout — assuming done")
                self.is_moving = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "authorized": self.authorized,
                "ready": self.ready,
                "mode": self.mode,
                "is_moving": self.is_moving,
            }


class ArduinoSerial:
    """
    Thread يقرأ من الأردوينو باستمرار.
    MOCK: يشتغل بدون serial حقيقي.
    """

    def __init__(self, port: str = "MOCK", baud: int = 9600):
        self.port = port
        self.baud = baud
        self.mock = port.upper() == "MOCK"
        self.state = ArduinoState()
        self._ser = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_sent: str = ""
        self._last_sent_time: float = 0.0
        self._send_lock = threading.Lock()

        self.REPEAT_INTERVAL_S: float = 3.0  # إعادة إرسال نفس الأمر بعد كم ثانية

    def start(self):
        if not self.mock:
            try:
                import serial
                self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
                print(f"[Arduino] Connected on {self.port} @ {self.baud}")
            except Exception as e:
                print(f"[Arduino] Serial open failed: {e}")
                self._ser = None
        else:
            print("[Arduino] MOCK mode")
            def _mock_authorize():
                time.sleep(1.0)
                self.state.update("STATUS:AUTHORIZED")
                time.sleep(1.0)
                self.state.update("STATUS:FREE")
            threading.Thread(target=_mock_authorize, daemon=True).start()

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass

    def send(self, command: str, force: bool = False) -> bool:
        """
        بعث أمر للأردوينو.

        القاعدة الأساسية:
        ← إذا is_moving=True → لا نبعث GO جديد (إلا force=True للـ STOP)
        ← إذا نفس الأمر بُعث مؤخراً → ننتظر REPEAT_INTERVAL_S
        """
        now = time.time()
        state = self.state.snapshot()

        if not force:
            # ← الإصلاح: لا نبعث GO وهو شغال
            if state["is_moving"] and command.startswith("GO:"):
                return False

            with self._send_lock:
                if (
                    command == self._last_sent
                    and (now - self._last_sent_time) < self.REPEAT_INTERVAL_S
                ):
                    return False

        with self._send_lock:
            self._last_sent = command
            self._last_sent_time = now

        full_cmd = f"CMD:{command}\n"

        if self.mock:
            print(f"[Arduino MOCK] → {command}")
            # نسيمول الحركة في MOCK عشان نختبر الـ blocking
            if command.startswith("GO:"):
                def _mock_move():
                    self.state.update(f"STATUS:MOVING:{command[3:]}")
                    time.sleep(0.8)
                    self.state.update("STATUS:REACHED")
                    self.state.update("STATUS:FREE")
                threading.Thread(target=_mock_move, daemon=True).start()
            return True

        if self._ser and self._ser.is_open:
            try:
                self._ser.write(full_cmd.encode())
                return True
            except Exception as e:
                print(f"[Arduino] Send error: {e}")
        return False

    def _reader(self):
        """Thread رئيسي للقراءة."""
        buf = ""
        while not self._stop.is_set():
            if self.mock:
                time.sleep(0.05)
                continue

            if self._ser is None or not self._ser.is_open:
                time.sleep(0.5)
                continue

            try:
                data = self._ser.read(64).decode("utf-8", errors="ignore")
                if data:
                    buf += data
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            self.state.update(line)
            except Exception as e:
                if not self._stop.is_set():
                    print(f"[Arduino] Read error: {e}")
                time.sleep(0.2)