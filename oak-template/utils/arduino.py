# utils/arduino.py
"""
Smart Walker v8.0 — Arduino Serial
بسيط: thread يقرأ من الأردوينو، وfunction تبعث أوامر.
"""

import threading
import time
from typing import Optional


class ArduinoState:
    """الحالة الحالية للأردوينو — thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.authorized: bool = False
        self.ready: bool = False      # False إثناء auth sequence
        self.mode: str = "FREE"

    def update(self, msg: str):
        with self._lock:
            if msg == "STATUS:AUTHORIZED":
                self.authorized = True
                self.ready = False    # لازم ننتظر نهاية الـ auth sequence
                print("[Arduino] Authorized — waiting for auth sequence...")
            elif msg == "STATUS:UNAUTHORIZED":
                self.authorized = False
                self.ready = False
                print("[Arduino] Unauthorized")
            elif msg == "STATUS:FREE":
                self.mode = "FREE"
                self.ready = True     # auth sequence خلصت
                print("[Arduino] Ready (FREE mode)")
            elif msg == "STATUS:ASSIST":
                self.mode = "ASSIST"
                print("[Arduino] ASSIST mode")
            elif msg == "STATUS:STOPPED":
                print("[Arduino] Stopped")
            elif msg.startswith("STATUS:REACHED") or msg.startswith("STATUS:AT_TARGET"):
                print(f"[Arduino] {msg}")
            elif msg.startswith("BANK:"):
                print(f"[Arduino] Banknote: {msg}")
            elif msg.startswith("STATUS:SENSOR_ERROR"):
                print("[Arduino] ⚠️ Sensor error!")
            # باقي الرسائل نطبعها فقط
            else:
                print(f"[Arduino] {msg}")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "authorized": self.authorized,
                "ready": self.ready,
                "mode": self.mode,
            }


class ArduinoSerial:
    """
    Thread يقرأ من الأردوينو باستمرار.
    في حال MOCK: يشتغل بدون serial حقيقي (للتطوير على الكمبيوتر).
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

        # Rate limiting: ما نعيد إرسال نفس الأمر قبل X ثانية
        self.MIN_INTERVAL_S: float = 0.3
        self.REPEAT_INTERVAL_S: float = 2.5   # إعادة إرسال نفس الأمر كل 2.5 ثانية

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
            print("[Arduino] MOCK mode — simulating authorized state")
            # في MOCK: نعطي authorized + ready مباشرة بعد ثانية
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
        force=True: يتجاوز الـ rate limiting (للـ STOP مثلاً).
        يرجع True إذا بعثنا فعلاً.
        """
        now = time.time()

        with self._send_lock:
            # Rate limiting
            if not force:
                same_cmd = command == self._last_sent
                elapsed = now - self._last_sent_time

                if same_cmd and elapsed < self.REPEAT_INTERVAL_S:
                    return False  # نفس الأمر وما مضى وقت كافي
                if not same_cmd and elapsed < self.MIN_INTERVAL_S:
                    return False  # أمر مختلف لكن لسا سريع جداً

            self._last_sent = command
            self._last_sent_time = now

        full_cmd = f"CMD:{command}\n"

        if self.mock:
            print(f"[Arduino MOCK] → {command}")
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
                time.sleep(0.1)
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