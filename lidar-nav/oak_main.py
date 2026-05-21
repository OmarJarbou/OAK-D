"""
oak_audio_runner.py — OAK Camera + Audio Announcements (Standalone)
=====================================================================
يشتغل كـ process منفصل تماماً عن main.py (LiDAR).
لا يوجد أي تواصل مع Arduino أو LiDAR أو Serial.

تشغيل:
  Terminal 1:  python3 main.py           ← LiDAR + Arduino (ما تغيّر)
  Terminal 2:  python3 oak_audio_runner.py  ← OAK camera + صوت فقط

اختبار على لابتوب (بدون كاميرا OAK):
  بدّل السطر أدناه:  MOCK_OAK = True
"""

import time
import signal
import sys

# ── وضع الاختبار (True = لا يحتاج كاميرا OAK) ─────────────────────────
MOCK_OAK = False

# ── حد الخطر — يطابق DANGER_MM في navigator.py تلقائياً ───────────────
try:
    from navigator import DANGER_MM
except ImportError:
    DANGER_MM = 1200   # fallback لو شغّلت الملف وحده


def main():
    print("=== OAK Audio Runner ===")
    print(f"    MOCK_OAK  : {MOCK_OAK}")
    print(f"    Danger zone: ≤ {DANGER_MM} mm\n")

    # ── تشغيل OAK Detector ─────────────────────────────────────────────
    detector = None
    if not MOCK_OAK:
        from oak_detector import OakDetector
        detector = OakDetector(confidence_threshold=0.5)
        detector.start()
        print("[OAK] Warming up pipeline (3s)…")
        time.sleep(3)

    # ── تشغيل Audio System ─────────────────────────────────────────────
    from audio_system import AudioSystem
    audio = AudioSystem(
        detector=detector,
        danger_distance_mm=DANGER_MM,
        mock_detections=MOCK_OAK,
    )
    audio.start()

    # ── إيقاف نظيف ─────────────────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        print("\n[OAK-AUDIO] Shutting down…")
        audio.stop()
        if detector:
            detector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[OAK-AUDIO] Running. Press Ctrl+C to stop.\n")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()