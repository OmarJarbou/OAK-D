#!/usr/bin/env python3
"""
button_handler.py — أزرار مادية على الواكر (Raspberry Pi GPIO)
===============================================================
3 أزرار تتصل مباشرة بـ Pi:

  زر 1 (البيت)    → GPIO17
  زر 2 (المسجد)   → GPIO27
  زر 3 (إلغاء)    → GPIO22

الأسلاك لكل زر:
  أحد طرفَي الزر  → GPIO Pin
  الطرف الثاني    → GND
  (الـ Pi عنده pull-up داخلي — ما بدك مقاومات)

تثبيت:
  pip install RPi.GPIO --break-system-packages
  أو (على Pi 5 الجديد):
  pip install gpiozero --break-system-packages

ملاحظة: الملف هذا يُستدعى من main.py — ما يشتغل لوحده في الإنتاج.
"""

import threading
import time

# ══════════════════════════════════════════════════════════════════════
#  إعدادات الأزرار
# ══════════════════════════════════════════════════════════════════════

# GPIO BCM numbering
BUTTON_PINS = {
    "البيت":    17,
    "المسجد":   27,
    "إلغاء":    22,
}

DEBOUNCE_MS = 300   # تجنب الارتداد (milliseconds)

# ══════════════════════════════════════════════════════════════════════
#  الكلاس الرئيسي
# ══════════════════════════════════════════════════════════════════════

class ButtonHandler:
    """
    يستمع للأزرار المادية ويستدعي callback عند الضغط.

    الاستخدام:
        def on_button(name):
            if name == "إلغاء":
                navigator.cancel()
            else:
                navigator.set_destination(name)

        buttons = ButtonHandler(callback=on_button)
        buttons.start()
        # ...
        buttons.stop()
    """

    def __init__(self, callback):
        self._callback   = callback
        self._running    = False
        self._gpio_ok    = False
        self._last_press = {}   # {pin: timestamp} لـ debounce

        # جرّب import GPIO
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            self._gpio_ok = True
        except ImportError:
            print("[BUTTONS] RPi.GPIO not found — running in MOCK mode")
            self._GPIO = None

    def start(self):
        if self._gpio_ok:
            self._setup_gpio()
        else:
            # وضع اختبار على الـ PC
            self._running = True
            t = threading.Thread(target=self._mock_loop, daemon=True)
            t.start()
        print("[BUTTONS] Started")

    def stop(self):
        self._running = False
        if self._gpio_ok and self._GPIO:
            try:
                self._GPIO.cleanup()
            except Exception:
                pass
        print("[BUTTONS] Stopped")

    # ── GPIO setup ────────────────────────────────────────────────────

    def _setup_gpio(self):
        GPIO = self._GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for name, pin in BUTTON_PINS.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            # add_event_detect: يكشف الضغط (FALLING = من HIGH إلى LOW)
            GPIO.add_event_detect(
                pin,
                GPIO.FALLING,
                callback=lambda ch, n=name: self._on_press(n),
                bouncetime=DEBOUNCE_MS
            )
            print(f"[BUTTONS] GPIO{pin} → '{name}'")

    def _on_press(self, name: str):
        now = time.time()
        # debounce إضافي يدوي
        last = self._last_press.get(name, 0)
        if (now - last) < (DEBOUNCE_MS / 1000.0):
            return
        self._last_press[name] = now
        print(f"[BUTTONS] Pressed: '{name}'")
        try:
            self._callback(name)
        except Exception as e:
            print(f"[BUTTONS] Callback error: {e}")

    # ── Mock mode (اختبار بدون Pi) ────────────────────────────────────

    def _mock_loop(self):
        """في وضع الاختبار: اضغط Enter واكتب اسم الوجهة."""
        print("[BUTTONS] MOCK mode — type destination name and press Enter:")
        print(f"          Options: {list(BUTTON_PINS.keys())}")
        while self._running:
            try:
                name = input(">> ").strip()
                if name in BUTTON_PINS:
                    self._callback(name)
                else:
                    print(f"[BUTTONS] Unknown: '{name}'  Options: {list(BUTTON_PINS.keys())}")
            except EOFError:
                time.sleep(1)
            except KeyboardInterrupt:
                break
