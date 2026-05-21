"""
audio_system.py — Smart Walker Voice Announcer (Arabic)
========================================================
يولّد ملفات WAV عربية بـ espeak-ng مرة واحدة عند أول تشغيل،
ثم يشغّلها بـ aplay مباشرة (أسرع بكثير من TTS لحظي).

قواعد ثابتة:
  1. صمت خارج منطقة الخطر (distance > DANGER_DISTANCE_MM)
  2. يعلن عن الكائن الأقرب فقط عند وجود أكثر من كائن
  3. أوامر التنقل مستبعدة كلياً — كشف الكائنات فقط

متطلبات:
  sudo apt install espeak-ng

مجلد الأصوات (يُنشأ تلقائياً):
  sounds/objects/  ← WAV مُولَّدة ومخزَّنة
"""

import subprocess
import threading
import time
from pathlib import Path

# ── مجلد تخزين الأصوات على SD card ──────────────────────────────────
# غيّر المسار إذا كانت الـ SD card mounted في مكان مختلف
SOUNDS_DIR = Path("/home/lama/walker_sounds/objects")

# ── حد الخطر (يطابق DANGER_MM في navigator.py) ───────────────────────
DEFAULT_DANGER_MM = 1200

# ── Cooldown بين إعلانَين ─────────────────────────────────────────────
ANNOUNCE_COOLDOWN_SEC = 3.0
SAME_OBJECT_DELTA_MM  = 200

# ── espeak-ng: صوت عربي ───────────────────────────────────────────────
ESPEAK_VOICE = "ar"       # الصوت العربي
ESPEAK_SPEED = "130"      # كلمة/دقيقة — واضح وغير سريع
ESPEAK_AMP   = "180"      # حجم الصوت (0-200)

# ══════════════════════════════════════════════════════════════════════
#  ARABIC TEXT MAP — label + position → جملة عربية
# ══════════════════════════════════════════════════════════════════════

# المفتاح: "<label>_<position>"
# القيمة : النص العربي المنطوق + اسم ملف WAV
ARABIC_MAP: dict[str, tuple[str, str]] = {
    # شخص
    "person_left":       ("شخص على يسارك",    "person_left"),
    "person_right":      ("شخص على يمينك",    "person_right"),
    "person_center":     ("شخص أمامك",        "person_center"),

    # كرسي
    "chair_left":        ("كرسي على يسارك",   "chair_left"),
    "chair_right":       ("كرسي على يمينك",   "chair_right"),
    "chair_center":      ("كرسي أمامك",       "chair_center"),

    # دراجة
    "bicycle_left":      ("دراجة على يسارك",  "bicycle_left"),
    "bicycle_right":     ("دراجة على يمينك",  "bicycle_right"),
    "bicycle_center":    ("دراجة أمامك",      "bicycle_center"),

    # سيارة
    "car_left":          ("سيارة على يسارك",  "car_left"),
    "car_right":         ("سيارة على يمينك",  "car_right"),
    "car_center":        ("سيارة أمامك",      "car_center"),

    # كلب
    "dog_left":          ("كلب على يسارك",    "dog_left"),
    "dog_right":         ("كلب على يمينك",    "dog_right"),
    "dog_center":        ("كلب أمامك",        "dog_center"),

    # قطة
    "cat_left":          ("قطة على يسارك",    "cat_left"),
    "cat_right":         ("قطة على يمينك",    "cat_right"),
    "cat_center":        ("قطة أمامك",        "cat_center"),

    # دراجة نارية
    "motorbike_left":    ("دراجة نارية على يسارك",  "motorbike_left"),
    "motorbike_right":   ("دراجة نارية على يمينك",  "motorbike_right"),
    "motorbike_center":  ("دراجة نارية أمامك",      "motorbike_center"),

    # حافلة
    "bus_left":          ("حافلة على يسارك",  "bus_left"),
    "bus_right":         ("حافلة على يمينك",  "bus_right"),
    "bus_center":        ("حافلة أمامك",      "bus_center"),

    # طاولة / أريكة / عقبة عامة
    "diningtable_left":  ("طاولة على يسارك",  "diningtable_left"),
    "diningtable_right": ("طاولة على يمينك",  "diningtable_right"),
    "diningtable_center":("طاولة أمامك",      "diningtable_center"),

    "sofa_left":         ("أريكة على يسارك",  "sofa_left"),
    "sofa_right":        ("أريكة على يمينك",  "sofa_right"),
    "sofa_center":       ("أريكة أمامك",      "sofa_center"),

    "toilet_left":       ("عقبة على يسارك",   "toilet_left"),
    "toilet_right":      ("عقبة على يمينك",   "toilet_right"),
    "toilet_center":     ("عقبة أمامك",       "toilet_center"),
}


# ══════════════════════════════════════════════════════════════════════
#  WAV CACHE — يولّد الملفات مرة واحدة
# ══════════════════════════════════════════════════════════════════════

def _wav_path(filename: str) -> Path:
    return SOUNDS_DIR / f"{filename}.wav"


def pregenerate_sounds(force: bool = False):
    """
    يولّد جميع ملفات WAV بـ espeak-ng عند أول تشغيل.
    إذا الملف موجود مسبقاً لا يُعاد توليده (إلا مع force=True).
    """
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

    # تحقق من وجود espeak-ng
    check = subprocess.run(["which", "espeak-ng"], capture_output=True)
    if check.returncode != 0:
        print("[AUDIO] espeak-ng not found — install: sudo apt install espeak-ng")
        return False

    generated = 0
    skipped   = 0

    for key, (text, filename) in ARABIC_MAP.items():
        out = _wav_path(filename)
        if out.exists() and not force:
            skipped += 1
            continue

        cmd = [
            "espeak-ng",
            "-v", ESPEAK_VOICE,
            "-s", ESPEAK_SPEED,
            "-a", ESPEAK_AMP,
            "-w", str(out),
            text,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"[AUDIO] Generated: {out.name}  \"{text}\"")
            generated += 1
        else:
            print(f"[AUDIO] Failed: {out.name} — {result.stderr.decode()}")

    print(f"[AUDIO] Sound cache: {generated} generated, {skipped} already exist")
    return True


# ══════════════════════════════════════════════════════════════════════
#  PLAYBACK
# ══════════════════════════════════════════════════════════════════════

_current_proc: subprocess.Popen | None = None
_play_lock = threading.Lock()


def _play_wav(path: Path):
    """يشغّل ملف WAV بـ aplay في thread منفصل."""
    global _current_proc

    def _run():
        global _current_proc
        try:
            with _play_lock:
                # أوقف أي صوت حالي
                if _current_proc and _current_proc.poll() is None:
                    _current_proc.terminate()
                _current_proc = subprocess.Popen(
                    ["aplay", "-q", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            _current_proc.wait()
        except Exception as e:
            print(f"[AUDIO] aplay error: {e}")

    threading.Thread(target=_run, daemon=True).start()


def _play_espeak_fallback(text: str):
    """Fallback: ينطق النص مباشرة إذا الملف غير موجود."""
    cmd = ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED, text]
    threading.Thread(
        target=lambda: subprocess.run(cmd, capture_output=True),
        daemon=True
    ).start()


def play_object(label: str, position: str) -> bool:
    """
    شغّل الصوت المناسب لـ label + position.
    يرجع True إذا شغّل، False إذا ما في mapping.
    """
    key  = f"{label}_{position}"
    entry = ARABIC_MAP.get(key)

    if entry is None:
        print(f"[AUDIO] No Arabic mapping for '{key}'")
        return False

    text, filename = entry
    wav = _wav_path(filename)

    if wav.exists():
        _play_wav(wav)
    else:
        # الملف غير موجود → fallback مباشر
        print(f"[AUDIO] WAV missing ({wav.name}) — using espeak fallback")
        _play_espeak_fallback(text)

    return True


# ══════════════════════════════════════════════════════════════════════
#  AUDIO SYSTEM CLASS
# ══════════════════════════════════════════════════════════════════════

class AudioSystem:
    """
    نظام الإعلان الصوتي — كائنات فقط، بالعربي.

    Parameters
    ----------
    detector : OakDetector
        نسخة مشغّلة من OakDetector.
    danger_distance_mm : int
        الكائنات خارج هذا النطاق تُتجاهل صوتياً.
    mock_detections : bool
        True = اختبار بدون كاميرا OAK (يستخدم بيانات وهمية).
    """

    def __init__(self, detector=None,
                 danger_distance_mm: int = DEFAULT_DANGER_MM,
                 mock_detections: bool = False):
        self._detector  = detector
        self._danger_mm = danger_distance_mm
        self._mock      = mock_detections
        self._running   = False
        self._thread    = None

        self._last_announced_at = 0.0
        self._last_label        = None
        self._last_position     = None
        self._last_distance     = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        # ولّد الأصوات إذا لم تكن موجودة
        pregenerate_sounds()

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[AUDIO] Started — danger zone ≤ {self._danger_mm} mm | Arabic")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        # أوقف أي صوت جاري
        global _current_proc
        if _current_proc and _current_proc.poll() is None:
            _current_proc.terminate()
        print("[AUDIO] Stopped")

    def set_danger_distance(self, mm: int):
        self._danger_mm = mm

    # ── Internal loop ─────────────────────────────────────────────────

    def _loop(self):
        mock_gen = _mock_detection_generator() if self._mock else None

        while self._running:
            # ── اجلب الكشوفات ────────────────────────────────────────
            if self._mock:
                detections = next(mock_gen)
            else:
                detections = self._detector.get_detections() if self._detector else []

            # ── فلتر: منطقة الخطر فقط ────────────────────────────────
            in_danger = [d for d in detections
                         if d['distance_mm'] <= self._danger_mm]

            if not in_danger:
                time.sleep(0.2)
                continue

            # ── الأقرب فقط ───────────────────────────────────────────
            in_danger.sort(key=lambda d: d['distance_mm'])
            closest = in_danger[0]

            # ── Cooldown + delta ─────────────────────────────────────
            now           = time.time()
            same_label    = (closest['label']    == self._last_label)
            same_pos      = (closest['position'] == self._last_position)
            close_dist    = (self._last_distance is not None and
                             abs(closest['distance_mm'] - self._last_distance)
                             < SAME_OBJECT_DELTA_MM)
            cooldown_ok   = (now - self._last_announced_at) >= ANNOUNCE_COOLDOWN_SEC

            if same_label and same_pos and close_dist and not cooldown_ok:
                time.sleep(0.2)
                continue

            # ── أعلن ─────────────────────────────────────────────────
            played = play_object(closest['label'], closest['position'])

            key = f"{closest['label']}_{closest['position']}"
            ar_text = ARABIC_MAP.get(key, ("?", ""))[0]
            print(f"[AUDIO] {ar_text}  ({closest['distance_mm']} mm"
                  f" | {len(in_danger)} object(s) in zone)")

            if played:
                self._last_announced_at = time.time()
                self._last_label        = closest['label']
                self._last_position     = closest['position']
                self._last_distance     = closest['distance_mm']

            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════
#  Mock generator (اختبار بدون كاميرا)
# ══════════════════════════════════════════════════════════════════════

def _mock_detection_generator():
    scenarios = [
        [{'label': 'person',  'position': 'right',  'distance_mm': 900}],
        [{'label': 'person',  'position': 'right',  'distance_mm': 750},
         {'label': 'chair',   'position': 'center', 'distance_mm': 1100}],
        [{'label': 'chair',   'position': 'center', 'distance_mm': 300}],
        [],
        [{'label': 'bicycle', 'position': 'left',   'distance_mm': 1500}],  # خارج الخطر
        [{'label': 'dog',     'position': 'center', 'distance_mm': 600},
         {'label': 'person',  'position': 'left',   'distance_mm': 800}],
        [],
        [],
    ]
    i = 0
    while True:
        yield scenarios[i % len(scenarios)]
        i += 1
        time.sleep(2.5)


# ══════════════════════════════════════════════════════════════════════
#  اختبار مستقل
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Audio System — Arabic Mock Test ===")
    print(f"Danger zone: ≤ {DEFAULT_DANGER_MM} mm")
    print(f"Sounds dir : {SOUNDS_DIR}")
    print("Press Ctrl+C to stop\n")

    audio = AudioSystem(mock_detections=True, danger_distance_mm=DEFAULT_DANGER_MM)
    audio.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()