# # """
# # audio_system.py — Smart Walker Voice Announcer (Arabic)
# # ========================================================
# # يولّد ملفات WAV عربية بـ espeak-ng مرة واحدة عند أول تشغيل،
# # ثم يشغّلها بـ aplay مباشرة (أسرع بكثير من TTS لحظي).

# # قواعد ثابتة:
# #   1. صمت خارج منطقة الخطر (distance > DANGER_DISTANCE_MM)
# #   2. يعلن عن الكائن الأقرب فقط عند وجود أكثر من كائن
# #   3. أوامر التنقل مستبعدة كلياً — كشف الكائنات فقط

# # متطلبات:
# #   sudo apt install espeak-ng

# # مجلد الأصوات (يُنشأ تلقائياً):
# #   sounds/objects/  ← WAV مُولَّدة ومخزَّنة
# # """

# # import subprocess
# # import threading
# # import time
# # from pathlib import Path

# # # ── مجلد تخزين الأصوات على SD card ──────────────────────────────────
# # # غيّر المسار إذا كانت الـ SD card mounted في مكان مختلف
# # SOUNDS_DIR = Path("/home/lama/walker_sounds/objects")

# # # ── حد الخطر (يطابق DANGER_MM في navigator.py) ───────────────────────
# # DEFAULT_DANGER_MM = 1200

# # # ── Cooldown بين إعلانَين ─────────────────────────────────────────────
# # ANNOUNCE_COOLDOWN_SEC = 3.0
# # SAME_OBJECT_DELTA_MM  = 200
# # ALSA_DEVICE = "plughw:2,0"
# # # ── espeak-ng: صوت عربي ───────────────────────────────────────────────
# # ESPEAK_VOICE = "ar"       # الصوت العربي
# # ESPEAK_SPEED = "130"      # كلمة/دقيقة — واضح وغير سريع
# # ESPEAK_AMP   = "180"      # حجم الصوت (0-200)

# # # ══════════════════════════════════════════════════════════════════════
# # #  ARABIC TEXT MAP — label + position → جملة عربية
# # # ══════════════════════════════════════════════════════════════════════

# # # المفتاح: "<label>_<position>"
# # # القيمة : النص العربي المنطوق + اسم ملف WAV
# # ARABIC_MAP: dict[str, tuple[str, str]] = {
# #     # شخص
# #     "person_left":       ("شخص على يسارك",    "person_left"),
# #     "person_right":      ("شخص على يمينك",    "person_right"),
# #     "person_center":     ("شخص أمامك",        "person_center"),

# #     # كرسي
# #     "chair_left":        ("كرسي على يسارك",   "chair_left"),
# #     "chair_right":       ("كرسي على يمينك",   "chair_right"),
# #     "chair_center":      ("كرسي أمامك",       "chair_center"),

# #     # دراجة
# #     "bicycle_left":      ("دراجة على يسارك",  "bicycle_left"),
# #     "bicycle_right":     ("دراجة على يمينك",  "bicycle_right"),
# #     "bicycle_center":    ("دراجة أمامك",      "bicycle_center"),

# #     # سيارة
# #     "car_left":          ("سيارة على يسارك",  "car_left"),
# #     "car_right":         ("سيارة على يمينك",  "car_right"),
# #     "car_center":        ("سيارة أمامك",      "car_center"),

# #     # كلب
# #     "dog_left":          ("كلب على يسارك",    "dog_left"),
# #     "dog_right":         ("كلب على يمينك",    "dog_right"),
# #     "dog_center":        ("كلب أمامك",        "dog_center"),

# #     # قطة
# #     "cat_left":          ("قطة على يسارك",    "cat_left"),
# #     "cat_right":         ("قطة على يمينك",    "cat_right"),
# #     "cat_center":        ("قطة أمامك",        "cat_center"),

# #     # دراجة نارية
# #     "motorbike_left":    ("دراجة نارية على يسارك",  "motorbike_left"),
# #     "motorbike_right":   ("دراجة نارية على يمينك",  "motorbike_right"),
# #     "motorbike_center":  ("دراجة نارية أمامك",      "motorbike_center"),

# #     # حافلة
# #     "bus_left":          ("حافلة على يسارك",  "bus_left"),
# #     "bus_right":         ("حافلة على يمينك",  "bus_right"),
# #     "bus_center":        ("حافلة أمامك",      "bus_center"),

# #     # طاولة / أريكة / عقبة عامة
# #     "diningtable_left":  ("طاولة على يسارك",  "diningtable_left"),
# #     "diningtable_right": ("طاولة على يمينك",  "diningtable_right"),
# #     "diningtable_center":("طاولة أمامك",      "diningtable_center"),

# #     "sofa_left":         ("أريكة على يسارك",  "sofa_left"),
# #     "sofa_right":        ("أريكة على يمينك",  "sofa_right"),
# #     "sofa_center":       ("أريكة أمامك",      "sofa_center"),

# #     "toilet_left":       ("عقبة على يسارك",   "toilet_left"),
# #     "toilet_right":      ("عقبة على يمينك",   "toilet_right"),
# #     "toilet_center":     ("عقبة أمامك",       "toilet_center"),
# # }


# # # ══════════════════════════════════════════════════════════════════════
# # #  WAV CACHE — يولّد الملفات مرة واحدة
# # # ══════════════════════════════════════════════════════════════════════

# # def _wav_path(filename: str) -> Path:
# #     return SOUNDS_DIR / f"{filename}.wav"


# # def pregenerate_sounds(force: bool = False):
# #     """
# #     يولّد جميع ملفات WAV بـ espeak-ng عند أول تشغيل.
# #     إذا الملف موجود مسبقاً لا يُعاد توليده (إلا مع force=True).
# #     """
# #     SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

# #     # تحقق من وجود espeak-ng
# #     check = subprocess.run(["which", "espeak-ng"], capture_output=True)
# #     if check.returncode != 0:
# #         print("[AUDIO] espeak-ng not found — install: sudo apt install espeak-ng")
# #         return False

# #     generated = 0
# #     skipped   = 0

# #     for key, (text, filename) in ARABIC_MAP.items():
# #         out = _wav_path(filename)
# #         if out.exists() and not force:
# #             skipped += 1
# #             continue

# #         cmd = [
# #             "espeak-ng",
# #             "-v", ESPEAK_VOICE,
# #             "-s", ESPEAK_SPEED,
# #             "-a", ESPEAK_AMP,
# #             "-w", str(out),
# #         ]
# #         # مرّر النص عبر stdin لتجنب مشكلة encoding مع العربية
# #         result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
# #         if result.returncode == 0:
# #             print(f"[AUDIO] Generated: {out.name}  \"{text}\"")
# #             generated += 1
# #         else:
# #             print(f"[AUDIO] Failed: {out.name} — {result.stderr.decode()}")

# #     print(f"[AUDIO] Sound cache: {generated} generated, {skipped} already exist")
# #     return True


# # # ══════════════════════════════════════════════════════════════════════
# # #  PLAYBACK
# # # ══════════════════════════════════════════════════════════════════════

# # _current_proc: subprocess.Popen | None = None
# # _play_lock = threading.Lock()


# # def _play_wav(path: Path):
# #     """يشغّل ملف WAV بـ aplay في thread منفصل."""
# #     global _current_proc

# #     def _run():
# #         global _current_proc
# #         try:
# #             with _play_lock:
# #                 # أوقف أي صوت حالي
# #                 if _current_proc and _current_proc.poll() is None:
# #                     _current_proc.terminate()
# #             # ?? audio_system.py � ???? _play_wav
# #                 _current_proc = subprocess.Popen(
# #                     ["aplay", "-q", "-D", "plughw:2,0", str(path)],
# #                     stdout=subprocess.DEVNULL,
# #                     stderr=subprocess.DEVNULL,
# #                 )
# #             _current_proc.wait()
# #         except Exception as e:
# #             print(f"[AUDIO] aplay error: {e}")

# #     threading.Thread(target=_run, daemon=True).start()


# # def _play_espeak_fallback(text: str):
# #     """Fallback: ينطق النص مباشرة إذا الملف غير موجود."""
# #     cmd = ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED]
# #     threading.Thread(
# #         target=lambda: subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True),
# #         daemon=True
# #     ).start()


# # def play_object(label: str, position: str) -> bool:
# #     """
# #     شغّل الصوت المناسب لـ label + position.
# #     يرجع True إذا شغّل، False إذا ما في mapping.
# #     """
# #     key  = f"{label}_{position}"
# #     entry = ARABIC_MAP.get(key)

# #     if entry is None:
# #         print(f"[AUDIO] No Arabic mapping for '{key}'")
# #         return False

# #     text, filename = entry
# #     wav = _wav_path(filename)

# #     if wav.exists():
# #         _play_wav(wav)
# #     else:
# #         # الملف غير موجود → fallback مباشر
# #         print(f"[AUDIO] WAV missing ({wav.name}) — using espeak fallback")
# #         _play_espeak_fallback(text)

# #     return True


# # # ══════════════════════════════════════════════════════════════════════
# # #  AUDIO SYSTEM CLASS
# # # ══════════════════════════════════════════════════════════════════════

# # class AudioSystem:
# #     """
# #     نظام الإعلان الصوتي — كائنات فقط، بالعربي.

# #     Parameters
# #     ----------
# #     detector : OakDetector
# #         نسخة مشغّلة من OakDetector.
# #     danger_distance_mm : int
# #         الكائنات خارج هذا النطاق تُتجاهل صوتياً.
# #     mock_detections : bool
# #         True = اختبار بدون كاميرا OAK (يستخدم بيانات وهمية).
# #     """

# #     def __init__(self, detector=None,
# #                  danger_distance_mm: int = DEFAULT_DANGER_MM,
# #                  mock_detections: bool = False):
# #         self._detector  = detector
# #         self._danger_mm = danger_distance_mm
# #         self._mock      = mock_detections
# #         self._running   = False
# #         self._thread    = None

# #         self._last_announced_at = 0.0
# #         self._last_label        = None
# #         self._last_position     = None
# #         self._last_distance     = None

# #     # ── Public API ────────────────────────────────────────────────────

# #     def start(self):
# #         # ولّد الأصوات إذا لم تكن موجودة
# #         pregenerate_sounds()

# #         self._running = True
# #         self._thread  = threading.Thread(target=self._loop, daemon=True)
# #         self._thread.start()
# #         print(f"[AUDIO] Started — danger zone ≤ {self._danger_mm} mm | Arabic")

# #     def stop(self):
# #         self._running = False
# #         if self._thread:
# #             self._thread.join(timeout=3)
# #         # أوقف أي صوت جاري
# #         global _current_proc
# #         if _current_proc and _current_proc.poll() is None:
# #             _current_proc.terminate()
# #         print("[AUDIO] Stopped")

# #     def set_danger_distance(self, mm: int):
# #         self._danger_mm = mm

# #     # ── Internal loop ─────────────────────────────────────────────────

# #     def _loop(self):
# #         mock_gen = _mock_detection_generator() if self._mock else None

# #         while self._running:
# #             # ── اجلب الكشوفات ────────────────────────────────────────
# #             if self._mock:
# #                 detections = next(mock_gen)
# #             else:
# #                 detections = self._detector.get_detections() if self._detector else []

# #             # ── فلتر: منطقة الخطر فقط ────────────────────────────────
# #             in_danger = [d for d in detections
# #                          if d['distance_mm'] <= self._danger_mm]

# #             if not in_danger:
# #                 time.sleep(0.2)
# #                 continue

# #             # ── الأقرب فقط ───────────────────────────────────────────
# #             in_danger.sort(key=lambda d: d['distance_mm'])
# #             closest = in_danger[0]

# #             # ── Cooldown + delta ─────────────────────────────────────
# #             now           = time.time()
# #             same_label    = (closest['label']    == self._last_label)
# #             same_pos      = (closest['position'] == self._last_position)
# #             close_dist    = (self._last_distance is not None and
# #                              abs(closest['distance_mm'] - self._last_distance)
# #                              < SAME_OBJECT_DELTA_MM)
# #             cooldown_ok   = (now - self._last_announced_at) >= ANNOUNCE_COOLDOWN_SEC

# #             if same_label and same_pos and close_dist and not cooldown_ok:
# #                 time.sleep(0.2)
# #                 continue

# #             # ── أعلن ─────────────────────────────────────────────────
# #             played = play_object(closest['label'], closest['position'])

# #             key = f"{closest['label']}_{closest['position']}"
# #             ar_text = ARABIC_MAP.get(key, ("?", ""))[0]
# #             print(f"[AUDIO] {ar_text}  ({closest['distance_mm']} mm"
# #                   f" | {len(in_danger)} object(s) in zone)")

# #             if played:
# #                 self._last_announced_at = time.time()
# #                 self._last_label        = closest['label']
# #                 self._last_position     = closest['position']
# #                 self._last_distance     = closest['distance_mm']

# #             time.sleep(0.1)


# # # ══════════════════════════════════════════════════════════════════════
# # #  Mock generator (اختبار بدون كاميرا)
# # # ══════════════════════════════════════════════════════════════════════

# # def _mock_detection_generator():
# #     scenarios = [
# #         [{'label': 'person',  'position': 'right',  'distance_mm': 900}],
# #         [{'label': 'person',  'position': 'right',  'distance_mm': 750},
# #          {'label': 'chair',   'position': 'center', 'distance_mm': 1100}],
# #         [{'label': 'chair',   'position': 'center', 'distance_mm': 300}],
# #         [],
# #         [{'label': 'bicycle', 'position': 'left',   'distance_mm': 1500}],  # خارج الخطر
# #         [{'label': 'dog',     'position': 'center', 'distance_mm': 600},
# #          {'label': 'person',  'position': 'left',   'distance_mm': 800}],
# #         [],
# #         [],
# #     ]
# #     i = 0
# #     while True:
# #         yield scenarios[i % len(scenarios)]
# #         i += 1
# #         time.sleep(2.5)


# # # ══════════════════════════════════════════════════════════════════════
# # #  اختبار مستقل
# # # ══════════════════════════════════════════════════════════════════════

# # if __name__ == "__main__":
# #     print("=== Audio System — Arabic Mock Test ===")
# #     print(f"Danger zone: ≤ {DEFAULT_DANGER_MM} mm")
# #     print(f"Sounds dir : {SOUNDS_DIR}")
# #     print("Press Ctrl+C to stop\n")

# #     audio = AudioSystem(mock_detections=True, danger_distance_mm=DEFAULT_DANGER_MM)
# #     audio.start()

# #     try:
# #         while True:
# #             time.sleep(1)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         audio.stop()
# """
# audio_system.py — Smart Walker Voice Announcer
# ملفات صوتية جاهزة فقط، بدون TTS أو espeak.
# """

# import subprocess
# import threading
# import time
# from pathlib import Path

# # ── مجلد ملفاتك الصوتية ───────────────────────────────────────────────
# SOUNDS = Path("/home/lama/walker_sounds/objects")

# # ── Mapping مباشر: key → ملف WAV ─────────────────────────────────────
# # المفتاح: f"{label}_{position}"  (كلها lowercase)
# # القيمة: مسار الملف الصوتي الجاهز
# SOUND_MAP: dict[str, Path] = {
#     # شخص
#     "person_left":          SOUNDS / "person_left.wav",
#     "person_right":         SOUNDS / "person_right.wav",
#     "person_center":        SOUNDS / "person_center.wav",

#     # كرسي
#     "chair_left":           SOUNDS / "chair_left.wav",
#     "chair_right":          SOUNDS / "chair_right.wav",
#     "chair_center":         SOUNDS / "chair_center.wav",

#     # دراجة
#     "bicycle_left":         SOUNDS / "bicycle_left.wav",
#     "bicycle_right":        SOUNDS / "bicycle_right.wav",
#     "bicycle_center":       SOUNDS / "bicycle_center.wav",

#     # سيارة
#     "car_left":             SOUNDS / "car_left.wav",
#     "car_right":            SOUNDS / "car_right.wav",
#     "car_center":           SOUNDS / "car_center.wav",

#     # دراجة نارية
#     "motorcycle_left":      SOUNDS / "motorcycle_left.wav",
#     "motorcycle_right":     SOUNDS / "motorcycle_right.wav",
#     "motorcycle_center":    SOUNDS / "motorcycle_center.wav",

#     # حافلة
#     "bus_left":             SOUNDS / "bus_left.wav",
#     "bus_right":            SOUNDS / "bus_right.wav",
#     "bus_center":           SOUNDS / "bus_center.wav",

#     # كلب
#     "dog_left":             SOUNDS / "dog_left.wav",
#     "dog_right":            SOUNDS / "dog_right.wav",
#     "dog_center":           SOUNDS / "dog_center.wav",

#     # قطة
#     "cat_left":             SOUNDS / "cat_left.wav",
#     "cat_right":            SOUNDS / "cat_right.wav",
#     "cat_center":           SOUNDS / "cat_center.wav",

#     # طاولة
#     "diningtable_left":     SOUNDS / "diningtable_left.wav",
#     "diningtable_right":    SOUNDS / "diningtable_right.wav",
#     "diningtable_center":   SOUNDS / "diningtable_center.wav",

#     # أريكة
#     "sofa_left":            SOUNDS / "sofa_left.wav",
#     "sofa_right":           SOUNDS / "sofa_right.wav",
#     "sofa_center":          SOUNDS / "sofa_center.wav",

#     # عقبات عامة
#     "toilet_left":          SOUNDS / "toilet_left.wav",
#     "toilet_right":         SOUNDS / "toilet_right.wav",
#     "toilet_center":        SOUNDS / "toilet_center.wav",

#     # أضف أي label جديد هنا...
# }

# # ── إعدادات ───────────────────────────────────────────────────────────
# ALSA_DEVICE           = "plughw:2,0"
# DEFAULT_DANGER_MM     = 1200
# ANNOUNCE_COOLDOWN_SEC = 3.0
# SAME_OBJECT_DELTA_MM  = 200

# # ── Playback ─────────────────────────────────────────────────────────
# _current_proc: subprocess.Popen | None = None
# _play_lock = threading.Lock()


# def _play_wav(path: Path):
#     """شغّل WAV في thread منفصل، أوقف السابق تلقائياً."""
#     global _current_proc

#     def _run():
#         global _current_proc
#         with _play_lock:
#             if _current_proc and _current_proc.poll() is None:
#                 _current_proc.terminate()
#             _current_proc = subprocess.Popen(
#                 ["aplay", "-q", "-D", ALSA_DEVICE, str(path)],
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.DEVNULL,
#             )
#         _current_proc.wait()

#     threading.Thread(target=_run, daemon=True).start()


# def play_object(label: str, position: str) -> bool:
#     """
#     شغّل الملف الصوتي المناسب.
#     يرجع True إذا وجد الملف، False إذا لا.
#     """
#     key = f"{label}_{position}"
#     wav = SOUND_MAP.get(key)

#     if wav is None:
#         print(f"[AUDIO] No mapping for '{key}'")
#         return False

#     if not wav.exists():
#         print(f"[AUDIO] File missing: {wav}")
#         return False

#     _play_wav(wav)
#     return True


# def verify_sounds():
#     """تحقق من وجود كل الملفات عند الإقلاع — بلّغ عن الناقص."""
#     missing = [k for k, p in SOUND_MAP.items() if not p.exists()]
#     if missing:
#         print(f"[AUDIO] ⚠ Missing {len(missing)} sound files:")
#         for k in missing:
#             print(f"         {k} → {SOUND_MAP[k]}")
#     else:
#         print(f"[AUDIO] ✓ All {len(SOUND_MAP)} sound files found")
#     return len(missing) == 0


# # ── AudioSystem Class ─────────────────────────────────────────────────

# class AudioSystem:
#     def __init__(self, detector=None,
#                  danger_distance_mm: int = DEFAULT_DANGER_MM,
#                  mock_detections: bool = False):
#         self._detector  = detector
#         self._danger_mm = danger_distance_mm
#         self._mock      = mock_detections
#         self._running   = False
#         self._thread    = None

#         self._last_announced_at = 0.0
#         self._last_label        = None
#         self._last_position     = None
#         self._last_distance     = None

#     def start(self):
#         verify_sounds()   # تحقق مرة واحدة عند الإقلاع
#         self._running = True
#         self._thread  = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()
#         print(f"[AUDIO] Started — danger zone ≤ {self._danger_mm} mm")

#     def stop(self):
#         self._running = False
#         if self._thread:
#             self._thread.join(timeout=3)
#         global _current_proc
#         if _current_proc and _current_proc.poll() is None:
#             _current_proc.terminate()
#         print("[AUDIO] Stopped")

#     def _loop(self):
#         mock_gen = _mock_detection_generator() if self._mock else None

#         while self._running:
#             detections = next(mock_gen) if self._mock else (
#                 self._detector.get_detections() if self._detector else []
#             )

#             in_danger = [d for d in detections if d['distance_mm'] <= self._danger_mm]

#             if not in_danger:
#                 time.sleep(0.2)
#                 continue

#             in_danger.sort(key=lambda d: d['distance_mm'])
#             closest = in_danger[0]

#             now        = time.time()
#             same_label = closest['label']    == self._last_label
#             same_pos   = closest['position'] == self._last_position
#             close_dist = (self._last_distance is not None and
#                           abs(closest['distance_mm'] - self._last_distance) < SAME_OBJECT_DELTA_MM)
#             cooldown_ok = (now - self._last_announced_at) >= ANNOUNCE_COOLDOWN_SEC

#             if same_label and same_pos and close_dist and not cooldown_ok:
#                 time.sleep(0.2)
#                 continue

#             played = play_object(closest['label'], closest['position'])

#             if played:
#                 self._last_announced_at = now
#                 self._last_label        = closest['label']
#                 self._last_position     = closest['position']
#                 self._last_distance     = closest['distance_mm']
#                 print(f"[AUDIO] ▶ {closest['label']} {closest['position']} "
#                       f"({closest['distance_mm']} mm)")

#             time.sleep(0.1)


# def _mock_detection_generator():
#     import itertools
#     scenarios = [
#         [{'label': 'person',  'position': 'right',  'distance_mm': 900}],
#         [{'label': 'chair',   'position': 'center', 'distance_mm': 500}],
#         [],
#         [{'label': 'car',     'position': 'left',   'distance_mm': 800}],
#     ]
#     for s in itertools.cycle(scenarios):
#         yield s
#         time.sleep(2.5)

# //////////////////////////////////////////////////////////
# """
# audio_system.py â€” Smart Walker Voice Announcer (Arabic)
# ========================================================
# ÙŠÙˆÙ„Ù‘Ø¯ Ù…Ù„ÙØ§Øª WAV Ø¹Ø±Ø¨ÙŠØ© Ø¨Ù€ espeak-ng Ù…Ø±Ø© ÙˆØ§Ø­Ø¯Ø© Ø¹Ù†Ø¯ Ø£ÙˆÙ„ ØªØ´ØºÙŠÙ„ØŒ
# Ø«Ù… ÙŠØ´ØºÙ‘Ù„Ù‡Ø§ Ø¨Ù€ aplay Ù…Ø¨Ø§Ø´Ø±Ø© (Ø£Ø³Ø±Ø¹ Ø¨ÙƒØ«ÙŠØ± Ù…Ù† TTS Ù„Ø­Ø¸ÙŠ).

# Ù‚ÙˆØ§Ø¹Ø¯ Ø«Ø§Ø¨ØªØ©:
#   1. ØµÙ…Øª Ø®Ø§Ø±Ø¬ Ù…Ù†Ø·Ù‚Ø© Ø§Ù„Ø®Ø·Ø± (distance > DANGER_DISTANCE_MM)
#   2. ÙŠØ¹Ù„Ù† Ø¹Ù† Ø§Ù„ÙƒØ§Ø¦Ù† Ø§Ù„Ø£Ù‚Ø±Ø¨ ÙÙ‚Ø· Ø¹Ù†Ø¯ ÙˆØ¬ÙˆØ¯ Ø£ÙƒØ«Ø± Ù…Ù† ÙƒØ§Ø¦Ù†
#   3. Ø£ÙˆØ§Ù…Ø± Ø§Ù„ØªÙ†Ù‚Ù„ Ù…Ø³ØªØ¨Ø¹Ø¯Ø© ÙƒÙ„ÙŠØ§Ù‹ â€” ÙƒØ´Ù Ø§Ù„ÙƒØ§Ø¦Ù†Ø§Øª ÙÙ‚Ø·

# Ù…ØªØ·Ù„Ø¨Ø§Øª:
#   sudo apt install espeak-ng

# Ù…Ø¬Ù„Ø¯ Ø§Ù„Ø£ØµÙˆØ§Øª (ÙŠÙÙ†Ø´Ø£ ØªÙ„Ù‚Ø§Ø¦ÙŠØ§Ù‹):
#   sounds/objects/  â† WAV Ù…ÙÙˆÙ„ÙŽÙ‘Ø¯Ø© ÙˆÙ…Ø®Ø²ÙŽÙ‘Ù†Ø©
# """

# import subprocess
# import threading
# import time
# from pathlib import Path

# # â”€â”€ Ù…Ø¬Ù„Ø¯ ØªØ®Ø²ÙŠÙ† Ø§Ù„Ø£ØµÙˆØ§Øª Ø¹Ù„Ù‰ SD card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# # ØºÙŠÙ‘Ø± Ø§Ù„Ù…Ø³Ø§Ø± Ø¥Ø°Ø§ ÙƒØ§Ù†Øª Ø§Ù„Ù€ SD card mounted ÙÙŠ Ù…ÙƒØ§Ù† Ù…Ø®ØªÙ„Ù
# SOUNDS_DIR = Path("/home/lama/walker_sounds/objects")

# # â”€â”€ Ø­Ø¯ Ø§Ù„Ø®Ø·Ø± (ÙŠØ·Ø§Ø¨Ù‚ DANGER_MM ÙÙŠ navigator.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DEFAULT_DANGER_MM = 1200

# # â”€â”€ Cooldown Ø¨ÙŠÙ† Ø¥Ø¹Ù„Ø§Ù†ÙŽÙŠÙ† â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ANNOUNCE_COOLDOWN_SEC = 3.0
# SAME_OBJECT_DELTA_MM  = 200
# ALSA_DEVICE = "plughw:2,0"
# # â”€â”€ espeak-ng: ØµÙˆØª Ø¹Ø±Ø¨ÙŠ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ESPEAK_VOICE = "ar"       # Ø§Ù„ØµÙˆØª Ø§Ù„Ø¹Ø±Ø¨ÙŠ
# ESPEAK_SPEED = "130"      # ÙƒÙ„Ù…Ø©/Ø¯Ù‚ÙŠÙ‚Ø© â€” ÙˆØ§Ø¶Ø­ ÙˆØºÙŠØ± Ø³Ø±ÙŠØ¹
# ESPEAK_AMP   = "180"      # Ø­Ø¬Ù… Ø§Ù„ØµÙˆØª (0-200)

# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# #  ARABIC TEXT MAP â€” label + position â†’ Ø¬Ù…Ù„Ø© Ø¹Ø±Ø¨ÙŠØ©
# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# # Ø§Ù„Ù…ÙØªØ§Ø­: "<label>_<position>"
# # Ø§Ù„Ù‚ÙŠÙ…Ø© : Ø§Ù„Ù†Øµ Ø§Ù„Ø¹Ø±Ø¨ÙŠ Ø§Ù„Ù…Ù†Ø·ÙˆÙ‚ + Ø§Ø³Ù… Ù…Ù„Ù WAV
# ARABIC_MAP: dict[str, tuple[str, str]] = {
#     # Ø´Ø®Øµ
#     "person_left":       ("Ø´Ø®Øµ Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",    "person_left"),
#     "person_right":      ("Ø´Ø®Øµ Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",    "person_right"),
#     "person_center":     ("Ø´Ø®Øµ Ø£Ù…Ø§Ù…Ùƒ",        "person_center"),

#     # ÙƒØ±Ø³ÙŠ
#     "chair_left":        ("ÙƒØ±Ø³ÙŠ Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",   "chair_left"),
#     "chair_right":       ("ÙƒØ±Ø³ÙŠ Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",   "chair_right"),
#     "chair_center":      ("ÙƒØ±Ø³ÙŠ Ø£Ù…Ø§Ù…Ùƒ",       "chair_center"),

#     # Ø¯Ø±Ø§Ø¬Ø©
#     "bicycle_left":      ("Ø¯Ø±Ø§Ø¬Ø© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",  "bicycle_left"),
#     "bicycle_right":     ("Ø¯Ø±Ø§Ø¬Ø© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",  "bicycle_right"),
#     "bicycle_center":    ("Ø¯Ø±Ø§Ø¬Ø© Ø£Ù…Ø§Ù…Ùƒ",      "bicycle_center"),

#     # Ø³ÙŠØ§Ø±Ø©
#     "car_left":          ("Ø³ÙŠØ§Ø±Ø© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",  "car_left"),
#     "car_right":         ("Ø³ÙŠØ§Ø±Ø© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",  "car_right"),
#     "car_center":        ("Ø³ÙŠØ§Ø±Ø© Ø£Ù…Ø§Ù…Ùƒ",      "car_center"),

#     # ÙƒÙ„Ø¨
#     "dog_left":          ("ÙƒÙ„Ø¨ Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",    "dog_left"),
#     "dog_right":         ("ÙƒÙ„Ø¨ Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",    "dog_right"),
#     "dog_center":        ("ÙƒÙ„Ø¨ Ø£Ù…Ø§Ù…Ùƒ",        "dog_center"),

#     # Ù‚Ø·Ø©
#     "cat_left":          ("Ù‚Ø·Ø© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",    "cat_left"),
#     "cat_right":         ("Ù‚Ø·Ø© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",    "cat_right"),
#     "cat_center":        ("Ù‚Ø·Ø© Ø£Ù…Ø§Ù…Ùƒ",        "cat_center"),

#     # Ø¯Ø±Ø§Ø¬Ø© Ù†Ø§Ø±ÙŠØ©
#     "motorbike_left":    ("Ø¯Ø±Ø§Ø¬Ø© Ù†Ø§Ø±ÙŠØ© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",  "motorbike_left"),
#     "motorbike_right":   ("Ø¯Ø±Ø§Ø¬Ø© Ù†Ø§Ø±ÙŠØ© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",  "motorbike_right"),
#     "motorbike_center":  ("Ø¯Ø±Ø§Ø¬Ø© Ù†Ø§Ø±ÙŠØ© Ø£Ù…Ø§Ù…Ùƒ",      "motorbike_center"),

#     # Ø­Ø§ÙÙ„Ø©
#     "bus_left":          ("Ø­Ø§ÙÙ„Ø© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",  "bus_left"),
#     "bus_right":         ("Ø­Ø§ÙÙ„Ø© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",  "bus_right"),
#     "bus_center":        ("Ø­Ø§ÙÙ„Ø© Ø£Ù…Ø§Ù…Ùƒ",      "bus_center"),

#     # Ø·Ø§ÙˆÙ„Ø© / Ø£Ø±ÙŠÙƒØ© / Ø¹Ù‚Ø¨Ø© Ø¹Ø§Ù…Ø©
#     "diningtable_left":  ("Ø·Ø§ÙˆÙ„Ø© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",  "diningtable_left"),
#     "diningtable_right": ("Ø·Ø§ÙˆÙ„Ø© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",  "diningtable_right"),
#     "diningtable_center":("Ø·Ø§ÙˆÙ„Ø© Ø£Ù…Ø§Ù…Ùƒ",      "diningtable_center"),

#     "sofa_left":         ("Ø£Ø±ÙŠÙƒØ© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",  "sofa_left"),
#     "sofa_right":        ("Ø£Ø±ÙŠÙƒØ© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",  "sofa_right"),
#     "sofa_center":       ("Ø£Ø±ÙŠÙƒØ© Ø£Ù…Ø§Ù…Ùƒ",      "sofa_center"),

#     "toilet_left":       ("Ø¹Ù‚Ø¨Ø© Ø¹Ù„Ù‰ ÙŠØ³Ø§Ø±Ùƒ",   "toilet_left"),
#     "toilet_right":      ("Ø¹Ù‚Ø¨Ø© Ø¹Ù„Ù‰ ÙŠÙ…ÙŠÙ†Ùƒ",   "toilet_right"),
#     "toilet_center":     ("Ø¹Ù‚Ø¨Ø© Ø£Ù…Ø§Ù…Ùƒ",       "toilet_center"),
# }


# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# #  WAV CACHE â€” ÙŠÙˆÙ„Ù‘Ø¯ Ø§Ù„Ù…Ù„ÙØ§Øª Ù…Ø±Ø© ÙˆØ§Ø­Ø¯Ø©
# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# def _wav_path(filename: str) -> Path:
#     return SOUNDS_DIR / f"{filename}.wav"


# def pregenerate_sounds(force: bool = False):
#     """
#     ÙŠÙˆÙ„Ù‘Ø¯ Ø¬Ù…ÙŠØ¹ Ù…Ù„ÙØ§Øª WAV Ø¨Ù€ espeak-ng Ø¹Ù†Ø¯ Ø£ÙˆÙ„ ØªØ´ØºÙŠÙ„.
#     Ø¥Ø°Ø§ Ø§Ù„Ù…Ù„Ù Ù…ÙˆØ¬ÙˆØ¯ Ù…Ø³Ø¨Ù‚Ø§Ù‹ Ù„Ø§ ÙŠÙØ¹Ø§Ø¯ ØªÙˆÙ„ÙŠØ¯Ù‡ (Ø¥Ù„Ø§ Ù…Ø¹ force=True).
#     """
#     SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

#     # ØªØ­Ù‚Ù‚ Ù…Ù† ÙˆØ¬ÙˆØ¯ espeak-ng
#     check = subprocess.run(["which", "espeak-ng"], capture_output=True)
#     if check.returncode != 0:
#         print("[AUDIO] espeak-ng not found â€” install: sudo apt install espeak-ng")
#         return False

#     generated = 0
#     skipped   = 0

#     for key, (text, filename) in ARABIC_MAP.items():
#         out = _wav_path(filename)
#         if out.exists() and not force:
#             skipped += 1
#             continue

#         cmd = [
#             "espeak-ng",
#             "-v", ESPEAK_VOICE,
#             "-s", ESPEAK_SPEED,
#             "-a", ESPEAK_AMP,
#             "-w", str(out),
#         ]
#         # Ù…Ø±Ù‘Ø± Ø§Ù„Ù†Øµ Ø¹Ø¨Ø± stdin Ù„ØªØ¬Ù†Ø¨ Ù…Ø´ÙƒÙ„Ø© encoding Ù…Ø¹ Ø§Ù„Ø¹Ø±Ø¨ÙŠØ©
#         result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
#         if result.returncode == 0:
#             print(f"[AUDIO] Generated: {out.name}  \"{text}\"")
#             generated += 1
#         else:
#             print(f"[AUDIO] Failed: {out.name} â€” {result.stderr.decode()}")

#     print(f"[AUDIO] Sound cache: {generated} generated, {skipped} already exist")
#     return True


# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# #  PLAYBACK
# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# _current_proc: subprocess.Popen | None = None
# _play_lock = threading.Lock()


# def _play_wav(path: Path):
#     """ÙŠØ´ØºÙ‘Ù„ Ù…Ù„Ù WAV Ø¨Ù€ aplay ÙÙŠ thread Ù…Ù†ÙØµÙ„."""
#     global _current_proc

#     def _run():
#         global _current_proc
#         try:
#             with _play_lock:
#                 # Ø£ÙˆÙ‚Ù Ø£ÙŠ ØµÙˆØª Ø­Ø§Ù„ÙŠ
#                 if _current_proc and _current_proc.poll() is None:
#                     _current_proc.terminate()
#             # ?? audio_system.py ï¿½ ???? _play_wav
#                 _current_proc = subprocess.Popen(
#                     ["aplay", "-q", "-D", "plughw:2,0", str(path)],
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                 )
#             _current_proc.wait()
#         except Exception as e:
#             print(f"[AUDIO] aplay error: {e}")

#     threading.Thread(target=_run, daemon=True).start()


# def _play_espeak_fallback(text: str):
#     """Fallback: ÙŠÙ†Ø·Ù‚ Ø§Ù„Ù†Øµ Ù…Ø¨Ø§Ø´Ø±Ø© Ø¥Ø°Ø§ Ø§Ù„Ù…Ù„Ù ØºÙŠØ± Ù…ÙˆØ¬ÙˆØ¯."""
#     cmd = ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED]
#     threading.Thread(
#         target=lambda: subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True),
#         daemon=True
#     ).start()


# def play_object(label: str, position: str) -> bool:
#     """
#     Ø´ØºÙ‘Ù„ Ø§Ù„ØµÙˆØª Ø§Ù„Ù…Ù†Ø§Ø³Ø¨ Ù„Ù€ label + position.
#     ÙŠØ±Ø¬Ø¹ True Ø¥Ø°Ø§ Ø´ØºÙ‘Ù„ØŒ False Ø¥Ø°Ø§ Ù…Ø§ ÙÙŠ mapping.
#     """
#     key  = f"{label}_{position}"
#     entry = ARABIC_MAP.get(key)

#     if entry is None:
#         print(f"[AUDIO] No Arabic mapping for '{key}'")
#         return False

#     text, filename = entry
#     wav = _wav_path(filename)

#     if wav.exists():
#         _play_wav(wav)
#     else:
#         # Ø§Ù„Ù…Ù„Ù ØºÙŠØ± Ù…ÙˆØ¬ÙˆØ¯ â†’ fallback Ù…Ø¨Ø§Ø´Ø±
#         print(f"[AUDIO] WAV missing ({wav.name}) â€” using espeak fallback")
#         _play_espeak_fallback(text)

#     return True


# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# #  AUDIO SYSTEM CLASS
# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# class AudioSystem:
#     """
#     Ù†Ø¸Ø§Ù… Ø§Ù„Ø¥Ø¹Ù„Ø§Ù† Ø§Ù„ØµÙˆØªÙŠ â€” ÙƒØ§Ø¦Ù†Ø§Øª ÙÙ‚Ø·ØŒ Ø¨Ø§Ù„Ø¹Ø±Ø¨ÙŠ.

#     Parameters
#     ----------
#     detector : OakDetector
#         Ù†Ø³Ø®Ø© Ù…Ø´ØºÙ‘Ù„Ø© Ù…Ù† OakDetector.
#     danger_distance_mm : int
#         Ø§Ù„ÙƒØ§Ø¦Ù†Ø§Øª Ø®Ø§Ø±Ø¬ Ù‡Ø°Ø§ Ø§Ù„Ù†Ø·Ø§Ù‚ ØªÙØªØ¬Ø§Ù‡Ù„ ØµÙˆØªÙŠØ§Ù‹.
#     mock_detections : bool
#         True = Ø§Ø®ØªØ¨Ø§Ø± Ø¨Ø¯ÙˆÙ† ÙƒØ§Ù…ÙŠØ±Ø§ OAK (ÙŠØ³ØªØ®Ø¯Ù… Ø¨ÙŠØ§Ù†Ø§Øª ÙˆÙ‡Ù…ÙŠØ©).
#     """

#     def __init__(self, detector=None,
#                  danger_distance_mm: int = DEFAULT_DANGER_MM,
#                  mock_detections: bool = False):
#         self._detector  = detector
#         self._danger_mm = danger_distance_mm
#         self._mock      = mock_detections
#         self._running   = False
#         self._thread    = None

#         self._last_announced_at = 0.0
#         self._last_label        = None
#         self._last_position     = None
#         self._last_distance     = None

#     # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#     def start(self):
#         # ÙˆÙ„Ù‘Ø¯ Ø§Ù„Ø£ØµÙˆØ§Øª Ø¥Ø°Ø§ Ù„Ù… ØªÙƒÙ† Ù…ÙˆØ¬ÙˆØ¯Ø©
#         pregenerate_sounds()

#         self._running = True
#         self._thread  = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()
#         print(f"[AUDIO] Started â€” danger zone â‰¤ {self._danger_mm} mm | Arabic")

#     def stop(self):
#         self._running = False
#         if self._thread:
#             self._thread.join(timeout=3)
#         # Ø£ÙˆÙ‚Ù Ø£ÙŠ ØµÙˆØª Ø¬Ø§Ø±ÙŠ
#         global _current_proc
#         if _current_proc and _current_proc.poll() is None:
#             _current_proc.terminate()
#         print("[AUDIO] Stopped")

#     def set_danger_distance(self, mm: int):
#         self._danger_mm = mm

#     # â”€â”€ Internal loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#     def _loop(self):
#         mock_gen = _mock_detection_generator() if self._mock else None

#         while self._running:
#             # â”€â”€ Ø§Ø¬Ù„Ø¨ Ø§Ù„ÙƒØ´ÙˆÙØ§Øª â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#             if self._mock:
#                 detections = next(mock_gen)
#             else:
#                 detections = self._detector.get_detections() if self._detector else []

#             # â”€â”€ ÙÙ„ØªØ±: Ù…Ù†Ø·Ù‚Ø© Ø§Ù„Ø®Ø·Ø± ÙÙ‚Ø· â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#             in_danger = [d for d in detections
#                          if d['distance_mm'] <= self._danger_mm]

#             if not in_danger:
#                 time.sleep(0.2)
#                 continue

#             # â”€â”€ Ø§Ù„Ø£Ù‚Ø±Ø¨ ÙÙ‚Ø· â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#             in_danger.sort(key=lambda d: d['distance_mm'])
#             closest = in_danger[0]

#             # â”€â”€ Cooldown + delta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#             now           = time.time()
#             same_label    = (closest['label']    == self._last_label)
#             same_pos      = (closest['position'] == self._last_position)
#             close_dist    = (self._last_distance is not None and
#                              abs(closest['distance_mm'] - self._last_distance)
#                              < SAME_OBJECT_DELTA_MM)
#             cooldown_ok   = (now - self._last_announced_at) >= ANNOUNCE_COOLDOWN_SEC

#             if same_label and same_pos and close_dist and not cooldown_ok:
#                 time.sleep(0.2)
#                 continue

#             # â”€â”€ Ø£Ø¹Ù„Ù† â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#             played = play_object(closest['label'], closest['position'])

#             key = f"{closest['label']}_{closest['position']}"
#             ar_text = ARABIC_MAP.get(key, ("?", ""))[0]
#             print(f"[AUDIO] {ar_text}  ({closest['distance_mm']} mm"
#                   f" | {len(in_danger)} object(s) in zone)")

#             if played:
#                 self._last_announced_at = time.time()
#                 self._last_label        = closest['label']
#                 self._last_position     = closest['position']
#                 self._last_distance     = closest['distance_mm']

#             time.sleep(0.1)


# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# #  Mock generator (Ø§Ø®ØªØ¨Ø§Ø± Ø¨Ø¯ÙˆÙ† ÙƒØ§Ù…ÙŠØ±Ø§)
# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# def _mock_detection_generator():
#     scenarios = [
#         [{'label': 'person',  'position': 'right',  'distance_mm': 900}],
#         [{'label': 'person',  'position': 'right',  'distance_mm': 750},
#          {'label': 'chair',   'position': 'center', 'distance_mm': 1100}],
#         [{'label': 'chair',   'position': 'center', 'distance_mm': 300}],
#         [],
#         [{'label': 'bicycle', 'position': 'left',   'distance_mm': 1500}],  # Ø®Ø§Ø±Ø¬ Ø§Ù„Ø®Ø·Ø±
#         [{'label': 'dog',     'position': 'center', 'distance_mm': 600},
#          {'label': 'person',  'position': 'left',   'distance_mm': 800}],
#         [],
#         [],
#     ]
#     i = 0
#     while True:
#         yield scenarios[i % len(scenarios)]
#         i += 1
#         time.sleep(2.5)


# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# #  Ø§Ø®ØªØ¨Ø§Ø± Ù…Ø³ØªÙ‚Ù„
# # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# if __name__ == "__main__":
#     print("=== Audio System â€” Arabic Mock Test ===")
#     print(f"Danger zone: â‰¤ {DEFAULT_DANGER_MM} mm")
#     print(f"Sounds dir : {SOUNDS_DIR}")
#     print("Press Ctrl+C to stop\n")

#     audio = AudioSystem(mock_detections=True, danger_distance_mm=DEFAULT_DANGER_MM)
#     audio.start()

#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         audio.stop()
"""
audio_system.py â€” Smart Walker Voice Announcer
Ù…Ù„ÙØ§Øª ØµÙˆØªÙŠØ© Ø¬Ø§Ù‡Ø²Ø© ÙÙ‚Ø·ØŒ Ø¨Ø¯ÙˆÙ† TTS Ø£Ùˆ espeak.
"""

import subprocess
import threading
import time
from pathlib import Path

# â”€â”€ Ù…Ø¬Ù„Ø¯ Ù…Ù„ÙØ§ØªÙƒ Ø§Ù„ØµÙˆØªÙŠØ© â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SOUNDS = Path(__file__).parent / "sounds"
# â”€â”€ Mapping Ù…Ø¨Ø§Ø´Ø±: key â†’ Ù…Ù„Ù WAV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ø§Ù„Ù…ÙØªØ§Ø­: f"{label}_{position}"  (ÙƒÙ„Ù‡Ø§ lowercase)
# Ø§Ù„Ù‚ÙŠÙ…Ø©: Ù…Ø³Ø§Ø± Ø§Ù„Ù…Ù„Ù Ø§Ù„ØµÙˆØªÙŠ Ø§Ù„Ø¬Ø§Ù‡Ø²
SOUND_MAP: dict[str, Path] = {
    # Ø´Ø®Øµ
    "person_left":          SOUNDS / "person_left.wav",
    "person_right":         SOUNDS / "person_right.wav",
    "person_center":        SOUNDS / "person_center.wav",

    # ÙƒØ±Ø³ÙŠ
    "chair_left":           SOUNDS / "chair_left.wav",
    "chair_right":          SOUNDS / "chair_right.wav",
    "chair_center":         SOUNDS / "chair_center.wav",

    # Ø¯Ø±Ø§Ø¬Ø©
    "bicycle_left":         SOUNDS / "bicycle_left.wav",
    "bicycle_right":        SOUNDS / "bicycle_right.wav",
    "bicycle_center":       SOUNDS / "bicycle_center.wav",

    # Ø³ÙŠØ§Ø±Ø©
    "car_left":             SOUNDS / "car_left.wav",
    "car_right":            SOUNDS / "car_right.wav",
    "car_center":           SOUNDS / "car_center.wav",

    # Ø¯Ø±Ø§Ø¬Ø© Ù†Ø§Ø±ÙŠØ©
    "motorcycle_left":      SOUNDS / "motorcycle_left.wav",
    "motorcycle_right":     SOUNDS / "motorcycle_right.wav",
    "motorcycle_center":    SOUNDS / "motorcycle_center.wav",

    # Ø­Ø§ÙÙ„Ø©
    "bus_left":             SOUNDS / "bus_left.wav",
    "bus_right":            SOUNDS / "bus_right.wav",
    "bus_center":           SOUNDS / "bus_center.wav",

    # ÙƒÙ„Ø¨
    "dog_left":             SOUNDS / "dog_left.wav",
    "dog_right":            SOUNDS / "dog_right.wav",
    "dog_center":           SOUNDS / "dog_center.wav",

    # Ù‚Ø·Ø©
    "cat_left":             SOUNDS / "cat_left.wav",
    "cat_right":            SOUNDS / "cat_right.wav",
    "cat_center":           SOUNDS / "cat_center.wav",

    # Ø·Ø§ÙˆÙ„Ø©
    "diningtable_left":     SOUNDS / "diningtable_left.wav",
    "diningtable_right":    SOUNDS / "diningtable_right.wav",
    "diningtable_center":   SOUNDS / "diningtable_center.wav",

    # Ø£Ø±ÙŠÙƒØ©
    "sofa_left":            SOUNDS / "sofa_left.wav",
    "sofa_right":           SOUNDS / "sofa_right.wav",
    "sofa_center":          SOUNDS / "sofa_center.wav",

    # Ø¹Ù‚Ø¨Ø§Øª Ø¹Ø§Ù…Ø©
    "toilet_left":          SOUNDS / "toilet_left.wav",
    "toilet_right":         SOUNDS / "toilet_right.wav",
    "toilet_center":        SOUNDS / "toilet_center.wav",

    # Ø£Ø¶Ù Ø£ÙŠ label Ø¬Ø¯ÙŠØ¯ Ù‡Ù†Ø§...
}
BANKNOTE_SOUNDS = {
    "20 ILS":   SOUNDS / "20.wav",
    "50 ILS":   SOUNDS / "50.wav",
    "100 ILS":  SOUNDS / "100.wav",
    "200 ILS":  SOUNDS / "200.wav",
    "Unknown":  SOUNDS / "replace.wav",
    "Ambiguous": SOUNDS / "ambiguous.wav",
}

def play_banknote(banknote_name: str) -> bool:
    """تشغيل صوت لفئة العملة (مثل '20 ILS')"""
    wav = BANKNOTE_SOUNDS.get(banknote_name)
    if wav is None or not wav.exists():
        print(f"[AUDIO] Banknote sound missing: {banknote_name}")
        return False
    _play_wav(wav)
    return true

# â”€â”€ Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ALSA_DEVICE           = "plughw:2,0"
DEFAULT_DANGER_MM     = 1200
ANNOUNCE_COOLDOWN_SEC = 3.0
SAME_OBJECT_DELTA_MM  = 200

# â”€â”€ Playback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_current_proc: subprocess.Popen | None = None
_play_lock = threading.Lock()


def _play_wav(path: Path):
    def _run():
        global _current_proc
        try:
            with _play_lock:
                if _current_proc and _current_proc.poll() is None:
                    _current_proc.terminate()
                proc = subprocess.Popen(
                    ["mpg123", "-q", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _current_proc = proc
            if proc is not None:
                proc.wait()
        except Exception as e:
            print(f"[AUDIO] playback error: {e}")

    threading.Thread(target=_run, daemon=True).start()


def play_object(label: str, position: str) -> bool:
    """
    Ø´ØºÙ‘Ù„ Ø§Ù„Ù…Ù„Ù Ø§Ù„ØµÙˆØªÙŠ Ø§Ù„Ù…Ù†Ø§Ø³Ø¨.
    ÙŠØ±Ø¬Ø¹ True Ø¥Ø°Ø§ ÙˆØ¬Ø¯ Ø§Ù„Ù…Ù„ÙØŒ False Ø¥Ø°Ø§ Ù„Ø§.
    """
    key = f"{label}_{position}"
    wav = SOUND_MAP.get(key)

    if wav is None:
        print(f"[AUDIO] No mapping for '{key}'")
        return False

    if not wav.exists():
        print(f"[AUDIO] File missing: {wav}")
        return False

    _play_wav(wav)
    return True


def verify_sounds():
    """ØªØ­Ù‚Ù‚ Ù…Ù† ÙˆØ¬ÙˆØ¯ ÙƒÙ„ Ø§Ù„Ù…Ù„ÙØ§Øª Ø¹Ù†Ø¯ Ø§Ù„Ø¥Ù‚Ù„Ø§Ø¹ â€” Ø¨Ù„Ù‘Øº Ø¹Ù† Ø§Ù„Ù†Ø§Ù‚Øµ."""
    missing = [k for k, p in SOUND_MAP.items() if not p.exists()]
    if missing:
        print(f"[AUDIO] âš  Missing {len(missing)} sound files:")
        for k in missing:
            print(f"         {k} â†’ {SOUND_MAP[k]}")
    else:
        print(f"[AUDIO] âœ“ All {len(SOUND_MAP)} sound files found")
    return len(missing) == 0


# â”€â”€ AudioSystem Class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AudioSystem:
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

    def start(self):
        verify_sounds()   # ØªØ­Ù‚Ù‚ Ù…Ø±Ø© ÙˆØ§Ø­Ø¯Ø© Ø¹Ù†Ø¯ Ø§Ù„Ø¥Ù‚Ù„Ø§Ø¹
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[AUDIO] Started â€” danger zone â‰¤ {self._danger_mm} mm")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        global _current_proc
        if _current_proc and _current_proc.poll() is None:
            _current_proc.terminate()
        print("[AUDIO] Stopped")

    def _loop(self):
        mock_gen = _mock_detection_generator() if self._mock else None

        while self._running:
            detections = next(mock_gen) if self._mock else (
                self._detector.get_detections() if self._detector else []
            )

            in_danger = [d for d in detections if d['distance_mm'] <= self._danger_mm]

            if not in_danger:
                time.sleep(0.2)
                continue

            in_danger.sort(key=lambda d: d['distance_mm'])
            closest = in_danger[0]

            now        = time.time()
            same_label = closest['label']    == self._last_label
            same_pos   = closest['position'] == self._last_position
            close_dist = (self._last_distance is not None and
                          abs(closest['distance_mm'] - self._last_distance) < SAME_OBJECT_DELTA_MM)
            cooldown_ok = (now - self._last_announced_at) >= ANNOUNCE_COOLDOWN_SEC

            if same_label and same_pos and close_dist and not cooldown_ok:
                time.sleep(0.2)
                continue

            played = play_object(closest['label'], closest['position'])

            if played:
                self._last_announced_at = now
                self._last_label        = closest['label']
                self._last_position     = closest['position']
                self._last_distance     = closest['distance_mm']
                print(f"[AUDIO] â–¶ {closest['label']} {closest['position']} "
                      f"({closest['distance_mm']} mm)")

            time.sleep(0.1)


def _mock_detection_generator():
    import itertools
    scenarios = [
        [{'label': 'person',  'position': 'right',  'distance_mm': 900}],
        [{'label': 'chair',   'position': 'center', 'distance_mm': 500}],
        [],
        [{'label': 'car',     'position': 'left',   'distance_mm': 800}],
    ]
    for s in itertools.cycle(scenarios):
        yield s
        time.sleep(2.5)
