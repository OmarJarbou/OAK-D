# #!/usr/bin/env python3
# """
# gps_navigator.py — GPS Navigation with Waypoints
# ==================================================
# يدعم مسارات متعددة النقاط (Waypoints) لتجنب المباني والمنحنيات.

# كيف تضيف وجهة بـ Waypoints:
#   1. افتح Google Maps
#   2. انقر يميناً على كل نقطة في المسار → "ما إحداثيات هذا المكان؟"
#   3. أضفها بالترتيب في DESTINATIONS

# مثال:
#   "البيت": [
#       (31.9002, 35.2003),   # نقطة 1: عند بداية الشارع
#       (31.9005, 35.2008),   # نقطة 2: بعد المنعطف
#       (31.9010, 35.2015),   # نقطة 3: الوجهة النهائية
#   ]

# إذا الطريق مستقيم بدون عوائق — نقطة وحدة تكفي:
#   "المسجد": [(31.9010, 35.2015)]
# """

# import serial
# import threading
# import time
# import math
# import subprocess
# from pathlib import Path

# # ══════════════════════════════════════════════════════════════════════
# #  إعدادات
# # ══════════════════════════════════════════════════════════════════════

# GPS_PORT  = '/dev/ttyAMA0'
# GPS_BAUD  = 9600

# ARRIVAL_RADIUS_M  = 15     # متر — للوصول لكل waypoint
# ANNOUNCE_INTERVAL = 5.0    # ثواني بين كل تعليمة صوتية
# HEADING_TOLERANCE = 25     # درجة — قبل "استمر"

# SOUNDS_DIR  = Path("/home/lama/walker_sounds/nav")
# ALSA_DEVICE = "plughw:2,0"

# # ══════════════════════════════════════════════════════════════════════
# #  الوجهات مع Waypoints
# #  كل وجهة = قائمة من النقاط بالترتيب
# #  آخر نقطة = الوجهة النهائية
# # ══════════════════════════════════════════════════════════════════════

# DESTINATIONS = {
#     # مثال: طريق مستقيم — نقطة وحدة
#     "البيت": [
#         (31.9002, 35.2003),   # ← نقطة وسطى (عند التقاطع مثلاً)
#         (31.9010, 35.2015),   # ← الوجهة النهائية دائماً آخر عنصر
#     ],

#     # مثال: طريق فيه منعطفين
#     "المسجد": [
#         (31.9005, 35.2005),   # ← بعد أول منعطف
#         (31.9008, 35.2012),   # ← بعد ثاني منعطف
#         (31.9010, 35.2015),   # ← الوجهة النهائية
#     ],

#     # مثال: طريق مستقيم — نقطة وحدة تكفي
#     "الصيدلية": [
#         (31.9005, 35.2008),
#     ],

#     "المستشفى": [
#         (31.8995, 35.1995),
#         (31.8990, 35.1990),
#     ],

#     "المدرسة": [
#         (31.9020, 35.2025),
#     ],
# }

# # ══════════════════════════════════════════════════════════════════════
# #  الأصوات العربية
# # ══════════════════════════════════════════════════════════════════════

# NAV_AUDIO = {
#     "turn_right":       ("اتجه يميناً",              "nav_turn_right"),
#     "turn_left":        ("اتجه يساراً",              "nav_turn_left"),
#     "go_straight":      ("استمر للأمام",             "nav_go_straight"),
#     "arrived":          ("لقد وصلت إلى وجهتك",      "nav_arrived"),
#     "waypoint_reached": ("استمر نحو الوجهة",        "nav_waypoint"),
#     "no_gps":           ("في انتظار إشارة GPS",      "nav_no_gps"),
#     "nav_started":      ("بدأ الإرشاد",              "nav_started"),
#     "nav_cancelled":    ("تم إلغاء الإرشاد",        "nav_cancelled"),
#     # وجهات
#     "dest_البيت":       ("الوجهة: البيت",            "nav_dest_home"),
#     "dest_المسجد":      ("الوجهة: المسجد",           "nav_dest_mosque"),
#     "dest_الصيدلية":    ("الوجهة: الصيدلية",        "nav_dest_pharmacy"),
#     "dest_المستشفى":    ("الوجهة: المستشفى",        "nav_dest_hospital"),
#     "dest_المدرسة":     ("الوجهة: المدرسة",          "nav_dest_school"),
# }

# ESPEAK_VOICE = "ar"
# ESPEAK_SPEED = "130"
# ESPEAK_AMP   = "180"


# # ══════════════════════════════════════════════════════════════════════
# #  توليد ملفات الصوت
# # ══════════════════════════════════════════════════════════════════════

# def pregenerate_nav_sounds(force: bool = False):
#     SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
#     check = subprocess.run(["which", "espeak-ng"], capture_output=True)
#     if check.returncode != 0:
#         print("[GPS-NAV] espeak-ng not found")
#         return False
#     generated = skipped = 0
#     for key, (text, filename) in NAV_AUDIO.items():
#         out = SOUNDS_DIR / f"{filename}.wav"
#         if out.exists() and not force:
#             skipped += 1
#             continue
#         cmd = ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED,
#                "-a", ESPEAK_AMP, "-w", str(out)]
#         result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
#         if result.returncode == 0:
#             print(f"[GPS-NAV] Generated: {out.name}")
#             generated += 1
#     print(f"[GPS-NAV] Sounds: {generated} generated, {skipped} exist")
#     return True


# _play_proc = None
# _play_lock = threading.Lock()

# def _play_sound(key: str):
#     global _play_proc
#     entry = NAV_AUDIO.get(key)
#     if not entry:
#         return
#     text, filename = entry
#     wav = SOUNDS_DIR / f"{filename}.wav"

#     def _run():
#         global _play_proc
#         try:
#             with _play_lock:
#                 if _play_proc and _play_proc.poll() is None:
#                     _play_proc.terminate()
#                 if wav.exists():
#                     _play_proc = subprocess.Popen(
#                         ["aplay", "-q", "-D", ALSA_DEVICE, str(wav)],
#                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#                     )
#                 else:
#                     _play_proc = subprocess.Popen(
#                         ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED],
#                         input=text.encode("utf-8"),
#                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#                     )
#             if _play_proc:
#                 _play_proc.wait()
#         except Exception as e:
#             print(f"[GPS-NAV] Audio error: {e}")

#     threading.Thread(target=_run, daemon=True).start()


# # ══════════════════════════════════════════════════════════════════════
# #  حسابات جغرافية
# # ══════════════════════════════════════════════════════════════════════

# def _haversine_m(lat1, lon1, lat2, lon2) -> float:
#     R = 6371000.0
#     p1, p2 = math.radians(lat1), math.radians(lat2)
#     dp = math.radians(lat2 - lat1)
#     dl = math.radians(lon2 - lon1)
#     a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
#     return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
#     p1, p2 = math.radians(lat1), math.radians(lat2)
#     dl = math.radians(lon2 - lon1)
#     x  = math.sin(dl) * math.cos(p2)
#     y  = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
#     return (math.degrees(math.atan2(x, y)) + 360) % 360

# def _angle_diff(target: float, current: float) -> float:
#     return (target - current + 180) % 360 - 180


# # ══════════════════════════════════════════════════════════════════════
# #  قارئ GPS (NEO-6M)
# # ══════════════════════════════════════════════════════════════════════

# class GpsReader:
#     def __init__(self, port=GPS_PORT, baud=GPS_BAUD):
#         self._port    = port
#         self._baud    = baud
#         self._lock    = threading.Lock()
#         self._running = False
#         self._thread  = None
#         self.lat      = None
#         self.lon      = None
#         self.heading  = None
#         self.speed_ms = 0.0
#         self.fix      = False

#     def start(self):
#         self._running = True
#         self._thread  = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()
#         print(f"[GPS] Reader started on {self._port}")

#     def stop(self):
#         self._running = False

#     def get_position(self):
#         with self._lock:
#             return self.lat, self.lon, self.heading, self.fix

#     def _loop(self):
#         while self._running:
#             try:
#                 ser = serial.Serial(self._port, self._baud, timeout=1)
#                 print(f"[GPS] Serial open: {self._port}")
#                 while self._running:
#                     try:
#                         line = ser.readline().decode('ascii', errors='replace').strip()
#                         self._parse_nmea(line)
#                     except Exception:
#                         pass
#                 ser.close()
#             except serial.SerialException as e:
#                 print(f"[GPS] Cannot open {self._port}: {e} — retry in 5s")
#                 time.sleep(5)

#     def _parse_nmea(self, line: str):
#         if not line.startswith('$'):
#             return
#         if '*' in line:
#             data, checksum = line[1:].rsplit('*', 1)
#             calc = 0
#             for c in data:
#                 calc ^= ord(c)
#             if f"{calc:02X}" != checksum[:2].upper():
#                 return

#         parts   = line.split(',')
#         sentence = parts[0]

#         if sentence in ('$GPGGA', '$GNGGA') and len(parts) >= 10:
#             if parts[6] in ('0', ''):
#                 with self._lock:
#                     self.fix = False
#                 return
#             try:
#                 lat = self._nmea_to_deg(parts[2], parts[3])
#                 lon = self._nmea_to_deg(parts[4], parts[5])
#                 with self._lock:
#                     if self.lat and lat:
#                         if self.heading is None:
#                             dist = _haversine_m(self.lat, self.lon, lat, lon)
#                             if dist > 2.0:
#                                 self.heading = _bearing_deg(self.lat, self.lon, lat, lon)
#                     self.lat = lat
#                     self.lon = lon
#                     self.fix = True
#             except Exception:
#                 pass

#         elif sentence in ('$GPRMC', '$GNRMC') and len(parts) >= 10:
#             if parts[2] != 'A':
#                 with self._lock:
#                     self.fix = False
#                 return
#             try:
#                 lat    = self._nmea_to_deg(parts[3], parts[4])
#                 lon    = self._nmea_to_deg(parts[5], parts[6])
#                 speed  = float(parts[7]) * 0.514444 if parts[7] else 0.0
#                 course = float(parts[8]) if parts[8] else None
#                 with self._lock:
#                     self.lat      = lat
#                     self.lon      = lon
#                     self.speed_ms = speed
#                     if course is not None and course >= 0:
#                         self.heading = course
#                     self.fix = True
#             except Exception:
#                 pass

#     @staticmethod
#     def _nmea_to_deg(value: str, direction: str):
#         if not value:
#             return None
#         dot    = value.index('.')
#         deg    = float(value[:dot-2])
#         mins   = float(value[dot-2:])
#         result = deg + mins / 60.0
#         if direction in ('S', 'W'):
#             result = -result
#         return result


# # ══════════════════════════════════════════════════════════════════════
# #  نظام الإرشاد مع Waypoints
# # ══════════════════════════════════════════════════════════════════════

# class GpsNavigator:
#     """
#     نظام إرشاد GPS بـ Waypoints.

#     يمشي من نقطة لنقطة بالترتيب حتى يصل للوجهة النهائية.
#     عند الوصول لكل waypoint يقول "استمر نحو الوجهة" ويكمل للتالية.
#     عند الوصول للنقطة الأخيرة يقول "لقد وصلت".
#     """

#     def __init__(self, gps_reader: GpsReader):
#         self._gps          = gps_reader
#         self._dest_name    = None
#         self._waypoints    = []    # قائمة [(lat,lon), ...]
#         self._wp_index     = 0     # الـ waypoint الحالي
#         self._active       = False
#         self._running      = False
#         self._thread       = None
#         self._lock         = threading.Lock()
#         self._last_say     = 0.0
#         self._last_key     = None

#     # ── Public API ────────────────────────────────────────────────────

#     def start(self):
#         pregenerate_nav_sounds()
#         self._running = True
#         self._thread  = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()
#         print("[GPS-NAV] Navigator started (Waypoints mode)")

#     def stop(self):
#         self._running = False
#         if self._thread:
#             self._thread.join(timeout=3)
#         print("[GPS-NAV] Stopped")

#     def set_destination(self, name: str) -> bool:
#         name   = name.strip()
#         points = DESTINATIONS.get(name)
#         if not points:
#             print(f"[GPS-NAV] Unknown destination: '{name}'")
#             return False

#         # تأكد إنها قائمة من tuples
#         if isinstance(points, tuple):
#             points = [points]

#         with self._lock:
#             self._dest_name = name
#             self._waypoints = list(points)
#             self._wp_index  = 0
#             self._active    = True
#             self._last_say  = 0.0
#             self._last_key  = None

#         total = len(self._waypoints)
#         print(f"[GPS-NAV] Destination: {name} — {total} waypoint(s)")
#         for i, wp in enumerate(self._waypoints):
#             print(f"  [{i+1}/{total}] {wp}")

#         _play_sound(f"dest_{name}")
#         time.sleep(1.5)
#         _play_sound("nav_started")
#         return True

#     def set_destination_coords(self, name: str, lat: float, lon: float):
#         """وجهة بإحداثيات مباشرة — waypoint وحيد."""
#         with self._lock:
#             self._dest_name = name
#             self._waypoints = [(lat, lon)]
#             self._wp_index  = 0
#             self._active    = True
#             self._last_say  = 0.0
#             self._last_key  = None
#         print(f"[GPS-NAV] Custom destination: {name} ({lat}, {lon})")
#         _play_sound("nav_started")

#     def cancel(self):
#         with self._lock:
#             self._active    = False
#             self._dest_name = None
#             self._waypoints = []
#             self._wp_index  = 0
#         _play_sound("nav_cancelled")
#         print("[GPS-NAV] Cancelled")

#     def is_active(self) -> bool:
#         with self._lock:
#             return self._active

#     def get_status(self) -> dict:
#         lat, lon, heading, fix = self._gps.get_position()
#         with self._lock:
#             active     = self._active
#             dest_name  = self._dest_name
#             wp_index   = self._wp_index
#             waypoints  = list(self._waypoints)

#         current_wp  = waypoints[wp_index] if waypoints and wp_index < len(waypoints) else None
#         dist        = None
#         bearing     = None
#         if fix and lat and current_wp:
#             dist    = _haversine_m(lat, lon, current_wp[0], current_wp[1])
#             bearing = _bearing_deg(lat, lon, current_wp[0], current_wp[1])

#         return {
#             "active":          active,
#             "destination":     dest_name,
#             "waypoint_index":  wp_index,
#             "total_waypoints": len(waypoints),
#             "fix":             fix,
#             "lat":             lat,
#             "lon":             lon,
#             "heading":         heading,
#             "bearing":         bearing,
#             "distance_m":      dist,
#         }

#     # ── Internal loop ────────────────────────────────────────────────

#     def _loop(self):
#         print("[GPS-NAV] Loop running")
#         while self._running:
#             with self._lock:
#                 active    = self._active
#                 waypoints = list(self._waypoints)
#                 wp_index  = self._wp_index

#             if not active or not waypoints:
#                 time.sleep(0.5)
#                 continue

#             # الـ waypoint الحالي
#             if wp_index >= len(waypoints):
#                 # وصلنا لكل النقاط
#                 _play_sound("arrived")
#                 print(f"[GPS-NAV] Arrived at {self._dest_name}!")
#                 with self._lock:
#                     self._active = False
#                 time.sleep(3)
#                 continue

#             target_lat, target_lon = waypoints[wp_index]
#             is_last = (wp_index == len(waypoints) - 1)

#             lat, lon, heading, fix = self._gps.get_position()

#             # لا إشارة GPS
#             if not fix or lat is None:
#                 self._say_if_due("no_gps", force_interval=10.0)
#                 time.sleep(1.0)
#                 continue

#             dist    = _haversine_m(lat, lon, target_lat, target_lon)
#             bearing = _bearing_deg(lat, lon, target_lat, target_lon)

#             total = len(waypoints)
#             print(f"[GPS-NAV] WP[{wp_index+1}/{total}] "
#                   f"dist={dist:.0f}m  bearing={bearing:.0f}°  heading={heading}°")

#             # وصلنا لهذا الـ waypoint؟
#             if dist <= ARRIVAL_RADIUS_M:
#                 if is_last:
#                     # الوجهة النهائية
#                     _play_sound("arrived")
#                     print(f"[GPS-NAV] ✓ Final destination reached: {self._dest_name}")
#                     with self._lock:
#                         self._active = False
#                     time.sleep(3)
#                 else:
#                     # waypoint وسطي — انتقل للتالي
#                     print(f"[GPS-NAV] ✓ Waypoint {wp_index+1} reached — next!")
#                     _play_sound("waypoint_reached")
#                     with self._lock:
#                         self._wp_index += 1
#                     self._last_key = None   # reset لإعطاء تعليمة جديدة فوراً
#                     time.sleep(1.0)
#                 continue

#             # heading ما توفر بعد (واقف)
#             if heading is None:
#                 self._say_if_due("go_straight")
#                 time.sleep(1.0)
#                 continue

#             # قرار الاتجاه
#             diff = _angle_diff(bearing, heading)

#             if abs(diff) <= HEADING_TOLERANCE:
#                 key = "go_straight"
#             elif diff > 0:
#                 key = "turn_right"
#             else:
#                 key = "turn_left"

#             self._say_if_due(key)
#             time.sleep(0.5)

#     def _say_if_due(self, key: str, force_interval: float = None):
#         now      = time.time()
#         interval = force_interval if force_interval else ANNOUNCE_INTERVAL
#         if key != self._last_key or (now - self._last_say) >= interval:
#             _play_sound(key)
#             self._last_say = now
#             self._last_key = key


# # ══════════════════════════════════════════════════════════════════════
# #  اختبار مستقل
# # ══════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import sys
#     print("=== GPS Navigator — Waypoints Test ===")
#     print(f"Destinations: {list(DESTINATIONS.keys())}")
#     for name, wps in DESTINATIONS.items():
#         print(f"  {name}: {len(wps)} waypoint(s)")
#     print()

#     gps = GpsReader()
#     gps.start()

#     nav = GpsNavigator(gps)
#     nav.start()

#     dest = sys.argv[1] if len(sys.argv) > 1 else list(DESTINATIONS.keys())[0]
#     print(f"Navigating to: {dest}")
#     nav.set_destination(dest)

#     try:
#         while True:
#             s = nav.get_status()
#             print(f"  WP[{s['waypoint_index']+1}/{s['total_waypoints']}] "
#                   f"dist={s['distance_m']}m  heading={s['heading']}°")
#             time.sleep(3)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         nav.stop()
#         gps.stop()
# //////////////////////////////////////////////////////
#!/usr/bin/env python3
"""
gps_navigator.py — GPS Navigation with Waypoints
==================================================
يدعم مسارات متعددة النقاط (Waypoints) لتجنب المباني والمنحنيات.

كيف تضيف وجهة بـ Waypoints:
  1. افتح Google Maps
  2. انقر يميناً على كل نقطة في المسار → "ما إحداثيات هذا المكان؟"
  3. أضفها بالترتيب في DESTINATIONS

مثال:
  "البيت": [
      (31.9002, 35.2003),   # نقطة 1: عند بداية الشارع
      (31.9005, 35.2008),   # نقطة 2: بعد المنعطف
      (31.9010, 35.2015),   # نقطة 3: الوجهة النهائية
  ]

إذا الطريق مستقيم بدون عوائق — نقطة وحدة تكفي:
  "المسجد": [(31.9010, 35.2015)]
"""

import serial
import threading
import time
import math
import subprocess
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  إعدادات
# ══════════════════════════════════════════════════════════════════════

GPS_PORT  = '/dev/ttyAMA0'
GPS_BAUD  = 9600

ARRIVAL_RADIUS_M  = 15     # متر — للوصول لكل waypoint
ANNOUNCE_INTERVAL = 5.0    # ثواني بين كل تعليمة صوتية
HEADING_TOLERANCE = 25     # درجة — قبل "استمر"

SOUNDS_DIR  = Path("/home/lama/walker_sounds/nav")
ALSA_DEVICE = "plughw:2,0"

# ══════════════════════════════════════════════════════════════════════
#  الوجهات مع Waypoints
#  كل وجهة = قائمة من النقاط بالترتيب
#  آخر نقطة = الوجهة النهائية
# ══════════════════════════════════════════════════════════════════════

DESTINATIONS = {
    # مثال: طريق مستقيم — نقطة وحدة
    "البيت": [
        (31.9002, 35.2003),   # ← نقطة وسطى (عند التقاطع مثلاً)
        (31.9010, 35.2015),   # ← الوجهة النهائية دائماً آخر عنصر
    ],

    # مثال: طريق فيه منعطفين
    "المسجد": [
        (31.9005, 35.2005),   # ← بعد أول منعطف
        (31.9008, 35.2012),   # ← بعد ثاني منعطف
        (31.9010, 35.2015),   # ← الوجهة النهائية
    ],

    # مثال: طريق مستقيم — نقطة وحدة تكفي
    "الصيدلية": [
        (31.9005, 35.2008),
    ],

    "المستشفى": [
        (31.8995, 35.1995),
        (31.8990, 35.1990),
    ],

    "المدرسة": [
        (31.9020, 35.2025),
    ],
}

# ══════════════════════════════════════════════════════════════════════
#  الأصوات العربية
# ══════════════════════════════════════════════════════════════════════

NAV_AUDIO = {
    "turn_right":       ("اتجه يميناً",              "nav_turn_right"),
    "turn_left":        ("اتجه يساراً",              "nav_turn_left"),
    "go_straight":      ("استمر للأمام",             "nav_go_straight"),
    "arrived":          ("لقد وصلت إلى وجهتك",      "nav_arrived"),
    "waypoint_reached": ("استمر نحو الوجهة",        "nav_waypoint"),
    "no_gps":           ("في انتظار إشارة GPS",      "nav_no_gps"),
    "nav_started":      ("بدأ الإرشاد",              "nav_started"),
    "nav_cancelled":    ("تم إلغاء الإرشاد",        "nav_cancelled"),
    # وجهات
    "dest_البيت":       ("الوجهة: البيت",            "nav_dest_home"),
    "dest_المسجد":      ("الوجهة: المسجد",           "nav_dest_mosque"),
    "dest_الصيدلية":    ("الوجهة: الصيدلية",        "nav_dest_pharmacy"),
    "dest_المستشفى":    ("الوجهة: المستشفى",        "nav_dest_hospital"),
    "dest_المدرسة":     ("الوجهة: المدرسة",          "nav_dest_school"),
}

ESPEAK_VOICE = "ar"
ESPEAK_SPEED = "130"
ESPEAK_AMP   = "180"


# ══════════════════════════════════════════════════════════════════════
#  توليد ملفات الصوت
# ══════════════════════════════════════════════════════════════════════

def pregenerate_nav_sounds(force: bool = False):
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    check = subprocess.run(["which", "espeak-ng"], capture_output=True)
    if check.returncode != 0:
        print("[GPS-NAV] espeak-ng not found")
        return False
    generated = skipped = 0
    for key, (text, filename) in NAV_AUDIO.items():
        out = SOUNDS_DIR / f"{filename}.wav"
        if out.exists() and not force:
            skipped += 1
            continue
        cmd = ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED,
               "-a", ESPEAK_AMP, "-w", str(out)]
        result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
        if result.returncode == 0:
            print(f"[GPS-NAV] Generated: {out.name}")
            generated += 1
    print(f"[GPS-NAV] Sounds: {generated} generated, {skipped} exist")
    return True


_play_proc = None
_play_lock = threading.Lock()

def _play_sound(key: str):
    global _play_proc
    entry = NAV_AUDIO.get(key)
    if not entry:
        return
    text, filename = entry
    wav = SOUNDS_DIR / f"{filename}.wav"

    def _run():
        global _play_proc
        try:
            with _play_lock:
                if _play_proc and _play_proc.poll() is None:
                    _play_proc.terminate()
                if wav.exists():
                    _play_proc = subprocess.Popen(
                        ["aplay", "-q", "-D", ALSA_DEVICE, str(wav)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                else:
                    _play_proc = subprocess.Popen(
                        ["espeak-ng", "-v", ESPEAK_VOICE, "-s", ESPEAK_SPEED],
                        input=text.encode("utf-8"),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
            if _play_proc:
                _play_proc.wait()
        except Exception as e:
            print(f"[GPS-NAV] Audio error: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
#  حسابات جغرافية
# ══════════════════════════════════════════════════════════════════════

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def _angle_diff(target: float, current: float) -> float:
    return (target - current + 180) % 360 - 180


# ══════════════════════════════════════════════════════════════════════
#  قارئ GPS (NEO-6M)
# ══════════════════════════════════════════════════════════════════════

class GpsReader:
    def __init__(self, port=GPS_PORT, baud=GPS_BAUD):
        self._port    = port
        self._baud    = baud
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self.lat      = None
        self.lon      = None
        self.heading  = None
        self.speed_ms = 0.0
        self.fix      = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[GPS] Reader started on {self._port}")

    def stop(self):
        self._running = False

    def get_position(self):
        with self._lock:
            return self.lat, self.lon, self.heading, self.fix

    def _loop(self):
        while self._running:
            try:
                ser = serial.Serial(self._port, self._baud, timeout=1)
                print(f"[GPS] Serial open: {self._port}")
                while self._running:
                    try:
                        line = ser.readline().decode('ascii', errors='replace').strip()
                        self._parse_nmea(line)
                    except Exception:
                        pass
                ser.close()
            except serial.SerialException as e:
                print(f"[GPS] Cannot open {self._port}: {e} — retry in 5s")
                time.sleep(5)

    def _parse_nmea(self, line: str):
        if not line.startswith('$'):
            return
        if '*' in line:
            data, checksum = line[1:].rsplit('*', 1)
            calc = 0
            for c in data:
                calc ^= ord(c)
            if f"{calc:02X}" != checksum[:2].upper():
                return

        parts   = line.split(',')
        sentence = parts[0]

        if sentence in ('$GPGGA', '$GNGGA') and len(parts) >= 10:
            if parts[6] in ('0', ''):
                with self._lock:
                    self.fix = False
                return
            try:
                lat = self._nmea_to_deg(parts[2], parts[3])
                lon = self._nmea_to_deg(parts[4], parts[5])
                with self._lock:
                    if self.lat and lat:
                        if self.heading is None:
                            dist = _haversine_m(self.lat, self.lon, lat, lon)
                            if dist > 2.0:
                                self.heading = _bearing_deg(self.lat, self.lon, lat, lon)
                    self.lat = lat
                    self.lon = lon
                    self.fix = True
            except Exception:
                pass

        elif sentence in ('$GPRMC', '$GNRMC') and len(parts) >= 10:
            if parts[2] != 'A':
                with self._lock:
                    self.fix = False
                return
            try:
                lat    = self._nmea_to_deg(parts[3], parts[4])
                lon    = self._nmea_to_deg(parts[5], parts[6])
                speed  = float(parts[7]) * 0.514444 if parts[7] else 0.0
                course = float(parts[8]) if parts[8] else None
                with self._lock:
                    self.lat      = lat
                    self.lon      = lon
                    self.speed_ms = speed
                    if course is not None and course >= 0:
                        self.heading = course
                    self.fix = True
            except Exception:
                pass

    @staticmethod
    def _nmea_to_deg(value: str, direction: str):
        if not value:
            return None
        dot    = value.index('.')
        deg    = float(value[:dot-2])
        mins   = float(value[dot-2:])
        result = deg + mins / 60.0
        if direction in ('S', 'W'):
            result = -result
        return result


# ══════════════════════════════════════════════════════════════════════
#  نظام الإرشاد مع Waypoints
# ══════════════════════════════════════════════════════════════════════

class GpsNavigator:
    """
    نظام إرشاد GPS بـ Waypoints.

    يمشي من نقطة لنقطة بالترتيب حتى يصل للوجهة النهائية.
    عند الوصول لكل waypoint يقول "استمر نحو الوجهة" ويكمل للتالية.
    عند الوصول للنقطة الأخيرة يقول "لقد وصلت".
    """

    def __init__(self, gps_reader: GpsReader):
        self._gps          = gps_reader
        self._dest_name    = None
        self._waypoints    = []    # قائمة [(lat,lon), ...]
        self._wp_index     = 0     # الـ waypoint الحالي
        self._active       = False
        self._running      = False
        self._thread       = None
        self._lock         = threading.Lock()
        self._last_say     = 0.0
        self._last_key     = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        pregenerate_nav_sounds()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[GPS-NAV] Navigator started (Waypoints mode)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[GPS-NAV] Stopped")

    def set_destination(self, name: str) -> bool:
        name   = name.strip()
        points = DESTINATIONS.get(name)
        if not points:
            print(f"[GPS-NAV] Unknown destination: '{name}'")
            return False

        # تأكد إنها قائمة من tuples
        if isinstance(points, tuple):
            points = [points]

        with self._lock:
            self._dest_name = name
            self._waypoints = list(points)
            self._wp_index  = 0
            self._active    = True
            self._last_say  = 0.0
            self._last_key  = None

        total = len(self._waypoints)
        print(f"[GPS-NAV] Destination: {name} — {total} waypoint(s)")
        for i, wp in enumerate(self._waypoints):
            print(f"  [{i+1}/{total}] {wp}")

        _play_sound(f"dest_{name}")
        time.sleep(1.5)
        _play_sound("nav_started")
        return True

    def set_destination_coords(self, name: str, lat: float, lon: float):
        """وجهة بإحداثيات مباشرة — waypoint وحيد."""
        with self._lock:
            self._dest_name = name
            self._waypoints = [(lat, lon)]
            self._wp_index  = 0
            self._active    = True
            self._last_say  = 0.0
            self._last_key  = None
        print(f"[GPS-NAV] Custom destination: {name} ({lat}, {lon})")
        _play_sound("nav_started")

    def cancel(self):
        with self._lock:
            self._active    = False
            self._dest_name = None
            self._waypoints = []
            self._wp_index  = 0
        _play_sound("nav_cancelled")
        print("[GPS-NAV] Cancelled")

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_status(self) -> dict:
        lat, lon, heading, fix = self._gps.get_position()
        with self._lock:
            active     = self._active
            dest_name  = self._dest_name
            wp_index   = self._wp_index
            waypoints  = list(self._waypoints)

        current_wp  = waypoints[wp_index] if waypoints and wp_index < len(waypoints) else None
        dist        = None
        bearing     = None
        if fix and lat and current_wp:
            dist    = _haversine_m(lat, lon, current_wp[0], current_wp[1])
            bearing = _bearing_deg(lat, lon, current_wp[0], current_wp[1])

        return {
            "active":          active,
            "destination":     dest_name,
            "waypoint_index":  wp_index,
            "total_waypoints": len(waypoints),
            "fix":             fix,
            "lat":             lat,
            "lon":             lon,
            "heading":         heading,
            "bearing":         bearing,
            "distance_m":      dist,
        }

    # ── Internal loop ────────────────────────────────────────────────

    def _loop(self):
        print("[GPS-NAV] Loop running")
        while self._running:
            with self._lock:
                active    = self._active
                waypoints = list(self._waypoints)
                wp_index  = self._wp_index

            if not active or not waypoints:
                time.sleep(0.5)
                continue

            # الـ waypoint الحالي
            if wp_index >= len(waypoints):
                # وصلنا لكل النقاط
                _play_sound("arrived")
                print(f"[GPS-NAV] Arrived at {self._dest_name}!")
                with self._lock:
                    self._active = False
                time.sleep(3)
                continue

            target_lat, target_lon = waypoints[wp_index]
            is_last = (wp_index == len(waypoints) - 1)

            lat, lon, heading, fix = self._gps.get_position()

            # لا إشارة GPS
            if not fix or lat is None:
                self._say_if_due("no_gps", force_interval=10.0)
                time.sleep(1.0)
                continue

            dist    = _haversine_m(lat, lon, target_lat, target_lon)
            bearing = _bearing_deg(lat, lon, target_lat, target_lon)

            total = len(waypoints)
            print(f"[GPS-NAV] WP[{wp_index+1}/{total}] "
                  f"dist={dist:.0f}m  bearing={bearing:.0f}°  heading={heading}°")

            # وصلنا لهذا الـ waypoint؟
            if dist <= ARRIVAL_RADIUS_M:
                if is_last:
                    # الوجهة النهائية
                    _play_sound("arrived")
                    print(f"[GPS-NAV] ✓ Final destination reached: {self._dest_name}")
                    with self._lock:
                        self._active = False
                    time.sleep(3)
                else:
                    # waypoint وسطي — انتقل للتالي
                    print(f"[GPS-NAV] ✓ Waypoint {wp_index+1} reached — next!")
                    _play_sound("waypoint_reached")
                    with self._lock:
                        self._wp_index += 1
                    self._last_key = None   # reset لإعطاء تعليمة جديدة فوراً
                    time.sleep(1.0)
                continue

            # heading ما توفر بعد (واقف)
            if heading is None:
                self._say_if_due("go_straight")
                time.sleep(1.0)
                continue

            # قرار الاتجاه
            diff = _angle_diff(bearing, heading)

            if abs(diff) <= HEADING_TOLERANCE:
                key = "go_straight"
            elif diff > 0:
                key = "turn_right"
            else:
                key = "turn_left"

            self._say_if_due(key)
            time.sleep(0.5)

    def _say_if_due(self, key: str, force_interval: float = None):
        now      = time.time()
        interval = force_interval if force_interval else ANNOUNCE_INTERVAL
        if key != self._last_key or (now - self._last_say) >= interval:
            _play_sound(key)
            self._last_say = now
            self._last_key = key


# ══════════════════════════════════════════════════════════════════════
#  اختبار مستقل
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("=== GPS Navigator — Waypoints Test ===")
    print(f"Destinations: {list(DESTINATIONS.keys())}")
    for name, wps in DESTINATIONS.items():
        print(f"  {name}: {len(wps)} waypoint(s)")
    print()

    gps = GpsReader()
    gps.start()

    nav = GpsNavigator(gps)
    nav.start()

    dest = sys.argv[1] if len(sys.argv) > 1 else list(DESTINATIONS.keys())[0]
    print(f"Navigating to: {dest}")
    nav.set_destination(dest)

    try:
        while True:
            s = nav.get_status()
            print(f"  WP[{s['waypoint_index']+1}/{s['total_waypoints']}] "
                  f"dist={s['distance_m']}m  heading={s['heading']}°")
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        nav.stop()
        gps.stop()


# ══════════════════════════════════════════════════════════════════════
#  GPS Relay Handler
#  يستمع على Serial1 لطلبات من الأردوينو:
#    GET:GPS       → يرد بالموقع لإرساله للتطبيق
#    GET:GPS:SOS   → يرد بالموقع كـ SOS
#    NAV:GOTO:xxx  → يبدأ إرشاد لوجهة (من أزرار الأردوينو)
#    NAV:CANCEL    → يلغي الإرشاد
# ══════════════════════════════════════════════════════════════════════

class GpsRelayHandler:
    """
    يعمل في thread مستقل — لا يتعارض مع main loop.

    الاستخدام في main.py:
        relay = GpsRelayHandler(
            arduino_ser=ser,
            gps_reader=gps,
            navigator=navigator,
        )
        relay.start()
        # ...
        relay.stop()
    """

    def __init__(self, arduino_ser, gps_reader: GpsReader,
                 navigator: GpsNavigator):
        self._ser      = arduino_ser
        self._gps      = gps_reader
        self._nav      = navigator
        self._running  = False
        self._thread   = None
        self._buf      = ""

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[GPS-RELAY] Started")

    def stop(self):
        self._running = False
        print("[GPS-RELAY] Stopped")

    def _loop(self):
        while self._running:
            try:
                if self._ser.in_waiting:
                    chunk = self._ser.read(self._ser.in_waiting)
                    self._buf += chunk.decode(errors='replace')
                    while '\n' in self._buf:
                        line, self._buf = self._buf.split('\n', 1)
                        self._process(line.strip())
                time.sleep(0.05)
            except Exception as e:
                if self._running:
                    print(f"[GPS-RELAY] Error: {e}")
                time.sleep(1)

    def _process(self, line: str):
        if not line:
            return

        # ── GET:GPS — الأردوينو يطلب الموقع ──────────────────────────
        if line in ("GET:GPS", "GET:GPS:SOS"):
            is_sos = "SOS" in line
            lat, lon, _, fix = self._gps.get_position()

            if fix and lat and lon:
                if is_sos:
                    reply = f"GPS:SOS:{lat:.6f},{lon:.6f}\n"
                else:
                    reply = f"GPS:{lat:.6f},{lon:.6f}\n"
            else:
                reply = "GPS:NOT_READY\n"

            try:
                self._ser.write(reply.encode())
                print(f"[GPS-RELAY] Replied: {reply.strip()}")
            except Exception as e:
                print(f"[GPS-RELAY] Write error: {e}")

        # ── NAV:GOTO:وجهة — زر ضُغط على الأردوينو ────────────────────
        elif line.startswith("NAV:GOTO:"):
            dest = line[9:].strip()
            print(f"[GPS-RELAY] NAV GOTO: '{dest}'")
            if not self._nav.set_destination(dest):
                print(f"[GPS-RELAY] Unknown destination: '{dest}'")

        # ── NAV:CANCEL ────────────────────────────────────────────────
        elif line == "NAV:CANCEL":
            print("[GPS-RELAY] NAV CANCEL")
            self._nav.cancel()