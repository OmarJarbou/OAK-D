
import serial
import threading
import time
import math
import subprocess
from collections import deque
from pathlib import Path

# ----------------------------------------------------------------------
#  ??????? ??????
# ----------------------------------------------------------------------

GPS_PORT = '/dev/ttyAMA2'
GPS_BAUD = 9600

SOUNDS_DIR  = Path("/home/lama/walker_sounds/nav")
ALSA_DEVICE = "plughw:2,0"

# ----------------------------------------------------------------------
#  ??? ????? ???????
# ----------------------------------------------------------------------

ARRIVAL_RADIUS_M      = 12      
LOOKAHEAD_RADIUS_M    = 25     
ANNOUNCE_INTERVAL_SEC = 4.0     
HEADING_TOLERANCE_DEG = 22      

# ???? ?????? (Kalman gain: 0=???? ????, 1=?? ?????)
POS_KALMAN_GAIN       = 0.35

# ???? ??? heading
MIN_SPEED_MS          = 0.5     
HEADING_BUFFER_SIZE   = 6       # ???? ?? buffer
HEADING_ALPHA         = 0.35    # EMA ??? heading

# Outlier rejection
MAX_BEARING_JUMP_DEG  = 110     # ????? ?????? ?? bearing ? ?????
MAX_POS_JUMP_M        = 30      # ???? GPS ? ?????

# ????? ??????
CONFIRM_FRAMES        = 2       # frames ??? ???????

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

DESTINATIONS = {
    "?????": [
        (31.9002, 35.2003),
        (31.9010, 35.2015),
    ],
    "??????": [
        (31.9005, 35.2005),
        (31.9008, 35.2012),
        (31.9010, 35.2015),
    ],
    "????????": [
        (31.9005, 35.2008),
    ],
    "????????": [
        (31.8995, 35.1995),
        (31.8990, 35.1990),
    ],
    "???????": [
        (31.9020, 35.2025),
    ],
}

# ----------------------------------------------------------------------
#  ??????? ???????
# ----------------------------------------------------------------------

NAV_AUDIO = {
    "turn_right":       ("???? ??????",           "nav_turn_right"),
    "turn_left":        ("???? ??????",           "nav_turn_left"),
    "go_straight":      ("????? ??????",          "nav_go_straight"),
    "arrived":          ("??? ???? ??? ?????",   "nav_arrived"),
    "waypoint_reached": ("????? ??? ??????",     "nav_waypoint"),
    "no_gps":           ("?? ?????? ????? GPS",   "nav_no_gps"),
    "nav_started":      ("??? ???????",           "nav_started"),
    "nav_cancelled":    ("?? ????? ???????",     "nav_cancelled"),
    "dest_?????":       ("??????: ?????",         "nav_dest_home"),
    "dest_??????":      ("??????: ??????",        "nav_dest_mosque"),
    "dest_????????":    ("??????: ????????",     "nav_dest_pharmacy"),
    "dest_????????":    ("??????: ????????",     "nav_dest_hospital"),
    "dest_???????":     ("??????: ???????",       "nav_dest_school"),
}

ESPEAK_VOICE = "ar"
ESPEAK_SPEED = "130"
ESPEAK_AMP   = "180"
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
                        ["espeak-ng", "-v", ESPEAK_VOICE,
                         "-s", ESPEAK_SPEED, "-a", ESPEAK_AMP],
                        input=text.encode("utf-8"),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
            if _play_proc:
                _play_proc.wait()
        except Exception as e:
            print(f"[GPS-NAV] Audio error: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ----------------------------------------------------------------------
#  ?????? ???????
# ----------------------------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R  = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def _angle_diff(target: float, current: float) -> float:
    """????? ??? ??????? ?? [-180, +180]  (????=????? ????=????)"""
    return (target - current + 180) % 360 - 180


def _circular_mean(angles: list) -> float:
    """????? ????? ?????? ????? (?????? ?? wrap 0/360)"""
    sin_s = sum(math.sin(math.radians(a)) for a in angles)
    cos_s = sum(math.cos(math.radians(a)) for a in angles)
    return (math.degrees(math.atan2(sin_s, cos_s)) + 360) % 360


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class PositionFilter:
    """
    ???? ?? ????? GPS ????? ?? ??????? ?????? ?? gain ????.
    ???? ??????? ???? ?????? MAX_POS_JUMP_M (GPS glitch).
    """

    def __init__(self, gain: float = POS_KALMAN_GAIN,
                 max_jump_m: float = MAX_POS_JUMP_M):
        self._gain     = gain
        self._max_jump = max_jump_m
        self._lat      = None
        self._lon      = None

    def update(self, lat: float, lon: float):
        if self._lat is None:
            self._lat, self._lon = lat, lon
            return self._lat, self._lon

        dist = _haversine_m(self._lat, self._lon, lat, lon)
        if dist > self._max_jump:
            print(f"[POS-FILTER] Outlier rejected - jump={dist:.1f}m")
            return self._lat, self._lon

        self._lat += self._gain * (lat - self._lat)
        self._lon += self._gain * (lon - self._lon)
        return self._lat, self._lon

    def get(self):
        return self._lat, self._lon

    @property
    def ready(self) -> bool:
        return self._lat is not None


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class HeadingEstimator:
    """
    ???? heading ????? ??:
      1. $RMC NMEA course ??? ?????? >= MIN_SPEED_MS  (???)
      2. circular mean ??? bearings ?? buffer ??????  (???????)

    ????? EMA ?????? ?????? ????????.
    ???? ?????? ???? ?????? MAX_BEARING_JUMP_DEG ???? ?????.
    """

    def __init__(self, buffer_size: int = HEADING_BUFFER_SIZE,
                 alpha: float = HEADING_ALPHA,
                 min_speed: float = MIN_SPEED_MS):
        self._buf       = deque(maxlen=buffer_size)
        self._alpha     = alpha
        self._min_speed = min_speed
        self._heading   = None
def update(self, lat: float, lon: float,
               speed_ms: float, nmea_course: float = None) -> bool:
        """
        ??? ???? ?????? heading. ????? True ??? ??? heading ????.
        """
        self._buf.append((lat, lon))

        # -- 1. NMEA course (?????? ????) -----------------------------
        if nmea_course is not None and speed_ms >= self._min_speed:
            self._apply_raw(nmea_course)
            return True

        # -- 2. buffer ?????? (???????) -------------------------------
        if len(self._buf) < 2 or speed_ms < self._min_speed:
            return self._heading is not None

        bearings = []
        pts = list(self._buf)
        for i in range(1, len(pts)):
            d = _haversine_m(pts[i-1][0], pts[i-1][1],
                             pts[i][0],   pts[i][1])
            if d >= 1.0:
                b = _bearing_deg(pts[i-1][0], pts[i-1][1],
                                 pts[i][0],   pts[i][1])
                bearings.append(b)

        if not bearings:
            return self._heading is not None

        self._apply_raw(_circular_mean(bearings))
        return True

def _apply_raw(self, raw: float):
            """???? EMA + outlier damping ??? ???? heading ???."""
            if self._heading is None:
                self._heading = raw
                return
            diff = _angle_diff(raw, self._heading)
            if abs(diff) > MAX_BEARING_JUMP_DEG:
                # ???? ?????? ????? ?? ????? ?????
                print(f"[HDG] Jump {diff:.0f}deg - dampened")
                raw = (self._heading + math.copysign(
                       MAX_BEARING_JUMP_DEG * 0.5, diff)) % 360
                diff = _angle_diff(raw, self._heading)
            self._heading = (self._heading + self._alpha * diff + 360) % 360

def get(self):
            return self._heading

def reset(self):
            self._buf.clear()

class GpsReader:
    """
    ???? NMEA ?????? PositionFilter + HeadingEstimator.
    get_position() ????? ????? ???????? ???????????.
    get_raw() ??? debug ???.
    """

    def __init__(self, port=GPS_PORT, baud=GPS_BAUD):
        self._port       = port
        self._baud       = baud
        self._lock       = threading.Lock()
        self._running    = False
        self._thread     = None
        self._pos_filter = PositionFilter()
        self._hdg_est    = HeadingEstimator()
        self._raw_lat    = None
        self._raw_lon    = None
        self._speed_ms   = 0.0
        self._fix        = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[GPS] Reader v2 started on {self._port}")

    def stop(self):
        self._running = False

    def get_position(self):
        """????? (lat_filtered, lon_filtered, heading_smooth, fix)"""
        with self._lock:
            lat, lon = self._pos_filter.get()
            heading  = self._hdg_est.get()
            fix      = self._fix
        return lat, lon, heading, fix

    def get_raw(self):
        with self._lock:
            return self._raw_lat, self._raw_lon, self._speed_ms, self._fix

    # -- internal ------------------------------------------------------

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
                print(f"[GPS] Cannot open {self._port}: {e} - retry in 5s")
                time.sleep(5)

    def _parse_nmea(self, line: str):
        if not line.startswith('$'):
            return

        # ???? checksum
        if '*' in line:
            data, cs = line[1:].rsplit('*', 1)
            calc = 0
            for c in data:
                calc ^= ord(c)
            if f"{calc:02X}" != cs[:2].upper():
                return

        parts    = line.split(',')
        sentence = parts[0]

        # -- GGA -------------------------------------------------------
        if sentence in ('$GPGGA', '$GNGGA') and len(parts) >= 10:
            if parts[6] in ('0', ''):
                with self._lock:
                    self._fix = False
                return
            try:
                lat = self._nmea_to_deg(parts[2], parts[3])
                lon = self._nmea_to_deg(parts[4], parts[5])
                if lat is None or lon is None:
                    return
                with self._lock:
                    self._raw_lat = lat
                    self._raw_lon = lon
                    self._fix     = True
                    fl, flon = self._pos_filter.update(lat, lon)
                    self._hdg_est.update(fl, flon, self._speed_ms)
            except Exception:
                pass

        # -- RMC -------------------------------------------------------
        elif sentence in ('$GPRMC', '$GNRMC') and len(parts) >= 10:
            if parts[2] != 'A':
                with self._lock:
                    self._fix = False
                return
            try:
                lat   = self._nmea_to_deg(parts[3], parts[4])
                lon   = self._nmea_to_deg(parts[5], parts[6])
                if lat is None or lon is None:
                    return
                speed = float(parts[7]) * 0.514444 if parts[7] else 0.0
                course = None
                if parts[8]:
                    try:
                        c = float(parts[8])
                        if 0.0 <= c < 360.0:
                            course = c
                    except ValueError:
                        pass
                with self._lock:
                    self._raw_lat  = lat
                    self._raw_lon  = lon
                    self._speed_ms = speed
                    self._fix      = True
                    fl, flon = self._pos_filter.update(lat, lon)
                    self._hdg_est.update(fl, flon, speed, course)
            except Exception:
                pass
                
@staticmethod
def _nmea_to_deg(value: str, direction: str):
        if not value:
            return None
        try:
            dot    = value.index('.')
            deg    = float(value[:dot - 2])
            mins   = float(value[dot - 2:])
            result = deg + mins / 60.0
            if direction in ('S', 'W'):
                result = -result
            return result
        except Exception:
            return None


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class GpsNavigator:


    def __init__(self, gps_reader: GpsReader):
        self._gps           = gps_reader
        self._dest_name     = None
        self._waypoints     = []
        self._wp_index      = 0
        self._active        = False
        self._running       = False
        self._thread        = None
        self._lock          = threading.Lock()
        self._last_say      = 0.0
        self._last_key      = None
        self._pending_key   = None
        self._pending_count = 0

    # -- Public API ----------------------------------------------------

    def start(self):
        pregenerate_nav_sounds()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[GPS-NAV] Navigator v2 started")  
        
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
        if isinstance(points, tuple):
            points = [points]
        with self._lock:
            self._dest_name     = name
            self._waypoints     = list(points)
            self._wp_index      = 0
            self._active        = True
            self._last_say      = 0.0
            self._last_key      = None
            self._pending_key   = None
            self._pending_count = 0
        total = len(self._waypoints)
        print(f"[GPS-NAV] Destination: {name} - {total} waypoint(s)")
        _play_sound(f"dest_{name}")
        time.sleep(1.5)
        _play_sound("nav_started")
        return True

    def set_destination_coords(self, name: str, lat: float, lon: float):
        with self._lock:
            self._dest_name     = name
            self._waypoints     = [(lat, lon)]
            self._wp_index      = 0
            self._active        = True
            self._last_say      = 0.0
            self._last_key      = None
            self._pending_key   = None
            self._pending_count = 0
        print(f"[GPS-NAV] Custom dest: {name} ({lat:.6f}, {lon:.6f})")
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
            active    = self._active
            dest_name = self._dest_name
            wp_index  = self._wp_index
            waypoints = list(self._waypoints)
        cur_wp   = waypoints[wp_index] if waypoints and wp_index < len(waypoints) else None
        dist = bearing = None
        if fix and lat and cur_wp:
            dist    = _haversine_m(lat, lon, cur_wp[0], cur_wp[1])
            bearing = _bearing_deg(lat, lon, cur_wp[0], cur_wp[1])
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

    # -- Main loop -----------------------------------------------------

    def _loop(self):
        print("[GPS-NAV] Loop running")
        prev_bearing = None

        while self._running:
            with self._lock:
                active    = self._active
                waypoints = list(self._waypoints)
                wp_index  = self._wp_index

            if not active or not waypoints:
                time.sleep(0.5)
                prev_bearing = None
                continue

            if wp_index >= len(waypoints):
                _play_sound("arrived")
                print(f"[GPS-NAV] ? Arrived at {self._dest_name}")
                with self._lock:
                    self._active = False
                time.sleep(3)
                continue
                
            target_lat, target_lon = waypoints[wp_index]
            is_last  = (wp_index == len(waypoints) - 1)
            has_next = not is_last
            total    = len(waypoints)

            lat, lon, heading, fix    = self._gps.get_position()
            _, _, speed_ms, _         = self._gps.get_raw()

            # ?? ????? GPS
            if not fix or lat is None:
                self._say_if_due("no_gps", force_interval=10.0)
                time.sleep(1.0)
                prev_bearing = None
                continue

            dist    = _haversine_m(lat, lon, target_lat, target_lon)
            bearing = _bearing_deg(lat, lon, target_lat, target_lon)

            # -- Outlier rejection ??? bearing -------------------------
            if prev_bearing is not None:
                jump = abs(_angle_diff(bearing, prev_bearing))
                if jump > MAX_BEARING_JUMP_DEG:
                    print(f"[NAV] Bearing outlier: {prev_bearing:.0f} {bearing:.0f}deg "
                    f"jump={jump:.0f}deg - rejected")
                    bearing = prev_bearing
                else:
                    prev_bearing = bearing
            else:
                prev_bearing = bearing

            # -- Lookahead: ??? bearing ?? waypoint ?????? -------------
            effective_bearing = bearing
            if has_next and dist <= LOOKAHEAD_RADIUS_M:
                nxt_lat, nxt_lon = waypoints[wp_index + 1]
                nxt_bearing      = _bearing_deg(lat, lon, nxt_lat, nxt_lon)
                blend            = 1.0 - (dist / LOOKAHEAD_RADIUS_M)
                diff_nxt         = _angle_diff(nxt_bearing, bearing)
                effective_bearing = (bearing + blend * diff_nxt + 360) % 360
                print(f"[NAV] Lookahead blend={blend:.2f}  "
                      f"cur={bearing:.0f}deg nxt={nxt_bearing:.0f}deg "
                      f"?eff={effective_bearing:.0f}deg")

            print(f"[GPS-NAV] WP[{wp_index+1}/{total}] "
                  f"dist={dist:.0f}m  bearing={effective_bearing:.0f}deg "
                  f"heading={heading}deg  speed={speed_ms:.1f}m/s")

            # -- ?????? ------------------------------------------------
            if dist <= ARRIVAL_RADIUS_M:
                if is_last:
                    _play_sound("arrived")
                    print(f"[GPS-NAV] ? Final destination: {self._dest_name}")
                    with self._lock:
                        self._active = False
                    prev_bearing = None
                    time.sleep(3)
                else:
                    print(f"[GPS-NAV] ? WP{wp_index+1} reached - next!")
                    _play_sound("waypoint_reached")
                    with self._lock:
                        self._wp_index += 1
                    self._last_key      = None
                    self._pending_key   = None
                    self._pending_count = 0
                    prev_bearing        = None
                    time.sleep(1.0)
                continue

            # heading ??? ???? ???
            if heading is None:
                self._say_if_due("go_straight")
                time.sleep(1.0)
                continue

            # -- ???? ??????? ------------------------------------------
            diff    = _angle_diff(effective_bearing, heading)
            raw_key = self._diff_to_key(diff)

            confirmed = self._confirm(raw_key)
            if confirmed:
                self._say_if_due(confirmed)

            time.sleep(0.5)
    def _diff_to_key(self, diff: float) -> str:
        if abs(diff) <= HEADING_TOLERANCE_DEG:
            return "go_straight"
        return "turn_right" if diff > 0 else "turn_left"

    def _confirm(self, key: str):
        """????? key ??? ??? CONFIRM_FRAMES frames ???????."""
        if key == self._pending_key:
            self._pending_count += 1
        else:
            self._pending_key   = key
            self._pending_count = 1
        return key if self._pending_count >= CONFIRM_FRAMES else None

    def _say_if_due(self, key: str, force_interval: float = None):
        now      = time.time()
        interval = force_interval or ANNOUNCE_INTERVAL_SEC
        if key != self._last_key or (now - self._last_say) >= interval:
            _play_sound(key)
            self._last_say = now
            self._last_key = key


# ----------------------------------------------------------------------
#  GpsRelayHandler (???? ????? ?????)
# ----------------------------------------------------------------------

class GpsRelayHandler:
    def __init__(self, arduino_ser, gps_reader: GpsReader,
                 navigator: GpsNavigator):
        self._ser     = arduino_ser
        self._gps     = gps_reader
        self._nav     = navigator
        self._running = False
        self._thread  = None
        self._buf     = ""

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
        if line in ("GET:GPS", "GET:GPS:SOS"):
            is_sos           = "SOS" in line
            lat, lon, _, fix = self._gps.get_position()
            if fix and lat and lon:
                prefix = "GPS:SOS:" if is_sos else "GPS:"
                reply  = f"{prefix}{lat:.6f},{lon:.6f}\n"
            else:
                reply = "GPS:NOT_READY\n"
            try:
                self._ser.write(reply.encode())
                print(f"[GPS-RELAY] Replied: {reply.strip()}")
            except Exception as e:
                print(f"[GPS-RELAY] Write error: {e}")
        elif line.startswith("NAV:GOTO:"):
            dest = line[9:].strip()
            if not self._nav.set_destination(dest):
                print(f"[GPS-RELAY] Unknown destination: '{dest}'")
        elif line == "NAV:CANCEL":
            self._nav.cancel()

if __name__ == "__main__":
    import sys
    print("=== GPS Navigator v2 ===")
    print(f"Destinations: {list(DESTINATIONS.keys())}")

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
            r_lat, r_lon, spd, _ = gps.get_raw()
            print(f"  WP[{s['waypoint_index']+1}/{s['total_waypoints']}] "
                  f"dist={s['distance_m']}m  brg={s['bearing']}deg  "
                  f"hdg={s['heading']}deg  spd={spd:.1f}m/s  "
                  f"raw=({r_lat},{r_lon})")
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        nav.stop()
        gps.stop()

