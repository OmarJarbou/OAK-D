
# """
# Smart Walker — LiDAR-Only Navigator
# ====================================
# Raspberry Pi  ←Serial1→  Arduino Mega  (115200 baud, pins 18/19 on Mega)

# Flow
# ----
#   LidarScanner thread  →  latest full-rotation scan dict
#   Main loop (LOOP_HZ)  →  decide()  →  CMD:ANGLE:<n> / CMD:STOP / CMD:FREE

# Commands sent to Arduino
# ------------------------
#   CMD:ASSIST         engage stepper motor (sent once at startup)
#   CMD:ANGLE:<n>      steer to n  (-100=full left, 0=straight, +100=full right)
#   CMD:STOP           emergency stop (brake + vibration on Arduino side)
#   CMD:FREE           release motor (sent on clean exit)

# Configuration checklist (edit the CAPS constants below)
# ---------------------------------------------------------
#   SERIAL_PORT   — UART connected to Arduino Serial1 (TX1/RX1)
#   LIDAR_PORT    — USB port of the RPLIDAR C1
#   FRONT_HEADING — angle in navigator.py matching "forward" direction
# """

# import serial
# import time
# import signal
# import sys

# from lidar_scanner import LidarScanner
# from navigator     import decide

# # ── Serial (Pi → Arduino) ─────────────────────────────────────────────
# SERIAL_PORT  = '/dev/serial0'
# SERIAL_BAUD  = 115200

# # ── RPLIDAR C1 ────────────────────────────────────────────────────────
# LIDAR_PORT   = '/dev/ttyUSB0'
# LIDAR_BAUD   = 460800

# # ── Loop tuning ───────────────────────────────────────────────────────
# LOOP_HZ           = 10
# ANGLE_DEAD_BAND   = 10
# STOP_HOLD_SEC     = 0.8
# SMOOTH_ALPHA      = 0.45
# CLEAR_FRAMES_CTR  = 4
# LOCK_UNLOCK_DELAY = 1.5
# READY_TIMEOUT_SEC = 2.0


# # ── Serial helpers ────────────────────────────────────────────────────

# def send(ser: serial.Serial, cmd: str):
#     ser.reset_output_buffer()
#     line = cmd.strip() + '\n'
#     ser.write(line.encode())
#     ser.flush()
#     print(f"[TX] {cmd}")


# def drain_rx(ser: serial.Serial) -> str:
#     lines = []
#     while ser.in_waiting:
#         try:
#             line = ser.readline().decode(errors='replace').strip()
#             if line:
#                 print(f"[RX] {line}")
#                 lines.append(line)
#         except Exception:
#             pass
#     return ' '.join(lines)


# def query_pot(ser: serial.Serial, timeout: float = 2.0):
#     send(ser, "CMD:POT")
#     deadline = time.time() + timeout
#     while time.time() < deadline:
#         rx = drain_rx(ser)
#         for token in rx.split():
#             if token.startswith("STATUS:POT:"):
#                 try:
#                     return int(token.split(":", 2)[2])
#                 except ValueError:
#                     pass
#         time.sleep(0.05)
#     return None


# def main():
#     print("=== Smart Walker — LiDAR Navigator ===")

#     try:
#         ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
#         ser.dtr = False
#         ser.rts = False
#         time.sleep(2)
#         print(f"[SERIAL] {SERIAL_PORT} @ {SERIAL_BAUD} baud  OK")
#     except serial.SerialException as exc:
#         print(f"[SERIAL] Cannot open {SERIAL_PORT}: {exc}")
#         sys.exit(1)

#     scanner = LidarScanner(port=LIDAR_PORT, baudrate=LIDAR_BAUD)
#     try:
#         scanner.start()
#         time.sleep(1)
#         print(f"[LIDAR]  {LIDAR_PORT} @ {LIDAR_BAUD} baud  OK")
#     except Exception as exc:
#         print(f"[LIDAR]  Failed to start: {exc}")
#         ser.close()
#         sys.exit(1)

#     def shutdown(sig=None, frame=None):
#         print("\n[MAIN] Shutting down…")
#         scanner.stop()
#         try:
#             send(ser, "CMD:FREE")
#             ser.close()
#         except Exception:
#             pass
#         sys.exit(0)

#     signal.signal(signal.SIGINT,  shutdown)
#     signal.signal(signal.SIGTERM, shutdown)

#     # ── Wait for RFID authorization ───────────────────────────────────
#     print("[AUTH] Waiting for RFID authorization…")
#     while True:
#         rx = drain_rx(ser)
#         if 'AUTHORIZED' in rx:
#             print("[AUTH] Card accepted ✓")
#             break
#         time.sleep(0.05)

#     # ── Wait for auth sequence to finish ─────────────────────────────
#     print("[AUTH] Waiting for auth sequence to finish...")
#     while True:
#         rx = drain_rx(ser)
#         if any(k in rx for k in ('AUTHORIZED_READY', 'AUTH_DONE', 'STATUS:FREE', 'STATUS:REACHED')):
#             print("[AUTH] Auth sequence complete")
#             break
#         time.sleep(0.1)

#     # ── Confirm center position ───────────────────────────────────────
#     time.sleep(1.0)
#     print("[AUTH] Confirming center position...")
#     send(ser, "CMD:ANGLE:0")
#     t_confirm = time.time()
#     while True:
#         rx = drain_rx(ser)
#         if any(k in rx for k in ('READY', 'AT_TARGET', 'REACHED')):
#             print("[AUTH] Center confirmed — starting navigation\n")
#             break
#         if time.time() - t_confirm > 5.0:
#             print("[AUTH] WARNING: no confirm — starting anyway\n")
#             break
#         time.sleep(0.05)

#     drain_rx(ser)

#     # ── State variables ───────────────────────────────────────────────
#     last_action      = None
#     last_angle_sent  = None
#     stop_until       = 0.0
#     smoothed_angle   = 0.0
#     clear_count      = 0
#     locked_dir       = None
#     lock_time        = 0.0
#     interval         = 1.0 / LOOP_HZ

#     # ── ACK-gate state ────────────────────────────────────────────────
#     arduino_ready  = True
#     ready_sent_at  = 0.0
#     pending_action = None
#     pending_angle  = 0

#     # ── LiDAR warmup ─────────────────────────────────────────────────
#     WARMUP_SEC = 2.0
#     print(f"[MAIN] LiDAR warmup ({WARMUP_SEC}s)…")
#     warmup_end = time.time() + WARMUP_SEC
#     while time.time() < warmup_end:
#         drain_rx(ser)
#         time.sleep(0.05)
#     smoothed_angle = 0.0
#     clear_count    = 0
#     print("[MAIN] Navigation running. Press Ctrl+C to stop.\n")

#     while True:
#         t0   = time.time()
#         scan = scanner.get_scan()

#         # ── Drain RX ─────────────────────────────────────────────────
#         rx  = drain_rx(ser)
#         now = time.time()

#         # ── Parse Arduino lock feedback ───────────────────────────────
#         if 'LOCKED_LEFT' in rx:
#             locked_dir = 'left';  lock_time = now
#         elif 'LOCKED_RIGHT' in rx:
#             locked_dir = 'right'; lock_time = now
#         elif any(k in rx for k in ('REACHED', 'AT_TARGET', 'UNLOCKED', 'STOPPED', 'FREE')):
#             locked_dir = None

#         # ── ACK: mark Arduino ready ───────────────────────────────────
#         if any(k in rx for k in ('REACHED', 'AT_TARGET', 'READY', 'INTERRUPTED')):
#             if not arduino_ready:
#                 print("[ACK] Arduino ready")
#             arduino_ready = True

#         # ── Timeout: force ready if ACK never arrived ─────────────────
#         if not arduino_ready and (now - ready_sent_at) > READY_TIMEOUT_SEC:
#             print("[ACK] Timeout — assuming ready")
#             arduino_ready = True

#         # ── Auto-unlock motor after lock delay ────────────────────────
#         if locked_dir and (now - lock_time) > LOCK_UNLOCK_DELAY:
#             send(ser, "CMD:UNLOCK")
#             locked_dir = None

#         # ── Navigation decision ───────────────────────────────────────
#         action, raw_angle = decide(scan)

#         # ── STOP holdoff ──────────────────────────────────────────────
#         if action == 'STOP':
#             stop_until = now + STOP_HOLD_SEC
#         if now < stop_until:
#             action, raw_angle = 'STOP', 0

#         # ── EMA smoothing + clear-path hysteresis ─────────────────────
#         if action == 'STEER':
#             clear_count    = 0
#             smoothed_angle = SMOOTH_ALPHA * raw_angle + (1 - SMOOTH_ALPHA) * smoothed_angle
#         elif action == 'CENTER':
#             clear_count += 1
#             if clear_count >= CLEAR_FRAMES_CTR:
#                 smoothed_angle = (1 - SMOOTH_ALPHA) * smoothed_angle

#         angle = int(round(smoothed_angle)) if action in ('STEER', 'CENTER') else raw_angle

#         # ── Respect motor lock direction ──────────────────────────────
#         if   locked_dir == 'right' and angle > 0: angle = 0
#         elif locked_dir == 'left'  and angle < 0: angle = 0

#         # ── STOP bypass ───────────────────────────────────────────────
#         if action == 'STOP':
#             if last_action != 'STOP':
#                 send(ser, "CMD:STOP")
#             arduino_ready   = True
#             pending_action  = None
#             last_angle_sent = None
#             last_action     = 'STOP'
#             elapsed = time.time() - t0
#             spare   = interval - elapsed
#             if spare > 0:
#                 time.sleep(spare)
#             continue

#         # ── Accumulate latest decision while gate is closed ───────────
#         pending_action = action
#         pending_angle  = angle

#         # ── On ACK: send if worthwhile ────────────────────────────────
#         if arduino_ready and pending_action is not None:
#             if pending_action in ('STEER', 'CENTER'):
#                 if (last_angle_sent is None or
#                         abs(pending_angle - last_angle_sent) > ANGLE_DEAD_BAND):
#                     send(ser, f"CMD:ANGLE:{pending_angle}")
#                     last_angle_sent = pending_angle
#                     arduino_ready   = False
#                     ready_sent_at   = now
#             elif pending_action == 'NODATA':
#                 if last_action != 'NODATA':
#                     print("[NAV] WARNING: no LiDAR data — holding position")
#             pending_action = None

#         last_action = action

#         # ── Sleep to maintain LOOP_HZ ─────────────────────────────────
#         elapsed = time.time() - t0
#         spare   = interval - elapsed
#         if spare > 0:
#             time.sleep(spare)


# if __name__ == '__main__':
#     main()
# ////////////////////////////////////////////////////////////////////////

#!/usr/bin/env python3
"""
Smart Walker — Full System (LiDAR + GPS Navigation + Buttons)
=============================================================
Raspberry Pi  ←Serial1→  Arduino Mega  (115200 baud, pins 18/19 on Mega)

Threads:
  1. LidarScanner   — تجنب العوائق
  2. GpsReader      — قراءة NEO-6M
  3. GpsNavigator   — إرشاد صوتي
  4. ButtonHandler  — أزرار GPIO على Pi
  5. GpsRelayHandler— يرد على GET:GPS من الأردوينو
  6. Main loop      — قرارات التوجيه (LiDAR)
"""

import serial
import time
import signal
import sys

from lidar_scanner  import LidarScanner
from navigator      import decide
from gps_navigator  import GpsReader, GpsNavigator, GpsRelayHandler, DESTINATIONS
from button_handler import ButtonHandler

# ══════════════════════════════════════════════════════════════════════
#  إعدادات Hardware
# ══════════════════════════════════════════════════════════════════════

SERIAL_PORT  = '/dev/serial0'   # Pi UART → Arduino Serial1
SERIAL_BAUD  = 115200

LIDAR_PORT   = '/dev/ttyUSB0'
LIDAR_BAUD   = 460800

GPS_PORT     = '/dev/ttyAMA0'   # NEO-6M
GPS_BAUD     = 9600

# ══════════════════════════════════════════════════════════════════════
#  Loop tuning
# ══════════════════════════════════════════════════════════════════════

LOOP_HZ           = 10
ANGLE_DEAD_BAND   = 10
STOP_HOLD_SEC     = 0.8
SMOOTH_ALPHA      = 0.45
CLEAR_FRAMES_CTR  = 4
LOCK_UNLOCK_DELAY = 1.5
READY_TIMEOUT_SEC = 2.0


# ══════════════════════════════════════════════════════════════════════
#  Serial helpers
# ══════════════════════════════════════════════════════════════════════

def send(ser: serial.Serial, cmd: str):
    ser.reset_output_buffer()
    line = cmd.strip() + '\n'
    ser.write(line.encode())
    ser.flush()
    print(f"[TX] {cmd}")


def drain_rx(ser: serial.Serial) -> str:
    lines = []
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='replace').strip()
            if line:
                print(f"[RX] {line}")
                lines.append(line)
        except Exception:
            pass
    return ' '.join(lines)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=== Smart Walker — Full System ===")

    # ── Arduino Serial ────────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
        ser.dtr = False
        ser.rts = False
        time.sleep(2)
        print(f"[SERIAL] {SERIAL_PORT} @ {SERIAL_BAUD} baud  OK")
    except serial.SerialException as exc:
        print(f"[SERIAL] Cannot open {SERIAL_PORT}: {exc}")
        sys.exit(1)

    # ── LiDAR ─────────────────────────────────────────────────────────
    scanner = LidarScanner(port=LIDAR_PORT, baudrate=LIDAR_BAUD)
    try:
        scanner.start()
        time.sleep(1)
        print(f"[LIDAR] {LIDAR_PORT} @ {LIDAR_BAUD} baud  OK")
    except Exception as exc:
        print(f"[LIDAR] Failed to start: {exc}")
        ser.close()
        sys.exit(1)

    # ── GPS ───────────────────────────────────────────────────────────
    gps = GpsReader(port=GPS_PORT, baud=GPS_BAUD)
    gps.start()

    # ── GPS Navigator ─────────────────────────────────────────────────
    navigator = GpsNavigator(gps_reader=gps)
    navigator.start()

    # ── GPS Relay — يرد على GET:GPS و NAV: من الأردوينو ──────────────
    relay = GpsRelayHandler(
        arduino_ser=ser,
        gps_reader=gps,
        navigator=navigator,
    )
    relay.start()

    # ── Button callback ───────────────────────────────────────────────
    def on_button(name: str):
        if name == "إلغاء":
            navigator.cancel()
        else:
            navigator.set_destination(name)
            print(f"[BTN] Navigate to: {name}")

    buttons = ButtonHandler(callback=on_button)
    buttons.start()

    # ── Shutdown ──────────────────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        print("\n[MAIN] Shutting down…")
        buttons.stop()
        relay.stop()
        navigator.stop()
        gps.stop()
        scanner.stop()
        try:
            send(ser, "CMD:FREE")
            ser.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ══════════════════════════════════════════════════════════════════
    #  انتظار RFID
    # ══════════════════════════════════════════════════════════════════
    print("[AUTH] Waiting for RFID authorization…")
    while True:
        rx = drain_rx(ser)
        if 'AUTHORIZED' in rx:
            print("[AUTH] Card accepted ✓")
            break
        time.sleep(0.05)

    print("[AUTH] Waiting for auth sequence to finish...")
    while True:
        rx = drain_rx(ser)
        if any(k in rx for k in ('AUTHORIZED_READY', 'AUTH_DONE', 'STATUS:FREE', 'STATUS:REACHED')):
            print("[AUTH] Auth sequence complete")
            break
        time.sleep(0.1)

    time.sleep(1.0)
    print("[AUTH] Confirming center position...")
    send(ser, "CMD:ANGLE:0")
    t_confirm = time.time()
    while True:
        rx = drain_rx(ser)
        if any(k in rx for k in ('READY', 'AT_TARGET', 'REACHED')):
            print("[AUTH] Center confirmed — starting navigation\n")
            break
        if time.time() - t_confirm > 5.0:
            print("[AUTH] WARNING: no confirm — starting anyway\n")
            break
        time.sleep(0.05)

    drain_rx(ser)

    # ══════════════════════════════════════════════════════════════════
    #  State variables
    # ══════════════════════════════════════════════════════════════════
    last_action      = None
    last_angle_sent  = None
    stop_until       = 0.0
    smoothed_angle   = 0.0
    clear_count      = 0
    locked_dir       = None
    lock_time        = 0.0
    interval         = 1.0 / LOOP_HZ

    arduino_ready  = True
    ready_sent_at  = 0.0
    pending_action = None
    pending_angle  = 0

    # ── LiDAR warmup ─────────────────────────────────────────────────
    WARMUP_SEC = 2.0
    print(f"[MAIN] LiDAR warmup ({WARMUP_SEC}s)…")
    warmup_end = time.time() + WARMUP_SEC
    while time.time() < warmup_end:
        drain_rx(ser)
        time.sleep(0.05)
    smoothed_angle = 0.0
    clear_count    = 0
    print("[MAIN] Running. Press Ctrl+C to stop.\n")
    print(f"[MAIN] Destinations: {list(DESTINATIONS.keys())}")
    print(f"[MAIN] Buttons: البيت(GPIO17) المسجد(GPIO27) إلغاء(GPIO22)\n")

    # ══════════════════════════════════════════════════════════════════
    #  Main Loop — LiDAR steering
    # ══════════════════════════════════════════════════════════════════
    while True:
        t0   = time.time()
        scan = scanner.get_scan()

        rx  = drain_rx(ser)
        now = time.time()

        # ── Arduino lock feedback ─────────────────────────────────────
        if 'LOCKED_LEFT' in rx:
            locked_dir = 'left';  lock_time = now
        elif 'LOCKED_RIGHT' in rx:
            locked_dir = 'right'; lock_time = now
        elif any(k in rx for k in ('REACHED', 'AT_TARGET', 'UNLOCKED', 'STOPPED', 'FREE')):
            locked_dir = None

        # ── ACK ───────────────────────────────────────────────────────
        if any(k in rx for k in ('REACHED', 'AT_TARGET', 'READY', 'INTERRUPTED')):
            if not arduino_ready:
                print("[ACK] Arduino ready")
            arduino_ready = True

        if not arduino_ready and (now - ready_sent_at) > READY_TIMEOUT_SEC:
            print("[ACK] Timeout — assuming ready")
            arduino_ready = True

        if locked_dir and (now - lock_time) > LOCK_UNLOCK_DELAY:
            send(ser, "CMD:UNLOCK")
            locked_dir = None

        # ── LiDAR decision ────────────────────────────────────────────
        action, raw_angle = decide(scan)

        if action == 'STOP':
            stop_until = now + STOP_HOLD_SEC
        if now < stop_until:
            action, raw_angle = 'STOP', 0

        if action == 'STEER':
            clear_count    = 0
            smoothed_angle = SMOOTH_ALPHA * raw_angle + (1 - SMOOTH_ALPHA) * smoothed_angle
        elif action == 'CENTER':
            clear_count += 1
            if clear_count >= CLEAR_FRAMES_CTR:
                smoothed_angle = (1 - SMOOTH_ALPHA) * smoothed_angle

        angle = int(round(smoothed_angle)) if action in ('STEER', 'CENTER') else raw_angle

        if   locked_dir == 'right' and angle > 0: angle = 0
        elif locked_dir == 'left'  and angle < 0: angle = 0

        # ── STOP ──────────────────────────────────────────────────────
        if action == 'STOP':
            if last_action != 'STOP':
                send(ser, "CMD:STOP")
            arduino_ready   = True
            pending_action  = None
            last_angle_sent = None
            last_action     = 'STOP'
            elapsed = time.time() - t0
            spare   = interval - elapsed
            if spare > 0:
                time.sleep(spare)
            continue

        pending_action = action
        pending_angle  = angle

        if arduino_ready and pending_action is not None:
            if pending_action in ('STEER', 'CENTER'):
                if (last_angle_sent is None or
                        abs(pending_angle - last_angle_sent) > ANGLE_DEAD_BAND):
                    send(ser, f"CMD:ANGLE:{pending_angle}")
                    last_angle_sent = pending_angle
                    arduino_ready   = False
                    ready_sent_at   = now
            elif pending_action == 'NODATA':
                if last_action != 'NODATA':
                    print("[NAV] WARNING: no LiDAR data — holding position")
            pending_action = None

        last_action = action

        elapsed = time.time() - t0
        spare   = interval - elapsed
        if spare > 0:
            time.sleep(spare)


if __name__ == '__main__':
    main()