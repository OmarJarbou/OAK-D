#!/usr/bin/env python3
"""
Smart Walker — LiDAR-Only Navigator
====================================
Raspberry Pi  ←Serial1→  Arduino Mega  (115200 baud, pins 18/19 on Mega)

Flow
----
  LidarScanner thread  →  latest full-rotation scan dict
  Main loop (LOOP_HZ)  →  decide()  →  CMD:ANGLE:<n> / CMD:STOP / CMD:FREE

Commands sent to Arduino
------------------------
  CMD:ASSIST         engage stepper motor (sent once at startup)
  CMD:ANGLE:<n>      steer to n  (-100=full left, 0=straight, +100=full right)
  CMD:STOP           emergency stop (brake + vibration on Arduino side)
  CMD:FREE           release motor (sent on clean exit)

Configuration checklist (edit the CAPS constants below)
---------------------------------------------------------
  SERIAL_PORT   — UART connected to Arduino Serial1 (TX1/RX1)
  LIDAR_PORT    — USB port of the RPLIDAR C1
  FRONT_HEADING — angle in navigator.py matching "forward" direction
"""

import serial
import time
import signal
import sys

from lidar_scanner import LidarScanner
from navigator     import decide

# ── Serial (Pi → Arduino) ─────────────────────────────────────────────
SERIAL_PORT  = '/dev/serial0'    # Pi UART → Arduino Serial1 (TX1/RX1)
                                  # use '/dev/ttyUSB0' if connected via USB cable
SERIAL_BAUD  = 115200

# ── RPLIDAR C1 ────────────────────────────────────────────────────────
LIDAR_PORT   = '/dev/ttyUSB0'
LIDAR_BAUD   = 460800

# ── Loop tuning ───────────────────────────────────────────────────────
LOOP_HZ          = 10    # steering decisions per second
ANGLE_DEAD_BAND  =  10    # don't resend if angle changed by less than this
STOP_HOLD_SEC    =  0.8  # hold CMD:STOP for this long before re-evaluating
SMOOTH_ALPHA     =  0.35 # EMA weight for new angle (lower = smoother, 0 = frozen)
CLEAR_FRAMES_CTR =  4    # consecutive CENTER frames before decaying toward 0
LOCK_UNLOCK_DELAY = 1.5  # seconds: auto-send CMD:UNLOCK after a motor lock


# ── Serial helpers ────────────────────────────────────────────────────

def send(ser: serial.Serial, cmd: str):
    ser.reset_output_buffer()   # امسح أي أوامر متراكمة ← أضف هذا
    line = cmd.strip() + '\n'
    ser.write(line.encode())
    ser.flush()
    print(f"[TX] {cmd}")


def drain_rx(ser: serial.Serial) -> str:
    """Read and print any incoming messages from Arduino (non-blocking).
    Returns the last non-empty line received."""
    last = ""
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='replace').strip()
            if line:
                print(f"[RX] {line}")
                last = line
        except Exception:
            pass
    return last


def main():
    print("=== Smart Walker — LiDAR Navigator ===")

    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
        time.sleep(2)
        print(f"[SERIAL] {SERIAL_PORT} @ {SERIAL_BAUD} baud  OK")
    except serial.SerialException as exc:
        print(f"[SERIAL] Cannot open {SERIAL_PORT}: {exc}")
        sys.exit(1)

    scanner = LidarScanner(port=LIDAR_PORT, baudrate=LIDAR_BAUD)
    try:
        scanner.start()
        time.sleep(1)
        print(f"[LIDAR]  {LIDAR_PORT} @ {LIDAR_BAUD} baud  OK")
    except Exception as exc:
        print(f"[LIDAR]  Failed to start: {exc}")
        ser.close()
        sys.exit(1)

    def shutdown(sig=None, frame=None):
        print("\n[MAIN] Shutting down…")
        scanner.stop()
        try:
            send(ser, "CMD:FREE")
            ser.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    send(ser, "CMD:ASSIST")
    time.sleep(0.2)

    # ── State variables ───────────────────────────────────────────────
    last_action      = None
    last_angle_sent  = None
    stop_until       = 0.0
    smoothed_angle   = 0.0
    clear_count      = 0
    locked_dir       = None
    lock_time        = 0.0
    interval         = 1.0 / LOOP_HZ

    # ══ جديد: متغير لمعرفة إذا الأردوينو جاهز ══
    arduino_ready    = True
    ready_timeout    = 0.0          # وقت إرسال آخر أمر
    READY_TIMEOUT_SEC = 2.0        # إذا ما جاء READY خلال 0.6 ثانية، نعتبره جاهز

    print("[MAIN] Navigation running. Press Ctrl+C to stop.\n")

    while True:
        t0   = time.time()
        scan = scanner.get_scan()
        rx   = drain_rx(ser)

        now = time.time()

        # ── Parse Arduino feedback ────────────────────────────────────
        if 'LOCKED_LEFT' in rx:
            locked_dir = 'left';  lock_time = now
        elif 'LOCKED_RIGHT' in rx:
            locked_dir = 'right'; lock_time = now
        elif any(k in rx for k in ('REACHED', 'AT_TARGET', 'UNLOCKED', 'STOPPED', 'FREE')):
            locked_dir = None

        # ══ جديد: اعتبر الأردوينو جاهز إذا أرسل READY أو AT_TARGET أو REACHED ══
        if any(k in rx for k in ('READY', 'AT_TARGET', 'REACHED', 'INTERRUPTED')):
            arduino_ready = True
            print("[ACK] Arduino ready")

        # ══ جديد: timeout — إذا ما جاء رد خلال READY_TIMEOUT_SEC نعتبره جاهز ══
        if not arduino_ready and (now - ready_timeout) > READY_TIMEOUT_SEC:
            arduino_ready = True
            print("[ACK] Timeout — assuming ready")

        if locked_dir and (now - lock_time) > LOCK_UNLOCK_DELAY:
            send(ser, "CMD:UNLOCK")
            locked_dir = None

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

        cmd_str = None

        if action == 'STOP':
            if last_action != 'STOP':
                cmd_str = "CMD:STOP"

        elif action in ('STEER', 'CENTER'):
            if (last_angle_sent is None or
                    abs(angle - last_angle_sent) > ANGLE_DEAD_BAND):
                cmd_str = f"CMD:ANGLE:{angle}"
                last_angle_sent = angle

        elif action == 'NODATA':
            if last_action != 'NODATA':
                print("[NAV] WARNING: no LiDAR data — holding position")

        # ══ جديد: أرسل فقط إذا الأردوينو جاهز ══
        # STOP هو استثناء — يُرسل دائماً بغض النظر عن الجاهزية
        if cmd_str:
            if action == 'STOP' or arduino_ready:
                send(ser, cmd_str)
                if action != 'STOP':
                    arduino_ready = False    # انتظر READY من الأردوينو
                    ready_timeout = now      # ابدأ عد الـ timeout

        last_action = action

        elapsed = time.time() - t0
        spare   = interval - elapsed
        if spare > 0:
            time.sleep(spare)


if __name__ == '__main__':
    main()
