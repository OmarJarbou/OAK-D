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
READY_TIMEOUT_SEC = 2.0  # if no ACK arrives within this time, force arduino_ready = True


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
    lines = []
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='replace').strip()
            if line:
                print(f"[RX] {line}")
                lines.append(line)
        except Exception:
            pass
    return ' '.join(lines)   # ← كل الأسطر في string واحد


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

    # ── Wait for RFID authorization ───────────────────────────────────
    print("[AUTH] Waiting for RFID authorization (scan your card)…")
    while True:
        rx = drain_rx(ser)
        if 'AUTHORIZED' in rx:
            print("[AUTH] Authorized ✓")
            break
        time.sleep(0.05)

    # ── Wait for motor IDLE (STATUS:FREE) ────────────────────────────
    print("[AUTH] Waiting for motor READY…")
    while True:
        rx = drain_rx(ser)
        if 'FREE' in rx or 'READY' in rx:
            print("[AUTH] Motor idle ✓")
            break
        time.sleep(0.05)

    # ── Engage motor and wait for STATUS:ASSIST ───────────────────────
    print("[AUTH] Entering ASSIST mode…")
    send(ser, "CMD:ASSIST")
    t_assist = time.time()
    while True:
        rx = drain_rx(ser)
        if 'ASSIST' in rx:
            print("[AUTH] ASSIST mode active ✓\n")
            break
        if time.time() - t_assist > 3.0:
            print("[AUTH] WARNING: no STATUS:ASSIST received — continuing anyway")
            break
        time.sleep(0.05)
    # ── Force re-center after ASSIST ─────────────────────────────────
    # After FREE mode the wheel may have drifted slightly.
    # CMD:ASSIST engages the motor but does NOT re-center.
    # Send CMD:GO:CENTER explicitly so the Arduino runs a pot-based
    # centering move (identical to stopAuthSequence) and confirms with
    # STATUS:REACHED or STATUS:AT_TARGET before we start navigation.
    print("[AUTH] Forcing center position…")
    send(ser, "CMD:GO:CENTER")
    t_center = time.time()
    while True:
        rx = drain_rx(ser)
        if any(k in rx for k in ('REACHED', 'AT_TARGET')):
            print("[AUTH] Centered ✓ — starting navigation\n")
            break
        if time.time() - t_center > 5.0:
            print("[AUTH] WARNING: center timeout — starting navigation anyway\n")
            break
        time.sleep(0.05)

    # ── State variables ───────────────────────────────────────────────

    last_action      = None
    last_angle_sent  = None
    stop_until       = 0.0
    smoothed_angle   = 0.0
    clear_count      = 0
    locked_dir       = None
    lock_time        = 0.0
    interval         = 1.0 / LOOP_HZ

    # ── ACK-gate state ────────────────────────────────────────────────
    arduino_ready  = True   # True = OK to send CMD:ANGLE
    ready_sent_at  = 0.0    # timestamp of last CMD:ANGLE send (for timeout)
    pending_action = None   # latest decision accumulated while gate is closed
    pending_angle  = 0      # latest angle accumulated while gate is closed

    print("[MAIN] Navigation running. Press Ctrl+C to stop.\n")

    while True:
        t0   = time.time()
        scan = scanner.get_scan()

        # ── Always drain RX — needed to receive ACK ───────────────────
        rx  = drain_rx(ser)
        now = time.time()

        # ── Parse Arduino lock feedback ───────────────────────────────
        if 'LOCKED_LEFT' in rx:
            locked_dir = 'left';  lock_time = now
        elif 'LOCKED_RIGHT' in rx:
            locked_dir = 'right'; lock_time = now
        elif any(k in rx for k in ('REACHED', 'AT_TARGET', 'UNLOCKED', 'STOPPED', 'FREE')):
            locked_dir = None

        # ── ACK: mark Arduino ready ───────────────────────────────────
        if any(k in rx for k in ('REACHED', 'AT_TARGET', 'READY', 'INTERRUPTED')):
            if not arduino_ready:
                print("[ACK] Arduino ready")
            arduino_ready = True

        # ── Timeout: force ready if ACK never arrived ─────────────────
        if not arduino_ready and (now - ready_sent_at) > READY_TIMEOUT_SEC:
            print("[ACK] Timeout — assuming ready")
            arduino_ready = True

        # ── Auto-unlock motor after lock delay ────────────────────────
        if locked_dir and (now - lock_time) > LOCK_UNLOCK_DELAY:
            send(ser, "CMD:UNLOCK")
            locked_dir = None

        # ── Run navigation decision every tick ────────────────────────
        action, raw_angle = decide(scan)

        # ── STOP holdoff ──────────────────────────────────────────────
        if action == 'STOP':
            stop_until = now + STOP_HOLD_SEC
        if now < stop_until:
            action, raw_angle = 'STOP', 0

        # ── EMA smoothing + clear-path hysteresis ─────────────────────
        if action == 'STEER':
            clear_count    = 0
            smoothed_angle = SMOOTH_ALPHA * raw_angle + (1 - SMOOTH_ALPHA) * smoothed_angle
        elif action == 'CENTER':
            clear_count += 1
            if clear_count >= CLEAR_FRAMES_CTR:
                smoothed_angle = (1 - SMOOTH_ALPHA) * smoothed_angle  # decay toward 0

        angle = int(round(smoothed_angle)) if action in ('STEER', 'CENTER') else raw_angle

        # ── Respect motor lock direction ──────────────────────────────
        if   locked_dir == 'right' and angle > 0: angle = 0
        elif locked_dir == 'left'  and angle < 0: angle = 0

        # ── STOP bypass: always send immediately, skip gate ───────────
        if action == 'STOP':
            if last_action != 'STOP':
                send(ser, "CMD:STOP")
            # STOP does not require an ACK — keep gate open
            arduino_ready  = True
            pending_action = None
            last_action    = 'STOP'
            elapsed = time.time() - t0
            spare   = interval - elapsed
            if spare > 0:
                time.sleep(spare)
            continue

        # ── Accumulate latest decision while gate is closed ───────────
        pending_action = action
        pending_angle  = angle

        # ── On ACK: evaluate pending and send if worthwhile ───────────
        if arduino_ready and pending_action is not None:
            if pending_action in ('STEER', 'CENTER'):
                if (last_angle_sent is None or
                        abs(pending_angle - last_angle_sent) > ANGLE_DEAD_BAND):
                    send(ser, f"CMD:ANGLE:{pending_angle}")
                    last_angle_sent = pending_angle
                    arduino_ready   = False   # wait for next ACK
                    ready_sent_at   = now
            elif pending_action == 'NODATA':
                if last_action != 'NODATA':
                    print("[NAV] WARNING: no LiDAR data — holding position")
            pending_action = None

        last_action = action

        # ── Sleep to maintain LOOP_HZ ────────────────────────────────
        elapsed = time.time() - t0
        spare   = interval - elapsed
        if spare > 0:
            time.sleep(spare)


if __name__ == '__main__':
    main()
