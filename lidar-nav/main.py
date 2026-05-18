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
ANGLE_DEAD_BAND  =  5    # don't resend if smoothed angle changed < this
STOP_HOLD_SEC    =  0.8  # hold CMD:STOP for this long before re-evaluating

# ── EMA angle smoothing ────────────────────────────────────────────────
# Prevents the noisy LiDAR scan from causing rapid oscillation in the
# steering output (e.g. -90 / 0 / -90 / 0 every cycle).
# Formula: smooth = ALPHA * raw + (1 - ALPHA) * smooth
#   ALPHA = 0.0  → angle never changes   (too sluggish)
#   ALPHA = 1.0  → no smoothing          (raw, oscillates)
#   ALPHA = 0.35 → good balance for a walker at walking speed
EMA_ALPHA        = 0.35


# ── Serial helpers ────────────────────────────────────────────────────

def send(ser: serial.Serial, cmd: str):
    line = cmd.strip() + '\n'
    ser.write(line.encode())
    ser.flush()
    print(f"[TX] {cmd}")


def drain_rx(ser: serial.Serial):
    """Read and print any incoming messages from Arduino (non-blocking)."""
    while ser.in_waiting:
        try:
            line = ser.readline().decode(errors='replace').strip()
            if line:
                print(f"[RX] {line}")
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=== Smart Walker — LiDAR Navigator ===")

    # ── Open serial port ─────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
        time.sleep(2)           # wait for Arduino reset after DTR toggle
        print(f"[SERIAL] {SERIAL_PORT} @ {SERIAL_BAUD} baud  OK")
    except serial.SerialException as exc:
        print(f"[SERIAL] Cannot open {SERIAL_PORT}: {exc}")
        sys.exit(1)

    # ── Start LiDAR ──────────────────────────────────────────────────
    scanner = LidarScanner(port=LIDAR_PORT, baudrate=LIDAR_BAUD)
    try:
        scanner.start()
        time.sleep(1)           # allow first scan to arrive
        print(f"[LIDAR]  {LIDAR_PORT} @ {LIDAR_BAUD} baud  OK")
    except Exception as exc:
        print(f"[LIDAR]  Failed to start: {exc}")
        ser.close()
        sys.exit(1)

    # ── Graceful shutdown ────────────────────────────────────────────
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

    # ── Engage motor once ────────────────────────────────────────────
    send(ser, "CMD:ASSIST")
    time.sleep(0.2)

    # ── State variables ───────────────────────────────────────────────
    last_action      = None
    last_angle_sent  = None
    stop_until       = 0.0
    interval         = 1.0 / LOOP_HZ
    ema_angle        = 0.0      # running EMA of the steering angle

    print("[MAIN] Navigation running. Press Ctrl+C to stop.\n")

    while True:
        t0   = time.time()
        scan = scanner.get_scan()
        drain_rx(ser)

        action, raw_angle = decide(scan)

        now = time.time()

        # ── STOP holdoff: once triggered, keep it for STOP_HOLD_SEC ──
        if action == 'STOP':
            stop_until = now + STOP_HOLD_SEC
        if now < stop_until:
            action, raw_angle = 'STOP', 0

        # ── EMA smoothing ─────────────────────────────────────────────
        if action == 'STOP':
            ema_angle = 0.0         # reset so recovery starts from centre
            angle     = 0
        elif action in ('STEER', 'CENTER'):
            ema_angle = EMA_ALPHA * raw_angle + (1.0 - EMA_ALPHA) * ema_angle
            angle     = int(ema_angle)
        else:
            angle = 0               # NODATA — don't update EMA

        # ── Decide what to transmit ───────────────────────────────────
        cmd_str = None

        if action == 'STOP':
            if last_action != 'STOP':
                cmd_str = "CMD:STOP"

        elif action in ('STEER', 'CENTER'):
            # Send only when smoothed angle changed by more than dead-band
            if (last_angle_sent is None or
                    abs(angle - last_angle_sent) > ANGLE_DEAD_BAND):
                cmd_str = f"CMD:ANGLE:{angle}"
                last_angle_sent = angle

        elif action == 'NODATA':
            if last_action != 'NODATA':
                print("[NAV] WARNING: no LiDAR data — holding position")

        if cmd_str:
            send(ser, cmd_str)

        last_action = action

        # ── Sleep to keep LOOP_HZ ────────────────────────────────────
        elapsed = time.time() - t0
        spare   = interval - elapsed
        if spare > 0:
            time.sleep(spare)


if __name__ == '__main__':
    main()
