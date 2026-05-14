#!/usr/bin/env python3
"""
Quick serial diagnostic - run on Raspberry Pi to check if Arduino data arrives.
Usage: python3 serial_test.py
"""
import serial
import time
import sys

PORT = "/dev/serial0"
BAUD = 9600

print(f"Opening {PORT} @ {BAUD}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(0.5)
    print(f"Connected! is_open={ser.is_open}")
    print("Waiting for data from Arduino (up to 30 seconds)...")
    print("Now tap your RFID card or send 'B' via USB Serial Monitor...")
    print("-" * 50)

    start = time.time()
    buffer = ""
    while time.time() - start < 30:
        raw = ser.read(256)
        if raw:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = repr(raw)
            buffer += text
            print(f"[RAW HEX] {raw.hex()}")
            print(f"[RAW TXT] {repr(text)}")
            # Print clean lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if line:
                    print(f"[LINE] {line}")
        else:
            elapsed = int(time.time() - start)
            print(f"\r  ... waiting {elapsed}s", end="", flush=True)

    print("\n" + "-" * 50)
    print("Done. If no [RAW] lines appeared, check wiring or UART config.")
    ser.close()
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
