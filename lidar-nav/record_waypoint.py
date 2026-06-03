#!/usr/bin/env python3
"""
record_waypoints.py
===================

Record GPS waypoints for navigation routes.

Usage:
    python3 record_waypoints.py
"""

from gps_navigator import GpsReader
import time

GPS_PORT = "/dev/ttyAMA2"
GPS_BAUD = 9600


def main():
    print("========================================")
    print("GPS Waypoint Recorder")
    print("========================================\n")

    gps = GpsReader(port=GPS_PORT, baud=GPS_BAUD)
    gps.start()

    print("Waiting for GPS data...")

    gps_ready = False

    for i in range(30):
        lat, lon, _, fix = gps.get_position()

        # ? ????? ???: ?? ????? ??? fix ???
        if lat is not None and lon is not None:
            if fix:
                print(f"GPS FIX OK: ({lat:.6f}, {lon:.6f})")
            else:
                print(f"GPS DATA OK (no fix yet): ({lat:.6f}, {lon:.6f})")

            gps_ready = True
            break

        print(f"[{i+1}/30] Waiting for GPS...")
        time.sleep(1)

    if not gps_ready:
        print("ERROR: No GPS data detected.")
        print("Move to an open area and try again.")
        gps.stop()
        return

    print("\nCommands:")
    print("  Enter  -> Save current waypoint")
    print("  done   -> Finish recording")
    print("  del    -> Delete last waypoint")
    print("  show   -> Show all waypoints")
    print("\nStart walking and save points at turns or key locations.\n")

    points = []

    while True:
        cmd = input(">> ").strip().lower()

        if cmd == "done":
            break

        elif cmd == "del":
            if points:
                removed = points.pop()
                print(f"Deleted: {removed}")
            else:
                print("No points to delete.")

        elif cmd == "show":
            if points:
                print(f"\nSaved waypoints ({len(points)}):")
                for i, p in enumerate(points, start=1):
                    print(f"  [{i}] {p}")
                print()
            else:
                print("No waypoints recorded.\n")

        else:
            lat, lon, heading, fix = gps.get_position()

            if lat is not None and lon is not None:
                point = (round(lat, 6), round(lon, 6))
                points.append(point)

                print(
                    f"Waypoint {len(points)} saved: "
                    f"({lat:.6f}, {lon:.6f}) "
                    f"Heading={heading}"
                    + (" [FIX]" if fix else " [NO FIX]")
                )
            else:
                print("No GPS data. Try again.")

    if not points:
        print("No waypoints were recorded.")
        gps.stop()
        return

    destination = input("\nDestination name: ").strip()

    if not destination:
        destination = "new_destination"

    print("\n" + "=" * 60)
    print("Copy this into DESTINATIONS in gps_navigator.py")
    print("=" * 60)

    print(f'\n"{destination}": [')

    for point in points:
        print(f"    {point},")

    print("],")

    print("\n" + "=" * 60)

    gps.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
