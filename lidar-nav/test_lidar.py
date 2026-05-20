from lidar_scanner import LidarScanner
import time

scanner = LidarScanner(port='/dev/ttyUSB0', baudrate=460800)
scanner.start()
print("Warming up...")
time.sleep(2)

print("\nIs there a wall in front of you? -- Finding FRONT heading...\n")

for i in range(10):
    scan = scanner.get_scan()
    if not scan:
        print("No scan yet...")
        time.sleep(0.5)
        continue

    min_angle = min(scan, key=scan.get)
    min_dist  = scan[min_angle]

    max_angle = max(scan, key=scan.get)
    max_dist  = scan[max_angle]

    sorted_pts = sorted(scan.items(), key=lambda x: x[1])[:5]
    avg_close  = sum(a for a,d in sorted_pts) / len(sorted_pts)
    avg_dist   = sum(d for a,d in sorted_pts) / len(sorted_pts)

    print(f"[{i+1}] Closest: {min_angle:>3}deg @ {min_dist}mm  |  Top5 avg angle: {avg_close:.1f}deg  avg dist: {avg_dist:.0f}mm")
    time.sleep(0.8)

scanner.stop()
print("\n-> The repeated angle in Closest or Top5 is FRONT_HEADING")
