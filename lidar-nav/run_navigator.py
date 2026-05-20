from lidar_scanner import LidarScanner
from navigator import decide, FRONT_HEADING
import time

print(f"FRONT_HEADING = {FRONT_HEADING}")
scanner = LidarScanner(port='/dev/ttyUSB0', baudrate=460800)
scanner.start()
time.sleep(2)

for i in range(15):
    scan = scanner.get_scan()
    action, angle = decide(scan)
    print(f"[{i+1}] action={action}  angle={angle:+d}")
    time.sleep(0.5)

scanner.stop()
