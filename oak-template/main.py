#!/usr/bin/env python3
import time
import cv2
import depthai as dai
import threading
from utils.config import WalkerConfig
from utils.lidar_analyzer import LidarAnalyzer
from utils.arduino_serial import ArduinoSerial
from lidar_gap_follower import LidarGapFollower
from oakd_emergency import OakdEmergency


def main():
    cfg = WalkerConfig.from_env()
    print(f"Width required: {cfg.REQUIRED_CLEAR_WIDTH_M:.2f}m")

    # Arduino
    arduino = ArduinoSerial(cfg.ARDUINO_PORT, cfg.ARDUINO_BAUD)
    arduino.start()
    time.sleep(2)

    # LiDAR
    lidar = LidarAnalyzer(
        port=cfg.LIDAR_PORT,
        mock=cfg.LIDAR_PORT.upper() == "MOCK",
        backend=cfg.LIDAR_BACKEND,
        c1_baud=cfg.LIDAR_BAUD,
        legacy_baud=cfg.LIDAR_LEGACY_BAUD,
        safety_mm=cfg.LIDAR_SAFETY_MM,
        side_escape_mm=cfg.LIDAR_SIDE_ESCAPE_MM,
        scan_timeout_s=cfg.LIDAR_SCAN_TIMEOUT_S,
        front_arc_deg=cfg.LIDAR_FRONT_ARC_DEG,
        side_arc_start_deg=cfg.LIDAR_SIDE_ARC_START_DEG,
        side_arc_end_deg=cfg.LIDAR_SIDE_ARC_END_DEG,
    )
    lidar.start()
    gap_follower = LidarGapFollower(cfg, lidar)

    # OAK-D for emergency
    print("Initializing OAK-D...")
    pipeline = dai.Pipeline()
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setConfidenceThreshold(200)
    stereo.setOutputSize(640, 400)
    stereo.setLeftRightCheck(True)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    device = dai.Device(pipeline)
    depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    emergency_checker = OakdEmergency(cfg)

    print("System ready. Waiting for authorization...")
    last_sent = ""
    last_stop_time = 0
    STOP_HOLD = 1.0

    while True:
        # Check Arduino state
        state = arduino.state.snapshot()
        if not state["authorized"] or not state["ready"]:
            time.sleep(0.1)
            continue

        # Read depth frame (non-blocking)
        depth_frame = None
        in_depth = depth_queue.tryGet()
        if in_depth:
            depth_frame = in_depth.getFrame()

        # Emergency by OAK-D
        emergency = emergency_checker.check(depth_frame)

        if emergency:
            if time.time() - last_stop_time > STOP_HOLD:
                arduino.send_command("STOP")
                last_stop_time = time.time()
                print("[STOP] Emergency (OAK-D)")
            continue

        # Normal navigation: LiDAR gap follower
        cmd = gap_follower.get_command()
        if cmd is None:
            arduino.send_command("STOP")
            print("[No LiDAR] STOP")
            continue

        # Handle locked states to avoid oscillation
        if cmd == "RIGHT" and state.get("locked_right"):
            cmd = "LEFT"
            print("[LOCKED] Right is locked, forcing LEFT")
        elif cmd == "LEFT" and state.get("locked_left"):
            cmd = "RIGHT"
            print("[LOCKED] Left is locked, forcing RIGHT")

        # Convert to Arduino command
        if cmd == "CENTER":
            arduino_cmd = "GO:CENTER"
        elif cmd == "LEFT":
            arduino_cmd = "GO:LEFT"
        elif cmd == "RIGHT":
            arduino_cmd = "GO:RIGHT"
        else:
            arduino_cmd = "STOP"

        # Avoid spamming same command (send every 0.2s at most)
        if arduino_cmd != last_sent:
            arduino.send_command(arduino_cmd)
            last_sent = arduino_cmd
            print(f"[CMD] {arduino_cmd} (gap={gap_follower.last_command})")
        else:
            # Still send every 2 seconds to keep alive
            if time.time() - gap_follower.last_time > 2.0:
                arduino.send_command(arduino_cmd)
                last_sent = arduino_cmd

        time.sleep(0.05)  # ~20 Hz


if __name__ == "__main__":
    main()
