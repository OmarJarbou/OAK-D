# main.py
"""
Smart Walker — OAK-D Navigation (simplified).

Camera: 3 zones (LEFT / CENTER / RIGHT) → steering.
LiDAR:  front distance → emergency STOP only.
"""

import os
import time
import threading
import queue
from collections import defaultdict

import cv2
import numpy as np
from dotenv import load_dotenv

import depthai as dai
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node import ApplyDepthColormap

from utils.config import WalkerConfig
from utils.arduino_serial import ArduinoSerial
from utils.command_publisher import CommandPublisher
from utils.lidar_analyzer import LidarAnalyzer
from utils.simple_nav import (
    analyze_simple,
    decide_simple,
    apply_flip_lr,
    SimpleSmoother,
    ZONE_NAMES,
)
from utils.tts_player import TTSPlayer

try:
    from utils.snaps_producer import SnapsProducer
except Exception:
    class SnapsProducer(dai.node.HostNode):
        def __init__(self):
            super().__init__()
            self.em = None
            self.label_map = []
            self.confidence_threshold = 0.7
            self.labels = ["person"]
            self.last_update = 0.0
            self.time_interval = 60.0
            self._detection_streak = defaultdict(int)
            self._required_streak = 3

        def build(self, rgb, detections, label_map, confidence_threshold=0.7,
                  labels=None, time_interval=60.0, required_streak=3):
            self.link_args(rgb, detections)
            self.em = dai.EventsManager()
            self.em.setLogResponse(True)
            self.label_map = label_map
            self.confidence_threshold = confidence_threshold
            self.labels = labels if labels is not None else ["person"]
            self.last_update = 0.0
            self.time_interval = time_interval
            self._required_streak = required_streak
            return self

        def process(self, rgb, detections):
            if rgb is None or detections is None:
                return
            now = time.time()
            detected_this_frame = set()
            for det in detections.detections:
                label_str = self.label_map[det.label]
                if det.confidence >= self.confidence_threshold and label_str in self.labels:
                    detected_this_frame.add(label_str)
            for label in self.labels:
                if label in detected_this_frame:
                    self._detection_streak[label] += 1
                else:
                    self._detection_streak[label] = 0
            if now < self.last_update + self.time_interval:
                return
            try:
                for det in detections.detections:
                    label_str = self.label_map[det.label]
                    if (
                        det.confidence >= self.confidence_threshold
                        and label_str in self.labels
                        and self._detection_streak[label_str] >= self._required_streak
                    ):
                        self.last_update = now
                        det_xyxy = [det.xmin, det.ymin, det.xmax, det.ymax]
                        extra_data = {
                            "model": "luxonis/yolov8-instance-segmentation-nano:coco-512x288",
                            "detection_xyxy": str(det_xyxy),
                            "detection_label": str(det.label),
                            "detection_label_str": label_str,
                            "detection_confidence": str(det.confidence),
                            "detection_streak": str(self._detection_streak[label_str]),
                        }
                        self.em.sendSnap("rgb", "rgb", rgb, None, ["demo"], extra_data)
                        print(f"Event sent: {extra_data}")
                        break
            except Exception as e:
                print(f"[SnapsProducer] process error (skipping frame): {e}")

        def __del__(self):
            if self.em is not None:
                self.em.waitForPendingUploads()


load_dotenv()

cfg = WalkerConfig.from_env()
api_key = os.getenv("OAK_API_KEY", "")
use_visualizer = os.getenv("USE_VISUALIZER", "0") == "1"
model = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"

NAV_TTS_COOLDOWN = 3.0
tts_queue = queue.Queue(maxsize=2)
_tts_generation = 0
last_tts_time = 0.0
last_text = None
_tts_player: TTSPlayer | None = None

TTS_MAP = {
    "GO:CENTER": "Go forward",
    "GO:LEFT": "Turn left",
    "GO:RIGHT": "Turn right",
    "STOP": "Stop",
    "FREE": "Free mode",
}


def tts_worker():
    global last_tts_time, last_text, _tts_generation

    if not cfg.USE_TTS:
        print("[TTS] Disabled")
        return

    print("[TTS] Engine ready")
    while True:
        item = tts_queue.get()
        if item is None:
            break
        text, gen = item
        now = time.time()
        if text == last_text and (now - last_tts_time < NAV_TTS_COOLDOWN):
            continue
        if gen != _tts_generation:
            continue
        last_text = text
        last_tts_time = now
        try:
            print(f"[TTS] saying: {text}")
            if _tts_player is not None:
                _tts_player.play_blocking(text)
            else:
                os.system(f'espeak-ng -v en+f3 -s 135 "{text}" >/dev/null 2>&1')
        except Exception as e:
            print(f"[TTS] error: {e}")


def speak(text: str):
    global _tts_generation
    _tts_generation += 1
    gen = _tts_generation
    if _tts_player is not None:
        _tts_player.stop()
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    try:
        tts_queue.put_nowait((text, gen))
    except queue.Full:
        print("[TTS] queue full, skipped")


def build_debug_frame(depth_frame, analysis, result, arduino_state, last_sent):
    clipped = np.clip(depth_frame, cfg.MIN_DEPTH_MM, cfg.MAX_DEPTH_MM)
    norm = (
        (clipped - cfg.MIN_DEPTH_MM)
        / max(cfg.MAX_DEPTH_MM - cfg.MIN_DEPTH_MM, 1)
        * 255.0
    ).astype(np.uint8)
    norm = 255 - norm
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    h, w = depth_frame.shape[:2]
    x1, y1, x2, y2 = analysis.roi_box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

    roi_w = x2 - x1
    zone_w = roi_w // 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    cmd_zone = {
        "GO:LEFT": "LEFT", "GO:CENTER": "CENTER", "GO:RIGHT": "RIGHT",
    }.get(result.stable_command, "")

    for i, name in enumerate(ZONE_NAMES):
        zx1 = x1 + i * zone_w
        zx2 = x2 if i == 2 else x1 + (i + 1) * zone_w
        if i > 0:
            cv2.line(vis, (zx1, y1), (zx1, y2), (180, 180, 180), 1)
        p20 = analysis.metrics.get(name, 0)
        if "STOP" in result.stable_command:
            color = (0, 0, 255)
        elif name == cmd_zone:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        cv2.putText(vis, f"{name} {int(p20)}mm", (zx1 + 4, y1 + 20),
                    font, 0.45, color, 1, cv2.LINE_AA)

    panel_y = h - 8
    cv2.putText(vis, f"RAW: {result.raw_command}", (10, panel_y - 50),
                font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"CMD: {result.stable_command}", (10, panel_y - 28),
                font, 0.6, (0, 255, 0) if "GO" in result.stable_command else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Sent: {last_sent or '-'}", (10, panel_y - 8),
                font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    auth_txt = "AUTH" if arduino_state.get("authorized") else "LOCKED"
    cv2.putText(vis, auth_txt, (w - 90, panel_y - 28),
                font, 0.5, (0, 255, 0) if arduino_state.get("authorized") else (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def main():
    print("=" * 60)
    print("  Smart Walker - Simplified Navigation v4.0")
    print("=" * 60)
    print(f"  Arduino port : {cfg.ARDUINO_PORT}")
    print(f"  Zones        : {cfg.NUM_ZONES} ({', '.join(cfg.ZONE_NAMES)})")
    print(f"  STOP dist    : {cfg.STOP_DISTANCE_MM:.0f}mm")
    print(f"  LiDAR        : front STOP only")
    print(f"  Debug        : {'ON' if cfg.DEBUG_DISPLAY else 'OFF'}")
    print(f"  TTS          : {'ON' if cfg.USE_TTS else 'OFF'}")
    print("=" * 60)

    global _tts_player
    _tts_player = TTSPlayer(sounds_dir="/home/lama/OAK-D/sounds")
    if cfg.USE_TTS:
        _tts_player.pregenerate()

    arduino = ArduinoSerial(port=cfg.ARDUINO_PORT, baud=cfg.ARDUINO_BAUD)
    publisher = CommandPublisher(cfg, arduino)
    smoother = SimpleSmoother(
        hysteresis=cfg.CMD_HYSTERESIS_FRAMES,
        cooldown_ms=cfg.CMD_COOLDOWN_MS,
    )

    lidar_mock = cfg.LIDAR_PORT.upper() == "MOCK"
    _lb = cfg.LIDAR_BACKEND.lower().strip()
    _legacy_baud = cfg.LIDAR_BAUD if _lb == "legacy" else cfg.LIDAR_LEGACY_BAUD
    lidar = LidarAnalyzer(
        port=cfg.LIDAR_PORT,
        mock=lidar_mock,
        backend=cfg.LIDAR_BACKEND,
        c1_baud=cfg.LIDAR_BAUD,
        legacy_baud=_legacy_baud,
        safety_mm=cfg.LIDAR_SAFETY_MM,
        side_escape_mm=cfg.LIDAR_SIDE_ESCAPE_MM,
        scan_timeout_s=cfg.LIDAR_SCAN_TIMEOUT_S,
        front_arc_deg=cfg.LIDAR_FRONT_ARC_DEG,
        side_arc_start_deg=cfg.LIDAR_SIDE_ARC_START_DEG,
        side_arc_end_deg=cfg.LIDAR_SIDE_ARC_END_DEG,
    )

    visualizer = None
    if use_visualizer:
        visualizer = dai.RemoteConnection(httpPort=8082)

    arduino.start()
    lidar.start()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    was_authorized = False
    last_spoken_command = None

    try:
        with dai.Device() as device, dai.Pipeline(device) as pipeline:
            print("[Pipeline] Creating...")

            model_desc = dai.NNModelDescription(model)
            model_desc.platform = device.getPlatformAsString()
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc, apiKey=api_key))
            label_map = nn_archive.getConfigV1().model.heads[0].metadata.classes

            color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            color_cam.initialControl.setAutoExposureLimit(8000)
            left_cam.initialControl.setAutoExposureLimit(8000)
            right_cam.initialControl.setAutoExposureLimit(8000)

            color_preview = None
            if visualizer is not None:
                color_preview = color_cam.requestOutput(
                    size=(416, 416), type=dai.ImgFrame.Type.NV12, fps=30)

            left_out = left_cam.requestOutput(
                size=(400, 400), type=dai.ImgFrame.Type.NV12, fps=30)
            right_out = right_cam.requestOutput(
                size=(400, 400), type=dai.ImgFrame.Type.NV12, fps=30)

            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left_out, right=right_out)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
            stereo.setRectification(True)
            stereo.setExtendedDisparity(False)
            stereo.setSubpixel(True)
            stereo.setLeftRightCheck(True)
            stereo.setPostProcessingHardwareResources(2, 2)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(cfg.DEPTH_WIDTH, cfg.DEPTH_HEIGHT)

            depth_colormap = pipeline.create(ApplyDepthColormap).build(stereo.disparity)
            depth_colormap.setColormap(cv2.COLORMAP_JET)

            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                color_cam, nn_archive)

            if visualizer is not None:
                visualizer.addTopic("Color", color_preview, "images")
                visualizer.addTopic("Depth", depth_colormap.out, "images")
                visualizer.addTopic("YOLO", nn_with_parser.out, "images")

            if cfg.ENABLE_SNAPS:
                pipeline.create(SnapsProducer).build(
                    nn_with_parser.passthrough, nn_with_parser.out, label_map=label_map)

            depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            if visualizer is not None:
                visualizer.registerPipeline(pipeline)

            print("[Pipeline] Running. Waiting for RFID authorization...")
            frame_count = 0

            while pipeline.isRunning():
                try:
                    if visualizer is not None:
                        key = visualizer.waitKey(1)
                        if key == ord("q"):
                            pipeline.stop()
                            break

                    if not depth_queue.has():
                        time.sleep(0.005)
                        continue

                    depth_msg = depth_queue.get()
                    depth_frame = depth_msg.getFrame()
                    if depth_frame is None or depth_frame.size == 0:
                        continue

                    frame_count += 1
                    state = arduino.state.snapshot()

                    if state["authorized"] and not was_authorized:
                        print("[System] AUTHORIZED")
                        speak("System authorized")
                        was_authorized = True
                    elif not state["authorized"] and was_authorized:
                        print("[System] DEAUTHORIZED")
                        smoother.reset()
                        publisher.reset()
                        was_authorized = False

                    analysis = analyze_simple(depth_frame, cfg)
                    if analysis is None:
                        continue

                    lidar_scan = lidar.latest_scan
                    lidar_front = (
                        float(lidar_scan.front_min_mm)
                        if lidar_scan and lidar_scan.front_min_mm > 0
                        else None
                    )

                    raw_cmd, reason = decide_simple(
                        analysis.metrics, lidar_front, cfg
                    )
                    raw_cmd = apply_flip_lr(raw_cmd, cfg.FLIP_LR)

                    result = smoother.update(raw_cmd)
                    result.reason = reason

                    cmd_to_send = result.stable_command
                    if not state.get("authorized") or not state.get("ready"):
                        cmd_to_send = "NONE"

                    sent = publisher.publish(cmd_to_send, state, reason=reason)

                    if sent and cfg.USE_TTS and cmd_to_send not in ("NONE", ""):
                        if cmd_to_send != last_spoken_command:
                            tts_text = TTS_MAP.get(cmd_to_send, cmd_to_send)
                            print(f"[TTS] {tts_text}")
                            speak(tts_text)
                            last_spoken_command = cmd_to_send

                    if cfg.DEBUG_DISPLAY:
                        debug_frame = build_debug_frame(
                            depth_frame, analysis, result, state,
                            publisher.last_command or "-",
                        )
                        cv2.imshow("Smart Walker Navigation", debug_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            pipeline.stop()
                            break

                    if frame_count % 200 == 0:
                        lidar_str = (
                            f"front={lidar_front:.0f}mm"
                            if lidar_front
                            else "lidar=stale"
                        )
                        print(
                            f"[Status] f={frame_count} "
                            f"cmd={result.stable_command} "
                            f"L={analysis.metrics.get('LEFT', 0):.0f} "
                            f"C={analysis.metrics.get('CENTER', 0):.0f} "
                            f"R={analysis.metrics.get('RIGHT', 0):.0f} "
                            f"{lidar_str}"
                        )

                except KeyboardInterrupt:
                    pipeline.stop()
                    break
                except Exception as e:
                    print(f"[ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    break

    finally:
        try:
            arduino.stop()
        except Exception as e:
            print(f"[Shutdown] Arduino: {e}")
        try:
            lidar.stop()
        except Exception as e:
            print(f"[Shutdown] LiDAR: {e}")
        try:
            tts_queue.put(None)
        except Exception as e:
            print(f"[Shutdown] TTS: {e}")
        if cfg.DEBUG_DISPLAY:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()
