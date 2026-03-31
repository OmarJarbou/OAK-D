import os
import time
import threading
import queue
from collections import defaultdict

import cv2
import numpy as np
import serial
import pyttsx3
from dotenv import load_dotenv

import depthai as dai
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node import ApplyDepthColormap


# ============================================================
# Optional local imports fallback
# ============================================================
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

        def build(
            self,
            rgb: dai.Node.Output,
            detections: dai.Node.Output,
            label_map: list,
            confidence_threshold: float = 0.7,
            labels: list = None,
            time_interval: float = 60.0,
            required_streak: int = 3,
        ):
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

        def process(self, rgb: dai.Buffer, detections):
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


try:
    from utils.obstacle_alert import ObstacleAlert
except Exception:
    ZONE_PRIORITY = {"GREEN": 0, "YELLOW": 1, "RED": 2}

    class ObstacleAlert:
        def __init__(self, red_cooldown: float = 1.0, yellow_cooldown: float = 3.0, debounce_frames: int = 3):
            self.cooldowns = {"RED": red_cooldown, "YELLOW": yellow_cooldown}
            self.last_alert = {"RED": 0.0, "YELLOW": 0.0}
            self.active = {"RED": False, "YELLOW": False}
            self._candidate = "GREEN"
            self._candidate_count = 0
            self._committed = "GREEN"
            self.debounce_frames = debounce_frames

        def should_alert(self, zone: str, distance_m: float) -> bool:
            if zone == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = zone
                self._candidate_count = 1

            escalating = ZONE_PRIORITY[zone] > ZONE_PRIORITY[self._committed]
            if escalating or self._candidate_count >= self.debounce_frames:
                self._committed = zone

            committed = self._committed
            now = time.time()

            for z in ("RED", "YELLOW"):
                if z != committed and self.active[z]:
                    self.active[z] = False
                    self.last_alert[z] = 0.0

            if committed == "GREEN":
                return False

            self.active[committed] = True
            if now - self.last_alert[committed] >= self.cooldowns[committed]:
                self.last_alert[committed] = now
                return True
            return False

        @property
        def committed_zone(self):
            return self._committed


# ============================================================
# Config
# ============================================================
load_dotenv()

api_key = os.getenv("OAK_API_KEY", "")
arduino_port = os.getenv("ARDUINO_PORT", "/dev/ttyUSB0")
arduino_baud = int(os.getenv("ARDUINO_BAUD", 9600))
use_visualizer = os.getenv("USE_VISUALIZER", "0") == "1"
use_tts = os.getenv("USE_TTS", "1") == "1"

model = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"

OBSTACLE_LABELS = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "couch", "dining table", "door", "potted plant",
    "dog", "cat", "backpack", "suitcase"
}

ZONE_RED = 0.5
ZONE_YELLOW = 1.5


def classify_zone(distance_m: float) -> str:
    if distance_m < ZONE_RED:
        return "RED"
    elif distance_m < ZONE_YELLOW:
        return "YELLOW"
    return "GREEN"


# ============================================================
# Serial
# ============================================================
serial_queue = queue.Queue()


def serial_writer(port: str, baud: int):
    if port.upper() == "MOCK":
        print("[Serial] Running in MOCK mode — no Arduino connected")
        while True:
            msg = serial_queue.get()
            if msg is None:
                break
            print(f"[Serial → Arduino] {msg}")
        return

    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"Serial connected: {port} @ {baud}")
        while True:
            msg = serial_queue.get()
            if msg is None:
                break
            try:
                ser.write((msg + "\n").encode())
            except Exception as e:
                print(f"[Serial] write error: {e}")
        ser.close()
    except Exception as e:
        print(f"[Serial] Could not open {port}: {e}")


def send_to_arduino(zone: str, distance_m: float):
    if zone == "GREEN":
        serial_queue.put("GREEN")
    else:
        serial_queue.put(f"{zone}:{distance_m:.2f}")


# ============================================================
# TTS (Linux / Raspberry Pi)
# ============================================================
tts_queue = queue.Queue(maxsize=2)


def tts_worker():
    if not use_tts:
        print("[TTS] Disabled")
        while True:
            text = tts_queue.get()
            if text is None:
                break
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        print("[TTS] Engine initialized")
    except Exception as e:
        print(f"[TTS] init failed: {e}")
        while True:
            text = tts_queue.get()
            if text is None:
                break
            print(f"[TTS fallback] {text}")
        return

    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS] error: {e}")


def speak(text: str):
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    try:
        tts_queue.put_nowait(text)
    except queue.Full:
        pass


# ============================================================
# Main
# ============================================================
def main():
    visualizer = None
    if use_visualizer:
        visualizer = dai.RemoteConnection(httpPort=8082)
        print("Visualizer enabled on port 8082")
    else:
        print("Visualizer disabled for stability")

    serial_thread = threading.Thread(
        target=serial_writer,
        args=(arduino_port, arduino_baud),
        daemon=True
    )
    serial_thread.start()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    try:
        with dai.Device() as device, dai.Pipeline(device) as pipeline:
            print("Creating pipeline...")

            model_desc = dai.NNModelDescription(model)
            model_desc.platform = device.getPlatformAsString()
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc, apiKey=api_key))
            label_map = nn_archive.getConfigV1().model.heads[0].metadata.classes

            # Cameras
            color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

            # Lighter outputs for Raspberry Pi stability
            color_preview = None
            if visualizer is not None:
    	        color_preview = color_cam.requestOutput(
                    size=(416, 416),
                    type=dai.ImgFrame.Type.NV12,
                    fps=15
                )

            left_out = left_cam.requestOutput(
                size=(400, 400),
                type=dai.ImgFrame.Type.NV12,
                fps=15
            )
            right_out = right_cam.requestOutput(
                size=(400, 400),
                type=dai.ImgFrame.Type.NV12,
                fps=15
            )

            stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
            stereo.setRectification(True)
            stereo.setExtendedDisparity(True)   
            stereo.setLeftRightCheck(True)      
            stereo.setPostProcessingHardwareResources(2, 2)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(640, 400)

            depth_colormap = pipeline.create(ApplyDepthColormap).build(stereo.disparity)
            depth_colormap.setColormap(cv2.COLORMAP_JET)

            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(color_cam, nn_archive)

            if visualizer is not None:
                visualizer.addTopic("Color", color_preview, "images")
                visualizer.addTopic("Depth", depth_colormap.out, "images")
                visualizer.addTopic("YOLO", nn_with_parser.out, "images")

            _snaps_producer = pipeline.create(SnapsProducer).build(
                nn_with_parser.passthrough,
                nn_with_parser.out,
                label_map=label_map
            )

            detection_queue = nn_with_parser.out.createOutputQueue(maxSize=4, blocking=False)
            depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

            alert = ObstacleAlert(red_cooldown=1.0, yellow_cooldown=3.0)
            print("Pipeline created.")

            pipeline.start()

            if visualizer is not None:
                visualizer.registerPipeline(pipeline)

            last_zone = "GREEN"

            while pipeline.isRunning():
                try:
                    if visualizer is not None:
                        key = visualizer.waitKey(1)
                        if key == ord("q"):
                            print("Got q key! Exiting.")
                            pipeline.stop()
                            break
                    else:
                        time.sleep(0.005)

                    det = detection_queue.get() if detection_queue.has() else None
                    depth_msg = depth_queue.get() if depth_queue.has() else None

                    if det is None or depth_msg is None:
                        continue

                    depth_frame = depth_msg.getFrame()
                    if depth_frame is None or depth_frame.size == 0:
                        continue

                    depth_h, depth_w = depth_frame.shape

                    closest_distance = float("inf")
                    closest_label = None

                    for d in det.detections:
                        label_name = label_map[d.label]
                        if label_name not in OBSTACLE_LABELS:
                            continue
                        if d.confidence < 0.65:
                            continue

                        mask = None
                        if hasattr(d, "mask") and d.mask is not None:
                            try:
                                raw_mask = np.array(d.mask, dtype=np.uint8)
                                mask = cv2.resize(raw_mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
                            except Exception:
                                mask = None

                        if mask is not None:
                            roi = depth_frame[mask > 0]
                        else:
                            x1 = int(d.xmin * depth_w)
                            y1 = int(d.ymin * depth_h)
                            x2 = int(d.xmax * depth_w)
                            y2 = int(d.ymax * depth_h)

                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            hw, hh = (x2 - x1) // 4, (y2 - y1) // 4

                            x1 = max(0, cx - hw)
                            x2 = min(depth_w - 1, cx + hw)
                            y1 = max(0, cy - hh)
                            y2 = min(depth_h - 1, cy + hh)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = depth_frame[y1:y2, x1:x2].flatten()

                        if roi.size < 10:
                            continue

                        valid = roi[(roi > 100) & (roi < 10000)]
                        if valid.size < 5:
                            continue

                        distance_m = float(np.median(valid)) / 1000.0

                        if distance_m < closest_distance:
                            closest_distance = distance_m
                            closest_label = label_name

                    # Fallback from center depth
                    center_x1 = depth_w // 4
                    center_x2 = 3 * depth_w // 4
                    center_y1 = depth_h // 4
                    center_y2 = 3 * depth_h // 4
                    center_roi = depth_frame[center_y1:center_y2, center_x1:center_x2]
                    center_valid = center_roi[(center_roi > 100) & (center_roi < 10000)]

                    if center_valid.size > 50:
                        center_dist = float(np.median(center_valid)) / 1000.0
                        if center_dist < 0.4 and (closest_label is None or center_dist < closest_distance):
                            closest_distance = center_dist
                            closest_label = "obstacle"

                    if closest_label is not None:
                        zone = classify_zone(closest_distance)
                    else:
                        zone = "GREEN"

                    should_serial = alert.should_alert(zone, closest_distance if closest_label is not None else 0.0)
                    stable_zone = alert.committed_zone

                    if stable_zone != last_zone:
                        zone = stable_zone
                        if zone == "RED":
                            speak(f"Stop. {closest_distance:.1f} meters")
                        elif zone == "YELLOW":
                            speak(f"Caution. {closest_distance:.1f} meters")
                        elif zone == "GREEN":
                            speak("Clear")

                        print(
                            f"[ZONE] {last_zone} -> {zone}"
                            + (f" | {closest_label} at {closest_distance:.2f}m" if closest_label else "")
                        )
                        last_zone = stable_zone

                    if should_serial:
                        send_to_arduino(zone, closest_distance if zone != "GREEN" else 0.0)

                except KeyboardInterrupt:
                    print("Interrupted by user.")
                    pipeline.stop()
                    break
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    break

    finally:
        serial_queue.put(None)
        tts_queue.put(None)


if __name__ == "__main__":
    main()
