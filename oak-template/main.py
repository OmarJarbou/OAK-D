# main.py
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


# ============================================================
# Safe direction engine (navigation)
# ============================================================
from utils.safe_direction import SafeDirectionEngine, YoloObstacle



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


def send_command(command: str):
    """Send a CMD:<command> message to the Arduino."""
    serial_queue.put(f"CMD:{command}")


# Navigation TTS cooldown (seconds)
NAV_TTS_COOLDOWN = 3.0


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

    # try:
    #     engine = pyttsx3.init()
    #     engine.setProperty("rate", 170)
    #     print("[TTS] Engine initialized")
    # except Exception as e:
    #     print(f"[TTS] init failed: {e}")
    #     while True:
    #         text = tts_queue.get()
    #         if text is None:
    #             break
    #         print(f"[TTS fallback] {text}")
    #     return
    print("[TTS] Engine ready")
    while True:
        text = tts_queue.get()
        if text is None:
            break
        
        # On Windows, pyttsx3 often hangs after the first speech when run in a persistent 
        # background thread due to COM apartment message pump limitations.
        # Spawning a short-lived thread for each utterance forces COM to clean up properly.
        def _say(txt):
            try:
                #engine.say(text)
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.say(txt)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] error: {e}")

        import threading
        t = threading.Thread(target=_say, args=(text,), daemon=True)
        t.start()
        t.join() # Wait so we don't overlap speech


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

            # Navigation direction engine
            nav_engine = SafeDirectionEngine()
            last_nav_command = "STOP"
            last_nav_tts_time = 0.0

            print("Pipeline created.")

            pipeline.start()

            if visualizer is not None:
                visualizer.registerPipeline(pipeline)

            last_zone = "GREEN"

            last_det = None

            while pipeline.isRunning():
                try:
                    if visualizer is not None:
                        key = visualizer.waitKey(1)
                        if key == ord("q"):
                            print("Got q key! Exiting.")
                            pipeline.stop()
                            break

                    # Always grab the latest detection if ones are queuing up
                    while detection_queue.has():
                        last_det = detection_queue.get()

                    # Only process a cycle when a fresh depth frame arrives
                    if depth_queue.has():
                        depth_msg = depth_queue.get()
                    else:
                        time.sleep(0.005)
                        continue

                    if last_det is None:
                        continue  # Wait until first YOLO frame arrives
                    
                    det = last_det
                    depth_frame = depth_msg.getFrame()
                    if depth_frame is None or depth_frame.size == 0:
                        continue

                    depth_h, depth_w = depth_frame.shape

                    yolo_obstacles = []

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

                        cx = (d.xmin + d.xmax) / 2.0

                        if mask is not None:
                            roi = depth_frame[mask > 0]
                        else:
                            x1 = int(d.xmin * depth_w)
                            y1 = int(d.ymin * depth_h)
                            x2 = int(d.xmax * depth_w)
                            y2 = int(d.ymax * depth_h)

                            cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2
                            hw, hh = (x2 - x1) // 4, (y2 - y1) // 4

                            x1 = max(0, cx_px - hw)
                            x2 = min(depth_w - 1, cx_px + hw)
                            y1 = max(0, cy_px - hh)
                            y2 = min(depth_h - 1, cy_px + hh)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = depth_frame[y1:y2, x1:x2].flatten()

                        if roi.size < 10:
                            continue

                        valid = roi[(roi > 100) & (roi < 10000)]
                        if valid.size < 5:
                            continue

                        distance_m = float(np.median(valid)) / 1000.0
                        yolo_obstacles.append(YoloObstacle(label=label_name, distance_mm=distance_m * 1000.0, cx_ratio=cx))

                    # ── Unified Decision Processing ──────────
                    raw_nav, nav_metrics, nav_vis = nav_engine.analyze_scene(depth_frame, yolo_obstacles)
                    stable_nav = nav_engine.smooth_decision(raw_nav)

                    # Draw stable command
                    stable_nav_color = {
                        "FORWARD": (0, 255, 0),
                        "LEFT": (0, 165, 255),
                        "RIGHT": (255, 100, 0),
                        "STOP": (0, 0, 255),
                    }.get(stable_nav, (255, 255, 255))
                    cv2.putText(
                        nav_vis,
                        f"CMD: {stable_nav}",
                        (20, nav_vis.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        stable_nav_color,
                        3,
                        cv2.LINE_AA,
                    )
                    

                    if stable_nav != last_nav_command:
                        now_t = time.time()
                        
                        send_command(stable_nav)
                        print(f"[CMD] {last_nav_command} -> {stable_nav}")
                        
                        # TTS with cooldown
                        if now_t - last_nav_tts_time >= NAV_TTS_COOLDOWN:
                            tts_map = {
                                "FORWARD": "Go forward",
                                "LEFT": "Turn left",
                                "RIGHT": "Turn right",
                                "STOP": "Stop",
                            }
                            speak(tts_map.get(stable_nav, stable_nav))
                            last_nav_tts_time = now_t
                            
                        last_nav_command = stable_nav

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
