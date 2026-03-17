import os
import serial
import threading
import queue
import pyttsx3
from dotenv import load_dotenv
import depthai as dai
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node import ApplyDepthColormap
from utils.snaps_producer import SnapsProducer
from utils.obstacle_alert import ObstacleAlert
import cv2
import numpy as np

load_dotenv()
api_key      = os.getenv("OAK_API_KEY")
arduino_port = os.getenv("ARDUINO_PORT", "COM3")   # e.g. COM3 on Windows, /dev/ttyUSB0 on Pi
arduino_baud = int(os.getenv("ARDUINO_BAUD", 9600))

model = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"

# Labels that actually matter on a walking path — ignore everything else
OBSTACLE_LABELS = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "couch", "dining table", "door", "potted plant",
    "dog", "cat", "backpack", "suitcase"
}

# Danger zones (meters)
ZONE_RED    = 0.5    # immediate stop
ZONE_YELLOW = 1.5    # slow down / warn

def classify_zone(distance_m):
    if distance_m < ZONE_RED:
        return "RED"
    elif distance_m < ZONE_YELLOW:
        return "YELLOW"
    return "GREEN"

# ------------------------------------------------------------------
# Serial writer (runs in its own thread so it never blocks the pipeline)
# ------------------------------------------------------------------
serial_queue = queue.Queue()

def serial_writer(port, baud):
    if port == "MOCK":
        print("[Serial] Running in MOCK mode — no Arduino connected")
        while True:
            msg = serial_queue.get()
            if msg is None:
                break
            print(f"[Serial → Arduino] {msg}")  # just print instead of sending
        return

    # real serial path (unchanged)
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

def send_to_arduino(zone, distance_m):
    # Protocol: "RED:0.42\n" / "YELLOW:1.20\n" / "GREEN\n"
    if zone == "GREEN":
        serial_queue.put("GREEN")
    else:
        serial_queue.put(f"{zone}:{distance_m:.2f}")

# ------------------------------------------------------------------
# TTS voice alert — uses Windows SAPI directly (reliable from threads)
# ------------------------------------------------------------------
tts_queue = queue.Queue(maxsize=2)

def tts_worker():
    import win32com.client
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 1   # slightly faster than default
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            speaker.Speak(text)
        except Exception as e:
            print(f"[TTS] error: {e}")

def speak(text):
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    try:
        tts_queue.put_nowait(text)
    except queue.Full:
        pass

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
visualizer = dai.RemoteConnection(httpPort=8082)

# Start background threads
serial_thread = threading.Thread(target=serial_writer, args=(arduino_port, arduino_baud), daemon=True)
serial_thread.start()
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

try:
    with dai.Device() as device, dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        model_desc          = dai.NNModelDescription(model)
        model_desc.platform = device.getPlatformAsString()
        nn_archive          = dai.NNArchive(dai.getModelFromZoo(model_desc, apiKey=api_key))
        label_map           = nn_archive.getConfigV1().model.heads[0].metadata.classes

        color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left_cam  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        color_preview = color_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)
        left_out  = left_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)
        right_out = right_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)

        stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)  # extended disparity requires this
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)   # better close-range depth (critical for walker)
        stereo.setLeftRightCheck(True)
        stereo.setPostProcessingHardwareResources(2, 2)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(1280, 720)

        depth_colormap = pipeline.create(ApplyDepthColormap).build(stereo.disparity)
        depth_colormap.setColormap(cv2.COLORMAP_JET)

        nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(color_cam, nn_archive)

        visualizer.addTopic("Color", color_preview, "images")
        visualizer.addTopic("Depth", depth_colormap.out, "images")
        visualizer.addTopic("YOLO",  nn_with_parser.out, "images")

        snaps_producer = pipeline.create(SnapsProducer).build(
            nn_with_parser.passthrough, nn_with_parser.out,
            label_map=label_map
        )

        detection_queue = nn_with_parser.out.createOutputQueue()
        depth_queue     = stereo.depth.createOutputQueue()

        # Throttle: only re-alert same zone every N seconds
        alert = ObstacleAlert(red_cooldown=1.0, yellow_cooldown=3.0)

        print("Pipeline created.")
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        last_zone = "GREEN"

        while pipeline.isRunning():
            try:
                key = visualizer.waitKey(1)
                if key == ord('q'):
                    print("Got q key! Exiting.")
                    pipeline.stop()
                    break

                det       = detection_queue.get() if detection_queue.has() else None
                depth_msg = depth_queue.get()     if depth_queue.has()     else None

                if det is None or depth_msg is None:
                    continue

                depth_frame = depth_msg.getFrame()
                if depth_frame is None or depth_frame.size == 0:
                    continue

                depth_h, depth_w = depth_frame.shape

                # Find the closest relevant obstacle this frame
                closest_distance = float("inf")
                closest_label    = None

                for d in det.detections:
                    label_name = label_map[d.label]
                    if label_name not in OBSTACLE_LABELS:
                        continue
                    if d.confidence < 0.65:
                        continue

                    # Use segmentation mask if available
                    mask = None
                    if hasattr(d, "mask") and d.mask is not None:
                        try:
                            raw_mask = np.array(d.mask, dtype=np.uint8)
                            mask = cv2.resize(raw_mask, (depth_w, depth_h),
                                              interpolation=cv2.INTER_NEAREST)
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
                        hw, hh = (x2 - x1) // 4,  (y2 - y1) // 4
                        x1 = max(0, cx - hw);  x2 = min(depth_w - 1, cx + hw)
                        y1 = max(0, cy - hh);  y2 = min(depth_h - 1, cy + hh)
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
                        closest_label    = label_name

                # Proximity fallback: if center of frame is very close,
                # treat as RED even if YOLO missed the detection
                center_x1 = depth_w // 4
                center_x2 = 3 * depth_w // 4
                center_y1 = depth_h // 4
                center_y2 = 3 * depth_h // 4
                center_roi = depth_frame[center_y1:center_y2, center_x1:center_x2]
                center_valid = center_roi[(center_roi > 100) & (center_roi < 10000)]
                if center_valid.size > 50:
                    center_dist = float(np.median(center_valid)) / 1000.0
                    # If something is very close in center, override YOLO result
                    if center_dist < 0.4 and (closest_label is None or center_dist < closest_distance):
                        closest_distance = center_dist
                        closest_label    = "obstacle"

                # Determine zone for closest obstacle
                if closest_label is not None:
                    zone = classify_zone(closest_distance)
                else:
                    zone = "GREEN"

                # Send to Arduino + speak — only when zone changes or cooldown elapsed
                should_serial = alert.should_alert(zone, closest_distance)
                stable_zone = alert.committed_zone

                if stable_zone != last_zone:
                    zone = stable_zone  # use stable zone for all output below
                    if zone == "RED":
                        speak(f"Stop. {closest_distance:.1f} meters")
                    elif zone == "YELLOW":
                        speak(f"Caution. {closest_distance:.1f} meters")
                    elif zone == "GREEN":
                        speak("Clear")
                    print(f"[ZONE] {last_zone} → {zone}"
                          + (f" | {closest_label} at {closest_distance:.2f}m" if closest_label else ""))
                    last_zone = stable_zone

                # Serial: fires on cooldown — keeps Arduino regularly updated
                if should_serial:
                    send_to_arduino(zone, closest_distance if zone != "GREEN" else 0)

            except Exception as e:
                print(f"Error in main loop: {e}")
                break

finally:
    serial_queue.put(None)   # stop serial thread
    tts_queue.put(None)      # stop TTS thread