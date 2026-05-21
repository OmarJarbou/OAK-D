"""
oak_detector.py — OAK-D Object Detection + Spatial Depth
==========================================================
Uses the EXACT same pipeline pattern as your working main.py:
  - Camera.build(socket)
  - ParsingNeuralNetwork (from model zoo — no blob crash)
  - StereoDepth.build(left, right)
  - Depth sampled manually per bounding box (no SpatialDetectionNetwork)
  - createOutputQueue() for output — no XLinkOut

Model: luxonis/yolov6-nano:coco-416x416  (same zoo, lighter than yolov8)
       Falls back to yolov8n if unavailable.
"""

import depthai as dai
import threading
import time
import numpy as np
import os

# ── Position thresholds ───────────────────────────────────────────────
LEFT_BOUNDARY  = 0.35
RIGHT_BOUNDARY = 0.65

# ── Zoo model — same approach as your working main.py ────────────────
ZOO_MODELS_FALLBACK = [
    "luxonis/yolov8-instance-segmentation-nano:coco-512x288",  # exact model from your main.py
    "luxonis/yolov8n-detection:coco-512x288",
    "luxonis/yolov8n-detection:coco-416x416",
]
API_KEY = os.getenv("OAK_API_KEY", "")


def _classify_position(norm_x: float) -> str:
    if norm_x < LEFT_BOUNDARY:  return "left"
    if norm_x > RIGHT_BOUNDARY: return "right"
    return "center"


def _sample_depth(depth_frame: np.ndarray,
                  xmin: float, ymin: float,
                  xmax: float, ymax: float) -> int:
    """Sample median depth (mm) inside the bounding box centre patch."""
    h, w = depth_frame.shape
    cx = int((xmin + xmax) / 2.0 * w)
    cy = int((ymin + ymax) / 2.0 * h)
    # 10% of bbox as patch size, minimum 4 px
    pw = max(4, int((xmax - xmin) * w * 0.10))
    ph = max(4, int((ymax - ymin) * h * 0.10))
    x0, x1 = max(0, cx - pw), min(w, cx + pw)
    y0, y1 = max(0, cy - ph), min(h, cy + ph)
    patch = depth_frame[y0:y1, x0:x1].flatten()
    valid = patch[(patch > 100) & (patch < 12000)]
    if len(valid) == 0:
        return 0
    return int(np.median(valid))


class OakDetector:
    def __init__(self, confidence_threshold: float = 0.5, api_key: str = API_KEY):
        self._confidence = confidence_threshold
        self._api_key    = api_key
        self._detections = []
        self._lock       = threading.Lock()
        self._running    = False
        self._thread     = None
        self._pipeline   = None
        self._device     = None
        self._label_map  = []

    def start(self):
        self._device   = dai.Device()
        self._pipeline = self._build_pipeline(self._device)
        self._running  = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[OAK] Detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[OAK] Detector stopped")

    def get_detections(self) -> list:
        with self._lock:
            return list(self._detections)

    # ── Pipeline — mirrors your working main.py exactly ───────────────
    def _build_pipeline(self, device) -> dai.Pipeline:
        pipeline = dai.Pipeline(device)

        # ── Load model from zoo (same as main.py) ─────────────────────
        model_desc = None
        nn_archive = None
        for model_name in ZOO_MODELS_FALLBACK:
            try:
                print(f"[OAK] Trying model: {model_name}")
                model_desc = dai.NNModelDescription(model_name)
                model_desc.platform = device.getPlatformAsString()
                nn_archive = dai.NNArchive(
                    dai.getModelFromZoo(model_desc, apiKey=self._api_key))
                print(f"[OAK] Model loaded: {model_name}")
                break
            except Exception as e:
                print(f"[OAK] Model {model_name} failed: {e}")
                continue

        if nn_archive is None:
            raise RuntimeError("[OAK] Could not load any model from zoo. "
                               "Check internet connection or set API_KEY.")

        self._label_map = nn_archive.getConfigV1().model.heads[0].metadata.classes
        print(f"[OAK] Label map: {len(self._label_map)} classes")

        # ── Cameras — exact pattern from your main.py ──────────────────
        cam_rgb   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        cam_rgb.initialControl.setAutoExposureLimit(8000)
        cam_left.initialControl.setAutoExposureLimit(8000)
        cam_right.initialControl.setAutoExposureLimit(8000)

        left_out  = cam_left.requestOutput(
            size=(400, 400), type=dai.ImgFrame.Type.NV12, fps=20)
        right_out = cam_right.requestOutput(
            size=(400, 400), type=dai.ImgFrame.Type.NV12, fps=20)

        # ── Stereo depth — exact pattern from your main.py ─────────────
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_out, right=right_out)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(416, 416)

        # ── ParsingNeuralNetwork — exact pattern from your main.py ──────
        from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
        nn = pipeline.create(ParsingNeuralNetwork).build(cam_rgb, nn_archive)

        # ── Output queues ──────────────────────────────────────────────
        self._det_queue   = nn.out.createOutputQueue(maxSize=4, blocking=False)
        self._depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print("[OAK] Pipeline started")
        return pipeline

    # ── Detection loop ─────────────────────────────────────────────────
    def _loop(self):
        print("[OAK] Detection loop running")
        depth_frame = None

        try:
            while self._running and self._pipeline.isRunning():
                # Always grab latest depth
                if self._depth_queue.has():
                    depth_frame = self._depth_queue.get().getFrame()

                if not self._det_queue.has():
                    time.sleep(0.02)
                    continue

                nn_data    = self._det_queue.get()
                detections = getattr(nn_data, 'detections', [])

                results = []
                for det in detections:
                    label_idx = getattr(det, 'label', -1)
                    if label_idx < 0 or label_idx >= len(self._label_map):
                        continue

                    if depth_frame is not None:
                        z = _sample_depth(depth_frame,
                                          det.xmin, det.ymin,
                                          det.xmax, det.ymax)
                    else:
                        z = 0

                    if z <= 0:
                        continue

                    norm_x = (det.xmin + det.xmax) / 2.0
                    results.append({
                        'label':       self._label_map[label_idx],
                        'position':    _classify_position(norm_x),
                        'distance_mm': z,
                    })

                results.sort(key=lambda d: d['distance_mm'])

                with self._lock:
                    self._detections = results

                if results:
                    t = results[0]
                    print(f"[OAK] {t['label']:<14} | {t['position']:<6} "
                          f"| {t['distance_mm']:>5} mm  ({len(results)} total)")

        except Exception as exc:
            print(f"[OAK] Loop error: {exc}")
            import traceback; traceback.print_exc()


# ── Standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== OAK-D Detector — standalone test ===")
    print("Press Ctrl+C to stop\n")

    detector = OakDetector(confidence_threshold=0.5)
    detector.start()

    try:
        while True:
            dets = detector.get_detections()
            if dets:
                print("\n── Detections (closest first) ──────────────")
                for d in dets:
                    print(f"  {d['label']:<15} | {d['position']:<6} | {d['distance_mm']:>5} mm")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()