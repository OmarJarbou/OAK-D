# main.py
"""
Smart Walker — OAK-D Navigation System v3.0
════════════════════════════════════════════
Full integration with Arduino Mega via Serial1.

Architecture:
  WalkerConfig        → all tunable parameters
  ArduinoSerial       → bidirectional serial (reader + writer threads)
  CorridorAnalyzer    → 7-zone depth + merged free-space groups
  DecisionEngine      → safety gates, group selection, temporal smoothing
  CommandPublisher    → rate-limited command dispatch
  Visualizer          → OpenCV debug overlay (optional)
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

# ── Local Modules ─────────────────────────────────────────────
from utils.config import WalkerConfig
from utils.arduino_serial import ArduinoSerial
from utils.corridor_analyzer import CorridorAnalyzer
from utils.decision_engine import DecisionEngine
from utils.command_publisher import CommandPublisher
from utils.lidar_analyzer import LidarAnalyzer
from utils.fusion_layer import FusionLayer
from utils.tts_player import TTSPlayer

# ── SnapsProducer ─────────────────────────────────────────────
try:
    from utils.snaps_producer import SnapsProducer
except Exception:
    class SnapsProducer(dai.node.HostNode):
        """Fallback SnapsProducer when the real one is unavailable."""
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


# ── Config ────────────────────────────────────────────────────
load_dotenv()

cfg = WalkerConfig.from_env()
api_key = os.getenv("OAK_API_KEY", "")
use_visualizer = os.getenv("USE_VISUALIZER", "0") == "1"
model = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"


# ── TTS ───────────────────────────────────────────────────────
NAV_TTS_COOLDOWN = 3.0
tts_queue = queue.Queue(maxsize=2)
_tts_generation = 0

last_tts_time = 0
last_text = None

# Pre-cached WAV player (initialised in main() after cfg is ready)
_tts_player: TTSPlayer | None = None


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
            print(f"[TTS] skipped stale: {text}")
            continue

        last_text = text
        last_tts_time = now

        try:
            print(f"[TTS worker] saying: {text}")
            if _tts_player is not None:
                _tts_player.play_blocking(text)
            else:
                import os
                os.system(f'espeak-ng -v en+f3 -s 135 "{text}" >/dev/null 2>&1')
            if gen != _tts_generation:
                print(f"[TTS] aborted stale after play: {text}")
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


TTS_MAP = {
    "GO:CENTER": "Go forward",
    "GO:L1":     "Slight left",
    "GO:L2":     "Turn left",
    "GO:LEFT":   "Hard left",
    "GO:R1":     "Slight right",
    "GO:R2":     "Turn right",
    "GO:RIGHT":  "Hard right",
    "STOP":      "Stop",
    "FREE":      "Free mode",
}


# ══════════════════════════════════════════════════════════════
# Debug Visualization — shows merged groups
# ══════════════════════════════════════════════════════════════

def build_debug_frame(depth_frame, analysis, result, arduino_state,
                      last_sent, cfg_ref):
    """Build debug visualization showing zones, merged groups, and decision.

    Overlay uses a colormap for human viewing only; corridor logic uses raw mm.
    """
    # Colorize raw depth (mm) for display — not used for obstacle classification
    clipped = np.clip(depth_frame, cfg_ref.MIN_DEPTH_MM, cfg_ref.MAX_DEPTH_MM)
    norm = (
        (clipped - cfg_ref.MIN_DEPTH_MM)
        / max(cfg_ref.MAX_DEPTH_MM - cfg_ref.MIN_DEPTH_MM, 1)
        * 255.0
    ).astype(np.uint8)
    norm = 255 - norm
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    h, w = depth_frame.shape
    x1, y1, x2, y2 = analysis.roi_box

    # Blackout floor
    if analysis.floor_mask is not None and analysis.floor_mask.size > 1:
        roi_vis = vis[y1:y2, x1:x2]
        if roi_vis.shape[:2] == analysis.floor_mask.shape:
            roi_vis[analysis.floor_mask] = [0, 0, 0]

    # Draw ROI rectangle
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

    roi_w = x2 - x1
    zone_w = roi_w // cfg_ref.NUM_ZONES
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Draw merged group backgrounds first ──────────────
    for g in analysis.groups:
        if not g.zone_indices:
            continue
        gx1 = x1 + g.zone_indices[0] * zone_w
        gx2 = x1 + (g.zone_indices[-1] + 1) * zone_w
        if g.zone_indices[-1] == cfg_ref.NUM_ZONES - 1:
            gx2 = x2  # Last zone extends to edge

        # Valid group = green tint, invalid = orange tint
        color = (0, 180, 0) if g.is_valid else (0, 100, 180)
        overlay = vis.copy()
        cv2.rectangle(overlay, (gx1, y2 - 30), (gx2, y2), color, -1)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw merged width label
        width_txt = f"{g.total_width_m:.2f}m"
        status = "OK" if g.is_valid else "NARROW"
        mid_x = (gx1 + gx2) // 2 - 40
        cv2.putText(vis, f"{width_txt} {status}", (mid_x, y2 - 10),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Highlight chosen corridor ────────────────────────
    if result.chosen_corridor:
        chosen_m = analysis.corridors.get(result.chosen_corridor)
        if chosen_m:
            ci = chosen_m.zone_index
            cx1 = x1 + ci * zone_w
            cx2 = cx1 + zone_w
            overlay = vis.copy()
            cv2.rectangle(overlay, (cx1, y1), (cx2, y2 - 30), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
            # Draw target marker
            cv2.arrowedLine(vis, ((cx1 + cx2) // 2, y2 - 35),
                           ((cx1 + cx2) // 2, y1 + 10),
                           (0, 255, 0), 3, tipLength=0.15)

    # ── Draw individual zone info ────────────────────────
    for i, name in enumerate(cfg_ref.ZONE_NAMES):
        zx = x1 + i * zone_w

        # Separator
        if i > 0:
            cv2.line(vis, (zx, y1), (zx, y2), (180, 180, 180), 1)

        m = analysis.corridors.get(name)
        if m is None:
            continue

        # Color by clear/blocked status
        if name == result.chosen_corridor:
            color = (0, 255, 0)         # Bright green — target
        elif m.is_clear:
            color = (255, 255, 0)       # Cyan — clear
        else:
            color = (0, 0, 255)         # Red — blocked

        # Zone metrics
        text_x = zx + 3
        text_y = y1 + 16
        fs = 0.35
        lines = [
            name,
            f"p20:{int(m.p20_depth)}",
            f"cc:{m.largest_close_blob_px}",
            f"v:{m.vertical_close_run_frac:.2f}",
            f"sf:{m.safety_score:.2f}",
            f"w:{m.zone_width_m:.2f}m",
            f"{'CLR' if m.is_clear else 'BLK'}",
        ]
        for j, txt in enumerate(lines):
            cv2.putText(vis, txt, (text_x, text_y + j * 14),
                        font, fs, color, 1, cv2.LINE_AA)

    # ── Bottom Info Panel ─────────────────────────────────
    panel_y = h - 5

    # Raw + Stable command
    raw_color = (0, 0, 255) if "STOP" in result.raw_command else (0, 200, 255)
    cv2.putText(vis, f"RAW: {result.raw_command}", (10, panel_y - 80),
                font, 0.55, raw_color, 2, cv2.LINE_AA)

    stable_color = {
        "STOP": (0, 0, 255), "GO:CENTER": (0, 255, 0), "NONE": (128, 128, 128),
    }.get(result.stable_command, (0, 200, 255))
    cv2.putText(vis, f"CMD: {result.stable_command}", (10, panel_y - 55),
                font, 0.65, stable_color, 2, cv2.LINE_AA)

    # Confidence
    conf_color = ((0, 255, 0) if result.confidence > 0.6
                  else (0, 200, 255) if result.confidence > 0.3
                  else (0, 0, 255))
    cv2.putText(vis, f"Conf: {result.confidence:.0%}", (10, panel_y - 32),
                font, 0.5, conf_color, 2, cv2.LINE_AA)

    # Last sent + reason
    cv2.putText(vis, f"Sent: {last_sent or '-'}", (10, panel_y - 12),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, result.reason[:70], (10, panel_y),
                font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Arduino state (right side)
    auth_txt = "AUTH" if arduino_state.get("authorized") else "LOCKED"
    auth_color = (0, 255, 0) if arduino_state.get("authorized") else (0, 0, 255)
    cv2.putText(vis, auth_txt, (w - 100, panel_y - 80),
                font, 0.55, auth_color, 2, cv2.LINE_AA)

    mode_txt = arduino_state.get("mode", "?")
    cv2.putText(vis, f"Mode:{mode_txt}", (w - 130, panel_y - 55),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    ready_txt = "READY" if arduino_state.get("ready") else "WAIT"
    ready_clr = (0, 255, 0) if arduino_state.get("ready") else (0, 165, 255)
    cv2.putText(vis, ready_txt, (w - 100, panel_y - 32),
                font, 0.5, ready_clr, 2, cv2.LINE_AA)

    # Group info
    n_groups = len(analysis.groups)
    n_valid = len(analysis.valid_groups)
    cv2.putText(vis, f"Groups:{n_groups} Valid:{n_valid}", (w - 200, panel_y - 12),
                font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Center rejection reason (if any)
    cbr = getattr(result, 'center_blocked_reason', '')
    if cbr:
        cv2.putText(vis, f"CTR: {cbr[:65]}", (10, y1 - 8),
                    font, 0.38, (0, 165, 255), 1, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Smart Walker - OAK-D Navigation System v3.0")
    print("=" * 60)
    print(f"  Arduino port : {cfg.ARDUINO_PORT}")
    print(f"  Walker width : {cfg.WALKER_WIDTH_M}m + 2x{cfg.SIDE_MARGIN_M}m margin"
          f" = {cfg.REQUIRED_CLEAR_WIDTH_M}m")
    print(f"  Debug display: {'ON' if cfg.DEBUG_DISPLAY else 'OFF'}")
    print(f"  TTS          : {'ON' if cfg.USE_TTS else 'OFF'}")
    print(f"  Snaps        : {'ON' if cfg.ENABLE_SNAPS else 'OFF'}")
    print(
        f"  LiDAR role   : "
        f"{'stop + L/R steering' if cfg.LIDAR_STEERING_ENABLED else 'STOP/front only (camera steers)'}"
    )
    print("=" * 60)

    # ── Components ────────────────────────────────────────
    global _tts_player
    _tts_player = TTSPlayer(sounds_dir="/home/lama/OAK-D/sounds")
    if cfg.USE_TTS:
        _tts_player.pregenerate()   # generate WAV files once at startup

    arduino = ArduinoSerial(port=cfg.ARDUINO_PORT, baud=cfg.ARDUINO_BAUD)
    corridor_analyzer = CorridorAnalyzer(cfg)
    decision_engine = DecisionEngine(cfg)
    publisher = CommandPublisher(cfg, arduino)
    lidar_mock = cfg.LIDAR_PORT.upper() == "MOCK"
    _lb = cfg.LIDAR_BACKEND.lower().strip()
    _legacy_baud = (
        cfg.LIDAR_BAUD if _lb == "legacy" else cfg.LIDAR_LEGACY_BAUD
    )
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
    fusion = FusionLayer(lidar, cfg)

    visualizer = None
    if use_visualizer:
        visualizer = dai.RemoteConnection(httpPort=8082)
        print("[Visualizer] Enabled on port 8082")
    else:
        print("[Visualizer] Disabled")

    arduino.start()
    lidar.start()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    last_nav_tts_time = 0.0
    was_authorized = False

    try:
        with dai.Device() as device, dai.Pipeline(device) as pipeline:
            print("[Pipeline] Creating...")

            model_desc = dai.NNModelDescription(model)
            model_desc.platform = device.getPlatformAsString()
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc, apiKey=api_key))
            label_map = nn_archive.getConfigV1().model.heads[0].metadata.classes

            # Cameras
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

            # Stereo
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

            # YOLO
            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                color_cam, nn_archive)

            if visualizer is not None:
                visualizer.addTopic("Color", color_preview, "images")
                visualizer.addTopic("Depth", depth_colormap.out, "images")
                visualizer.addTopic("YOLO", nn_with_parser.out, "images")

            _snaps_producer = None
            if cfg.ENABLE_SNAPS:
                _snaps_producer = pipeline.create(SnapsProducer).build(
                    nn_with_parser.passthrough, nn_with_parser.out, label_map=label_map)
                print("[Snaps] Enabled")
            else:
                print("[Snaps] Disabled")

            depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

            print("[Pipeline] Created. Starting...")
            pipeline.start()

            if visualizer is not None:
                visualizer.registerPipeline(pipeline)

            print("[Pipeline] Running. Waiting for RFID authorization...")

            # ══════════════════════════════════════════════
            # Main Loop
            # ══════════════════════════════════════════════
            frame_count = 0

            while pipeline.isRunning():
                try:
                    if visualizer is not None:
                        key = visualizer.waitKey(1)
                        if key == ord("q"):
                            print("Quit key pressed.")
                            pipeline.stop()
                            break

                    # Get depth frame
                    try:
                        if depth_queue.has():
                            depth_msg = depth_queue.get()
                        else:
                            time.sleep(0.005)
                            continue
                    except Exception as e:
                        if "QueueException" in str(e) or not pipeline.isRunning():
                            print("[Pipeline] Depth queue closed during shutdown.")
                            break
                        raise

                    depth_frame = depth_msg.getFrame()
                    if depth_frame is None or depth_frame.size == 0:
                        continue

                    frame_count += 1

                    # Arduino state
                    state = arduino.state.snapshot()

                    # Auth transitions
                    if state["authorized"] and not was_authorized:
                        print("[System] [OK] AUTHORIZED - navigation begins when ready")
                        speak("System authorized")
                        was_authorized = True
                    elif not state["authorized"] and was_authorized:
                        print("[System] DEAUTHORIZED - navigation stopped")
                        decision_engine.reset()
                        publisher.reset()
                        speak("System locked")
                        was_authorized = False

                    # Corridor analysis (merged groups)
                    analysis = corridor_analyzer.analyze(depth_frame)
                    fused = fusion.fuse(analysis)
                    # Apply fusion results back to analysis for decision engine
                    if fused.has_emergency != analysis.has_emergency:
                        from dataclasses import replace
                        analysis = replace(analysis, has_emergency=fused.has_emergency)

                    # Fix 2: LiDAR veto → immediate Stop announcement,
                    # Cooldown added to prevent spamming every frame.
                    # Only announce if the system is actually authorized and ready to move.
                    if state.get("authorized") and state.get("ready"):
                        if fused.fusion_reason == "lidar_veto_emergency" and cfg.USE_TTS:
                            now_t = time.time()
                            if now_t - getattr(main, "last_lidar_stop_time", 0.0) >= NAV_TTS_COOLDOWN:
                                speak("Stop")
                                main.last_lidar_stop_time = now_t

                    # Decision: camera 7-zone scoring for L/R; LiDAR optional side bias.
                    _ll = _lr = 0.0
                    _sel = _ser = False
                    if cfg.LIDAR_STEERING_ENABLED:
                        _ll = fused.lidar_left_mm
                        _lr = fused.lidar_right_mm
                        _sel = fused.side_escape_left
                        _ser = fused.side_escape_right
                        if cfg.FLIP_LR:
                            _ll, _lr = _lr, _ll
                            _sel, _ser = _ser, _sel

                    result = decision_engine.decide(
                        analysis,
                        state,
                        fusion_boost=fused.confidence_boost,
                        lidar_left_mm=_ll,
                        lidar_right_mm=_lr,
                        side_escape_left=_sel,
                        side_escape_right=_ser,
                        fusion_reason=fused.fusion_reason,
                        lidar_front_mm=fused.front_clear_mm,
                    )

                    # Critical stop uses FORWARD depth only (L1/CENTER/R1 + LiDAR front).
                    # Do not use side zones — close obstacle on the avoided side must not
                    # trigger STOP while turning away from it.
                    forward_depths: list[float] = []
                    for zname in ("L1", "CENTER", "R1"):
                        zm = analysis.corridors.get(zname)
                        if zm is not None and zm.p20_depth > 0:
                            forward_depths.append(float(zm.p20_depth))
                    if fused.front_clear_mm > 0:
                        forward_depths.append(float(fused.front_clear_mm))
                    min_p20 = min(forward_depths) if forward_depths else 0.0

                    sent = publisher.publish(
                        result.stable_command,
                        state,
                        reason=result.reason,
                        min_p20_depth=min_p20,
                        stable_count=result.stable_count,
                        critical_stop=result.critical_stop,
                        allow_recenter=result.allow_recenter,
                    )

                    # TTS only when Arduino actually received a new command
                    if sent and result.stable_command != "NONE":
                        now_t = time.time()

                        if not hasattr(main, "last_spoken_command"):
                            main.last_spoken_command = None

                        if (
                            result.stable_command != main.last_spoken_command
                            and now_t - last_nav_tts_time >= NAV_TTS_COOLDOWN
                        ):
                            soft_stop = (
                                result.stable_command == "STOP"
                                and not result.critical_stop
                                and "no valid group width" in result.reason
                                and min_p20 > cfg.SOFT_STOP_TTS_MIN_DEPTH_MM
                            )
                            if soft_stop:
                                print(
                                    f"[TTS] suppressed soft STOP "
                                    f"(depth={min_p20:.0f}mm > "
                                    f"{cfg.SOFT_STOP_TTS_MIN_DEPTH_MM:.0f}mm)"
                                )
                                main.last_spoken_command = result.stable_command
                                last_nav_tts_time = now_t
                            else:
                                tts_text = TTS_MAP.get(
                                    result.stable_command, result.stable_command
                                )
                                print(f"[TTS] speaking: {tts_text}")
                                speak(tts_text)

                                main.last_spoken_command = result.stable_command
                                last_nav_tts_time = now_t

                    # Debug visualization
                    if cfg.DEBUG_DISPLAY:
                        debug_frame = build_debug_frame(
                            depth_frame, analysis, result, state,
                            publisher.last_command or "-", cfg)
                        cv2.imshow("Smart Walker Navigation", debug_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("Quit key pressed.")
                            pipeline.stop()
                            break

                    # Periodic log
                    if frame_count % 200 == 0:
                        n_grp = len(analysis.groups)
                        n_val = len(analysis.valid_groups)
                        lidar_scan = lidar.latest_scan
                        lidar_str = (
                            f"lidar_front={lidar_scan.front_min_mm:.0f}mm"
                            if lidar_scan else "lidar=stale"
                        )
                        print(
                            f"[Status] f={frame_count} "
                            f"auth={state['authorized']} "
                            f"ready={state['ready']} "
                            f"cmd={result.stable_command} "
                            f"conf={result.confidence:.0%} "
                            f"groups={n_grp}/{n_val} "
                            f"target={result.chosen_corridor or '-'}"
                            f" {lidar_str}"
                        )

                except KeyboardInterrupt:
                    print("Interrupted.")
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
            print(f"[Shutdown] Arduino stop error: {e}")
        try:
            lidar.stop()
        except Exception as e:
            print(f"[Shutdown] LiDAR stop error: {e}")
        try:
            tts_queue.put(None)
        except Exception as e:
            print(f"[Shutdown] TTS queue stop error: {e}")
        if cfg.DEBUG_DISPLAY:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[Shutdown] cv2 cleanup error: {e}")
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()
