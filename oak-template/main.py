# main.py
"""
Smart Walker v8.0 — Clean Architecture
═══════════════════════════════════════

اعتمد على SpatialLocationCalculator المدمج في OAK-D:
  - يعطيك مسافة Z بالمليمتر لـ 3 zones (LEFT, CENTER, RIGHT)
  - يشتغل on-device → خفيف على الـ Pi
  - مستقر أكثر بكثير من raw depth pixels

Pipeline:
  StereoDepth → SpatialLocationCalculator (3 ROIs)
                        ↓
                   Navigator (قرار بسيط)
                        ↓
                   Arduino (stepper + brake)
"""

import os
import time
import queue
import threading

import cv2
import numpy as np
from dotenv import load_dotenv

import depthai as dai

from utils.config import Config
from utils.arduino import ArduinoSerial
from utils.navigator import Navigator

# ── Load Config ───────────────────────────────────────────────
load_dotenv()
cfg = Config.from_env()

# ── TTS (اختياري) ─────────────────────────────────────────────
TTS_MAP = {
    "FREE":     "Go forward",
    "STOP":     "Stop",
    "GO:L1":    "Slight left",
    "GO:L2":    "Turn left",
    "GO:R1":    "Slight right",
    "GO:R2":    "Turn right",
}

_tts_queue: queue.Queue = queue.Queue(maxsize=1)
_last_tts_time: float = 0.0
_last_tts_cmd: str = ""


def _tts_worker():
    while True:
        text = _tts_queue.get()
        if text is None:
            break
        try:
            os.system(f'espeak-ng -v en+f3 -s 135 "{text}" >/dev/null 2>&1')
        except Exception as e:
            print(f"[TTS] {e}")


def speak(text: str):
    global _last_tts_time, _last_tts_cmd
    now = time.time()
    if text == _last_tts_cmd and (now - _last_tts_time) < cfg.TTS_COOLDOWN_S:
        return
    _last_tts_time = now
    _last_tts_cmd = text
    try:
        _tts_queue.put_nowait(text)
    except queue.Full:
        pass


# ── Debug Overlay ─────────────────────────────────────────────

def _draw_debug(left_z, center_z, right_z, result, state):
    """رسم بسيط يوضح المسافات والقرار."""
    w, h = 640, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    zone_w = w // 3

    zones = [
        ("LEFT",   left_z,   result.command in ("GO:L1", "GO:L2")),
        ("CENTER", center_z, result.command == "FREE"),
        ("RIGHT",  right_z,  result.command in ("GO:R1", "GO:R2")),
    ]

    for i, (name, z, active) in enumerate(zones):
        x1 = i * zone_w
        x2 = x1 + zone_w

        # لون الـ zone
        if z == float("inf") or z > cfg.FREE_MM:
            bg = (0, 80, 0)       # أخضر داكن → آمن
        elif z > cfg.CAUTION_MM:
            bg = (0, 80, 80)      # أصفر داكن → تنبيه
        elif z > cfg.DANGER_MM:
            bg = (0, 60, 100)     # برتقالي داكن → خطر
        else:
            bg = (0, 0, 100)      # أحمر داكن → ممنوع

        cv2.rectangle(img, (x1, 0), (x2, h - 80), bg, -1)
        if active:
            cv2.rectangle(img, (x1 + 2, 2), (x2 - 2, h - 82), (0, 255, 0), 3)

        # المسافة
        z_txt = f"{z/1000:.2f}m" if z < float("inf") else "---"
        cv2.putText(img, name,  (x1 + 8, 40), font, 0.7, (255, 255, 255), 2)
        cv2.putText(img, z_txt, (x1 + 8, 80), font, 0.9, (0, 255, 255), 2)

        # شريط عمودي يمثل القرب
        bar_h = int(np.clip((1 - z / 4000), 0, 1) * (h - 100))
        bar_color = (0, 200, 255) if not active else (0, 255, 0)
        cv2.rectangle(img, (x1 + 20, h - 80 - bar_h), (x2 - 20, h - 80), bar_color, -1)

    # القرار النهائي
    cmd = result.command
    cmd_color = {
        "FREE": (0, 255, 0), "STOP": (0, 0, 255),
    }.get(cmd, (0, 200, 255))

    cv2.rectangle(img, (0, h - 78), (w, h), (30, 30, 30), -1)
    cv2.putText(img, f"CMD: {cmd}", (10, h - 45), font, 1.0, cmd_color, 2)
    cv2.putText(img, result.reason, (10, h - 15), font, 0.38, (180, 180, 180), 1)

    # حالة الأردوينو
    auth_txt = "AUTH" if state["authorized"] else "LOCKED"
    auth_clr = (0, 255, 0) if state["authorized"] else (0, 0, 255)
    ready_txt = "READY" if state["ready"] else "WAIT"
    cv2.putText(img, auth_txt, (w - 120, h - 45), font, 0.6, auth_clr, 2)
    cv2.putText(img, ready_txt, (w - 120, h - 15), font, 0.5, (200, 200, 200), 1)

    return img


# ── ROI Definitions ───────────────────────────────────────────

def _make_rois(cfg: Config) -> list:
    """
    3 zones أفقية متجاورة في النطاق العمودي بين ROI_Y_TOP و ROI_Y_BOT.
    نتجاهل السقف (فوق TOP) والأرضية (تحت BOT).

    نضيف margin أفقي صغير عشان نتجنب حواف الإطار (زوايا الكاميرا).
    """
    y0, y1 = cfg.ROI_Y_TOP, cfg.ROI_Y_BOT

    # LEFT:   x: 0.05 → 0.35
    # CENTER: x: 0.35 → 0.65
    # RIGHT:  x: 0.65 → 0.95
    rois = [
        dai.Rect(dai.Point2f(0.05, y0), dai.Point2f(0.35, y1)),  # LEFT
        dai.Rect(dai.Point2f(0.35, y0), dai.Point2f(0.65, y1)),  # CENTER
        dai.Rect(dai.Point2f(0.65, y0), dai.Point2f(0.95, y1)),  # RIGHT
    ]
    return rois


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Smart Walker v4.0 — SpatialLocationCalculator")
    print("=" * 55)
    print(f"  Arduino   : {cfg.ARDUINO_PORT}")
    print(f"  DANGER    : {cfg.DANGER_MM:.0f} mm")
    print(f"  CAUTION   : {cfg.CAUTION_MM:.0f} mm")
    print(f"  FREE      : {cfg.FREE_MM:.0f} mm")
    print(f"  Confirm   : {cfg.CONFIRM_FRAMES} frames")
    print(f"  Debug     : {'ON' if cfg.DEBUG_DISPLAY else 'OFF'}")
    print("=" * 55)

    # ── Components ────────────────────────────────────────────
    arduino = ArduinoSerial(port=cfg.ARDUINO_PORT, baud=cfg.ARDUINO_BAUD)
    navigator = Navigator(cfg)

    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    arduino.start()

    was_authorized = False
    last_cmd = ""
    frame_count = 0

    try:
        with dai.Device() as device, dai.Pipeline(device) as pipeline:
            print("[Pipeline] Building...")

            # ── Cameras ───────────────────────────────────────
            left_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B)
            right_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C)

            left_out = left_cam.requestOutput(
                size=(640, 400), type=dai.ImgFrame.Type.NV12, fps=30)
            right_out = right_cam.requestOutput(
                size=(640, 400), type=dai.ImgFrame.Type.NV12, fps=30)

            # ── Stereo Depth ──────────────────────────────────
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left_out, right=right_out)
            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.FAST_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setExtendedDisparity(False)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

            # ── SpatialLocationCalculator ─────────────────────
            # هاد هو القلب الجديد — يحسب Z لكل zone on-device
            slc = pipeline.create(dai.node.SpatialLocationCalculator)
            stereo.depth.link(slc.inputDepth)
            slc.inputConfig.setWaitForMessage(False)

            # أضف الـ 3 ROIs
            rois = _make_rois(cfg)
            for roi in rois:
                roi_cfg = dai.SpatialLocationCalculatorConfigData()
                roi_cfg.roi = roi
                roi_cfg.depthThresholds.lowerThreshold = 200   # mm
                roi_cfg.depthThresholds.upperThreshold = 5000  # mm
                roi_cfg.calculationAlgorithm = \
                    dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                slc.initialConfig.addROI(roi_cfg)

            # ── Output Queues ─────────────────────────────────
            spatial_q = slc.out.createOutputQueue(maxSize=4, blocking=False)

            # نحتاج depth للـ debug display بس
            depth_q = None
            if cfg.DEBUG_DISPLAY:
                depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

            print("[Pipeline] Starting...")
            pipeline.start()
            print("[Pipeline] Running. Waiting for RFID...")

            # ── Main Loop ──────────────────────────────────────
            while pipeline.isRunning():
                try:
                    # انتظر بيانات السلامة
                    if not spatial_q.has():
                        time.sleep(0.005)
                        continue

                    spatial_msg = spatial_q.get()
                    locations = spatial_msg.getSpatialLocations()

                    if len(locations) < 3:
                        continue

                    # Z بالمليمتر لكل zone (أو inf إذا ما في بيانات)
                    def safe_z(loc):
                        z = loc.spatialCoordinates.z
                        return z if (200 < z < 5000) else float("inf")

                    left_z   = safe_z(locations[0])
                    center_z = safe_z(locations[1])
                    right_z  = safe_z(locations[2])

                    frame_count += 1

                    # ── Arduino State ──────────────────────────
                    state = arduino.state.snapshot()

                    # Auth transitions
                    if state["authorized"] and not was_authorized:
                        print("[System] ✓ AUTHORIZED")
                        if cfg.USE_TTS:
                            speak("System authorized")
                        was_authorized = True
                    elif not state["authorized"] and was_authorized:
                        print("[System] LOCKED")
                        navigator.reset()
                        if cfg.USE_TTS:
                            speak("System locked")
                        was_authorized = False

                    # ── Navigate ───────────────────────────────
                    if not state["authorized"] or not state["ready"]:
                        continue

                    result = navigator.update(left_z, center_z, right_z)

                    # ── Send to Arduino ────────────────────────
                    cmd = result.command
                    if cmd != "NONE":
                        force = (cmd == "STOP")
                        sent = arduino.send(cmd, force=force)

                        if sent and cmd != last_cmd:
                            print(
                                f"[Nav] → {cmd:12s} | {result.reason}"
                            )
                            if cfg.USE_TTS:
                                tts_text = TTS_MAP.get(cmd, cmd)
                                speak(tts_text)
                            last_cmd = cmd

                    # ── Debug Display ──────────────────────────
                    if cfg.DEBUG_DISPLAY:
                        debug = _draw_debug(
                            left_z, center_z, right_z, result, state)
                        cv2.imshow("Smart Walker v4.0", debug)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Quit.")
                            break

                    # ── Periodic Log ──────────────────────────
                    if frame_count % 150 == 0:
                        lz = f"{left_z/1000:.2f}" if left_z < 9999 else "---"
                        cz = f"{center_z/1000:.2f}" if center_z < 9999 else "---"
                        rz = f"{right_z/1000:.2f}" if right_z < 9999 else "---"
                        print(
                            f"[Status] f={frame_count} "
                            f"L={lz}m C={cz}m R={rz}m "
                            f"cmd={result.command} "
                            f"auth={state['authorized']} ready={state['ready']}"
                        )

                except KeyboardInterrupt:
                    print("\nInterrupted.")
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
            _tts_queue.put(None)
        except Exception:
            pass
        if cfg.DEBUG_DISPLAY:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    main()