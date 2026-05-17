#!/usr/bin/env python3
"""
Record depth frames + navigation labels for ML training.

Controls (OpenCV window must be focused):
  W / Up    → CENTER
  A / Left  → LEFT
  D / Right → RIGHT
  S / Space → STOP
  Q         → quit

Output: data/nav_recordings/frame_XXXXXX_depth.npy + frame_XXXXXX_label.txt
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

import depthai as dai

from utils.config import WalkerConfig

load_dotenv()

KEY_TO_LABEL = {
    ord("a"): "LEFT",
    ord("A"): "LEFT",
    81: "LEFT",       # left arrow
    ord("d"): "RIGHT",
    ord("D"): "RIGHT",
    83: "RIGHT",      # right arrow
    ord("w"): "CENTER",
    ord("W"): "CENTER",
    82: "CENTER",     # up arrow
    ord("s"): "STOP",
    ord("S"): "STOP",
    ord(" "): "STOP",
}


def main() -> None:
    cfg = WalkerConfig.from_env()
    out_dir = Path(os.getenv("ML_RECORD_DIR", "data/nav_recordings"))
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("frame_*_depth.npy"))
    frame_idx = len(existing)
    current_label = "STOP"

    print("=" * 60)
    print("  Navigation data recording")
    print(f"  Output: {out_dir.resolve()}")
    print("  W=forward  A=left  D=right  S=stop  Q=quit")
    print("=" * 60)

    api_key = os.getenv("OAK_API_KEY", "")

    with dai.Device() as device, dai.Pipeline(device) as pipeline:
        left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        left_out = left_cam.requestOutput(
            size=(400, 400), type=dai.ImgFrame.Type.NV12, fps=30)
        right_out = right_cam.requestOutput(
            size=(400, 400), type=dai.ImgFrame.Type.NV12, fps=30)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_out, right=right_out)
        stereo.setRectification(True)
        stereo.setSubpixel(True)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(cfg.DEPTH_WIDTH, cfg.DEPTH_HEIGHT)

        depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()

        while pipeline.isRunning():
            if not depth_queue.has():
                time.sleep(0.005)
                continue

            depth_msg = depth_queue.get()
            depth = depth_msg.getFrame()
            if depth is None or depth.size == 0:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in KEY_TO_LABEL:
                current_label = KEY_TO_LABEL[key]

            stem = f"frame_{frame_idx:06d}"
            np.save(out_dir / f"{stem}_depth.npy", depth)
            (out_dir / f"{stem}_label.txt").write_text(current_label, encoding="utf-8")
            frame_idx += 1

            vis = np.clip(depth, cfg.MIN_DEPTH_MM, cfg.MAX_DEPTH_MM)
            vis = (
                (vis - cfg.MIN_DEPTH_MM)
                / max(cfg.MAX_DEPTH_MM - cfg.MIN_DEPTH_MM, 1)
                * 255
            ).astype(np.uint8)
            vis = 255 - vis
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.putText(
                vis, f"{current_label}  saved={frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imshow("Record navigation", vis)

    cv2.destroyAllWindows()
    print(f"[Record] Done. {frame_idx} frames in {out_dir}")


if __name__ == "__main__":
    main()
