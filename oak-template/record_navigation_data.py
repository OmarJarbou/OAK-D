#!/usr/bin/env python3
"""
Record depth frames + navigation labels for ML training.

Controls (OpenCV window must be focused):
  W / Up    → CENTER
  A / Left  → LEFT
  D / Right → RIGHT
  S / Space → STOP
  Q         → quit

Behavior:
- Frames are saved ONLY when a control key is pressed.
- This avoids incorrect continuous labels and produces cleaner datasets.

Output:
  data/nav_recordings/frame_XXXXXX_depth.npy
  data/nav_recordings/frame_XXXXXX_label.txt
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


def colorize_depth(depth: np.ndarray, cfg: WalkerConfig) -> np.ndarray:
    vis = np.clip(depth, cfg.MIN_DEPTH_MM, cfg.MAX_DEPTH_MM)

    vis = (
        (vis - cfg.MIN_DEPTH_MM)
        / max(cfg.MAX_DEPTH_MM - cfg.MIN_DEPTH_MM, 1)
        * 255
    ).astype(np.uint8)

    vis = 255 - vis
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    return vis


def save_sample(
    depth: np.ndarray,
    label: str,
    out_dir: Path,
    frame_idx: int,
) -> int:
    stem = f"frame_{frame_idx:06d}"

    np.save(out_dir / f"{stem}_depth.npy", depth)
    (out_dir / f"{stem}_label.txt").write_text(label, encoding="utf-8")

    print(f"[Saved] {stem} -> {label}")

    return frame_idx + 1


def main() -> None:
    cfg = WalkerConfig.from_env()

    out_dir = Path(
        os.getenv("ML_RECORD_DIR", "data/nav_recordings")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("frame_*_depth.npy"))
    frame_idx = len(existing)

    last_saved_label = "NONE"

    print("=" * 60)
    print("  Navigation data recording")
    print(f"  Output: {out_dir.resolve()}")
    print("  W=forward  A=left  D=right  S=stop  Q=quit")
    print("  Saves ONLY when a key is pressed")
    print("=" * 60)

    with dai.Device() as device, dai.Pipeline(device) as pipeline:

        left_cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B
        )

        right_cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C
        )

        left_out = left_cam.requestOutput(
            size=(400, 400),
            type=dai.ImgFrame.Type.NV12,
            fps=30,
        )

        right_out = right_cam.requestOutput(
            size=(400, 400),
            type=dai.ImgFrame.Type.NV12,
            fps=30,
        )

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_out,
            right=right_out,
        )

        stereo.setRectification(True)
        stereo.setSubpixel(True)
        stereo.setLeftRightCheck(True)

        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        stereo.setOutputSize(
            cfg.DEPTH_WIDTH,
            cfg.DEPTH_HEIGHT,
        )

        depth_queue = stereo.depth.createOutputQueue(
            maxSize=4,
            blocking=False,
        )

        pipeline.start()

        while pipeline.isRunning():

            if not depth_queue.has():
                time.sleep(0.005)
                continue

            depth_msg = depth_queue.get()

            depth = depth_msg.getFrame()

            if depth is None or depth.size == 0:
                continue

            # ---------------------------------------------------
            # Visualization
            # ---------------------------------------------------

            vis = colorize_depth(depth, cfg)

            cv2.putText(
                vis,
                f"Saved: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"Last label: {last_saved_label}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "Press W/A/D/S to save sample",
                (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Record navigation", vis)

            # ---------------------------------------------------
            # Keyboard input
            # ---------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in KEY_TO_LABEL:

                label = KEY_TO_LABEL[key]

                frame_idx = save_sample(
                    depth=depth,
                    label=label,
                    out_dir=out_dir,
                    frame_idx=frame_idx,
                )

                last_saved_label = label

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(f"[Record] Done. {frame_idx} total samples")
    print(f"[Record] Dataset path: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()