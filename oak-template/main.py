# main.py
import os
from dotenv import load_dotenv
import depthai as dai
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node import ApplyDepthColormap
from depthai_nodes.message import ImgDetectionsExtended
from utils.snaps_producer import SnapsProducer
import cv2
import numpy as np

load_dotenv()
api_key = os.getenv("OAK_API_KEY")
model = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"

visualizer = dai.RemoteConnection(httpPort=8082)

try:
    with dai.Device() as device, dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        model_desc = dai.NNModelDescription(model)
        model_desc.platform = device.getPlatformAsString()
        nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc, apiKey=api_key))

        color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left_cam  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        color_preview = color_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)
        left_out  = left_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)
        right_out = right_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)

        stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        stereo.setLeftRightCheck(True)
        stereo.setPostProcessingHardwareResources(2, 2)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # depth aligned to color camera
        stereo.setOutputSize(1280, 720)  # must be multiple of 16, close to native RGB res

        depth_colormap = pipeline.create(ApplyDepthColormap).build(stereo.disparity)
        depth_colormap.setColormap(cv2.COLORMAP_JET)

        nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(color_cam, nn_archive)

        visualizer.addTopic("Color", color_preview, "images")
        visualizer.addTopic("Depth", depth_colormap.out, "images")
        visualizer.addTopic("YOLO", nn_with_parser.out, "images")

        label_map = nn_archive.getConfigV1().model.heads[0].metadata.classes
        snaps_producer = pipeline.create(SnapsProducer).build(
            nn_with_parser.passthrough, nn_with_parser.out,
            label_map=label_map
        )

        detection_queue = nn_with_parser.out.createOutputQueue()
        depth_queue = stereo.depth.createOutputQueue()

        print("Pipeline created.")
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            try:
                key = visualizer.waitKey(1)
                if key == ord('q'):
                    print("Got q key! Exiting.")
                    pipeline.stop()
                    break

                det = detection_queue.get() if detection_queue.has() else None
                depth_msg = depth_queue.get() if depth_queue.has() else None

                if det is not None and depth_msg is not None:
                    depth_frame = depth_msg.getFrame()
                    if depth_frame is None or depth_frame.size == 0:
                        continue

                    depth_h, depth_w = depth_frame.shape

                    for d in det.detections:
                        label_name = label_map[d.label]

                        # --- Use segmentation mask for depth sampling ---
                        mask = None
                        if hasattr(d, 'mask') and d.mask is not None:
                            try:
                                # Resize mask to depth frame size
                                raw_mask = np.array(d.mask, dtype=np.uint8)
                                mask = cv2.resize(raw_mask, (depth_w, depth_h),
                                                  interpolation=cv2.INTER_NEAREST)
                            except Exception:
                                mask = None

                        if mask is not None:
                            # Sample depth only at mask pixels — most accurate
                            roi = depth_frame[mask > 0]
                        else:
                            # Fallback: center-cropped bbox (no mask available)
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
                        print(f"{label_name} at {distance_m:.2f} meters")

            except Exception as e:
                print("Error in main loop:", e)
                break

except Exception as e:
    print("Pipeline failed:", e)