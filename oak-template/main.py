# main.py
import os
from dotenv import load_dotenv
import depthai as dai
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from depthai_nodes.node import ApplyDepthColormap
from utils.snaps_producer import SnapsProducer
import cv2
import numpy as np

# 0. Environment and model setup
load_dotenv()
api_key = os.getenv("OAK_API_KEY")
depthai_hub_api_key = os.getenv("DEPTHAI_HUB_API_KEY")
model = "luxonis/yolov6-nano:r2-coco-512x288"

# 1. Visualizer & Device
visualizer = dai.RemoteConnection(httpPort=8082)

try:
    with dai.Device() as device, dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        # 2a. YOLO Model
        model_desc = dai.NNModelDescription(model)
        model_desc.platform = device.getPlatformAsString()
        nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc, apiKey=api_key))

        # 2b. Cameras
        color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left_cam  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        color_preview = color_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)
        left_out = left_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)
        right_out = right_cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.NV12, fps=30)

        # 2c. Stereo Depth node
        stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)
        stereo.setLeftRightCheck(True)
        stereo.setPostProcessingHardwareResources(2, 2)

        depth_colormap = pipeline.create(ApplyDepthColormap).build(stereo.disparity)
        depth_colormap.setColormap(cv2.COLORMAP_JET)

        # 2d. Parsing Neural Network (YOLO)
        nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(color_cam, nn_archive)

        # 2e. Visualizer topics
        visualizer.addTopic("Color", color_preview, "images")
        visualizer.addTopic("Depth", depth_colormap.out, "images")
        visualizer.addTopic("YOLO", nn_with_parser.out, "images")

        # 2f. SnapsProducer (event snapshots)
        snaps_producer = pipeline.create(SnapsProducer).build(
            nn_with_parser.passthrough, nn_with_parser.out,
            label_map=nn_archive.getConfigV1().model.heads[0].metadata.classes
        )

        # 3. Create output queues (v3 API)
        detection_queue = nn_with_parser.out.createOutputQueue()
        depth_queue = stereo.depth.createOutputQueue()

        # 4. Start pipeline
        print("Pipeline created.")
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        # 5. Main loop: fetch detections & depth safely
        while pipeline.isRunning():
            try:
                key = visualizer.waitKey(1)
                if key == ord('q'):
                    print("Got q key! Exiting.")
                    pipeline.stop()
                    break

                # Fetch messages safely
                det = detection_queue.get() if detection_queue.has() else None
                depth_msg = depth_queue.get() if depth_queue.has() else None

                if det is not None and depth_msg is not None:
                    depth_frame = depth_msg.getFrame()
                    if depth_frame is None or depth_frame.size == 0:
                        continue

                    depth_h, depth_w = depth_frame.shape
                    det_h, det_w = 480, 640  # YOLO output resolution

                    for d in det.detections:
                        # Scale bbox from YOLO space → depth frame space
                        x1_px = int(d.xmin * det_w * depth_w / det_w)
                        y1_px = int(d.ymin * det_h * depth_h / det_h)
                        x2_px = int(d.xmax * det_w * depth_w / det_w)
                        y2_px = int(d.ymax * det_h * depth_h / det_h)

                        # Shrink ROI to center 50% to avoid background bleed
                        cx = (x1_px + x2_px) // 2
                        cy = (y1_px + y2_px) // 2
                        half_w = (x2_px - x1_px) // 4  # 50% of half-width
                        half_h = (y2_px - y1_px) // 4

                        x1_px = max(0, cx - half_w)
                        x2_px = min(depth_w - 1, cx + half_w)
                        y1_px = max(0, cy - half_h)
                        y2_px = min(depth_h - 1, cy + half_h)

                        if x2_px <= x1_px or y2_px <= y1_px:
                            continue

                        roi = depth_frame[y1_px:y2_px, x1_px:x2_px]
                        if roi.size < 10:
                            continue

                        # Filter out invalid depth (0) and far outliers
                        valid = roi[(roi > 100) & (roi < 10000)]  # 10cm–10m range
                        if valid.size < 5:
                            continue

                        # Use median instead of mean — much more robust
                        distance_m = float(np.median(valid)) / 1000.0

                        label_name = nn_archive.getConfigV1().model.heads[0].metadata.classes[d.label]
                        print(f"{label_name} at {distance_m:.2f} meters")

            except Exception as e:
                print("Error in main loop:", e)
                break

except Exception as e:
    print("Pipeline failed:", e)