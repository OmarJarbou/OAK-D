# snaps_producer.py
import depthai as dai
import time
from collections import defaultdict


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
    ) -> "SnapsProducer":
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

        # Update streaks every frame
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

    def close(self):
        if self.em is not None:
            self.em.waitForPendingUploads()
            self.em = None

    def __del__(self):
        self.close()