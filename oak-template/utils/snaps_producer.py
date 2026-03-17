# utils/snaps_producer.py
import depthai as dai
import time


class SnapsProducer(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.em = None
        self.label_map = []
        self.confidence_threshold = 0.7
        self.labels = ["person"]
        self.last_update = 0.0
        self.time_interval = 60.0

    def build(
        self,
        rgb: dai.Node.Output,
        detections: dai.Node.Output,
        label_map: list,
        confidence_threshold: float = 0.7,
        labels: list = None,
        time_interval: float = 60.0,
    ) -> "SnapsProducer":
        self.link_args(rgb, detections)

        self.em = dai.EventsManager()
        self.em.setLogResponse(True)

        self.label_map = label_map
        self.confidence_threshold = confidence_threshold
        self.labels = labels if labels is not None else ["person"]
        self.last_update = 0.0  # ensures first valid detection fires immediately
        self.time_interval = time_interval

        return self

    def process(self, rgb: dai.Buffer, detections: dai.ImgDetections):
        # Guard: skip if pipeline is winding down
        if rgb is None or detections is None:
            return

        now = time.time()
        if now < self.last_update + self.time_interval:
            return  # cooldown not elapsed — skip all detections this frame

        try:
            for det in detections.detections:
                # FIX: was < (inverted), should be >= to require high confidence
                if (
                    det.confidence >= self.confidence_threshold
                    and self.label_map[det.label] in self.labels
                ):
                    self.last_update = now
                    det_xyxy = [det.xmin, det.ymin, det.xmax, det.ymax]
                    extra_data = {
                        "model": "luxonis/yolov6-nano:r2-coco-512x288",
                        "detection_xyxy": str(det_xyxy),
                        "detection_label": str(det.label),
                        "detection_label_str": self.label_map[det.label],
                        "detection_confidence": str(det.confidence),
                    }
                    self.em.sendSnap(
                        "rgb",
                        rgb,
                        [],
                        ["demo"],
                        extra_data,
                    )
                    print(f"Event sent: {extra_data}")
                    break  # one snap per interval is enough
        except Exception as e:
            print(f"[SnapsProducer] process error (skipping frame): {e}")