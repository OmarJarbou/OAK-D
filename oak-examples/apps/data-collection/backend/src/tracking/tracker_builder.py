import depthai as dai

from config import TrackingConfig


def build_tracker_node(
    pipeline: dai.Pipeline,
    input_frame: dai.Node.Output,
    input_detections: dai.Node.Output,
    cfg: TrackingConfig,
) -> dai.node.ObjectTracker:
    """Create and configure DepthAI ObjectTracker."""
    tracker_node = pipeline.create(dai.node.ObjectTracker)
    tracker_node.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    tracker_node.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker_node.setTrackingPerClass(cfg.track_per_class)
    tracker_node.setTrackletBirthThreshold(cfg.birth_threshold)
    tracker_node.setTrackletMaxLifespan(cfg.max_lifespan)
    tracker_node.setOcclusionRatioThreshold(cfg.occlusion_ratio_threshold)
    tracker_node.setTrackerThreshold(cfg.tracker_threshold)

    input_frame.link(tracker_node.inputTrackerFrame)
    input_frame.link(tracker_node.inputDetectionFrame)
    input_detections.link(tracker_node.inputDetections)

    return tracker_node
