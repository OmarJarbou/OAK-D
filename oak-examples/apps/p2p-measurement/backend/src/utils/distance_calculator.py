import numpy as np
from typing import List, Tuple


class DistanceCalculator:
    def __init__(self, camera_matrix: np.ndarray, measurement_history_length: int = 20):
        self.camera_matrix = camera_matrix
        self.show_confidence_interval = False
        self.distances = []
        self.measurement_history_length = measurement_history_length

    def calculate_distance(
        self, points: List[dict], depth_frame: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate distance between two tracked points using depth data
        Returns (distance, standard_deviation)
        """
        if len(points) < 2:
            return -1, 0

        # Get the centers of the bounding boxes
        point1 = self._get_bbox_center(points[0]["bbox"])
        point2 = self._get_bbox_center(points[1]["bbox"])

        return self._calculate_3d_distance(point1, point2, depth_frame)

    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x, y, w, h = bbox
        center_x = int(x + w // 2)
        center_y = int(y + h // 2)
        return (center_x, center_y)

    def _calculate_3d_distance(
        self, p1: Tuple[int, int], p2: Tuple[int, int], depth_frame: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate 3D distance between two points using camera intrinsics"""
        x1, y1 = p1
        x2, y2 = p2

        # Get depth values (check bounds)
        h, w = depth_frame.shape
        if not (0 <= y1 < h and 0 <= x1 < w and 0 <= y2 < h and 0 <= x2 < w):
            return -1, 0

        depth1 = depth_frame[y1, x1]
        depth2 = depth_frame[y2, x2]

        if depth1 == 0 or depth2 == 0:
            return -1, 0

        # Convert depth from mm to meters
        depth1_m = depth1 / 1000.0
        depth2_m = depth2 / 1000.0

        # Convert pixel coordinates to 3D world coordinates
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # 3D coordinates
        x1_3d = (x1 - cx) * depth1_m / fx
        y1_3d = (y1 - cy) * depth1_m / fy
        z1_3d = depth1_m

        x2_3d = (x2 - cx) * depth2_m / fx
        y2_3d = (y2 - cy) * depth2_m / fy
        z2_3d = depth2_m

        # Calculate Euclidean distance
        distance = np.sqrt(
            (x2_3d - x1_3d) ** 2 + (y2_3d - y1_3d) ** 2 + (z2_3d - z1_3d) ** 2
        )

        # Store for statistics
        if distance > 0:
            self.distances.append(distance)
        if len(self.distances) > self.measurement_history_length:
            self.distances = self.distances[1:]

        # Calculate standard deviation if we have multiple measurements
        std_dev = np.std(self.distances) if len(self.distances) > 1 else 0
        distance_median = np.median(self.distances) if len(self.distances) > 0 else 0

        return distance_median, std_dev

    def clear_distances(self):
        """Clear stored distance measurements"""
        self.distances.clear()

    def toggle_confidence_interval(self):
        """Toggle confidence interval display"""
        self.show_confidence_interval = not self.show_confidence_interval

    def get_average_distance(self) -> float:
        """Get average of all stored distances"""
        return np.mean(self.distances) if self.distances else 0
