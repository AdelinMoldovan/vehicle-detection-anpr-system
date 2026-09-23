import cv2
import numpy as np
from collections import defaultdict, deque


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points,
            self.matrix
        )
        return transformed_points.reshape(-1, 2)


class SpeedEstimator:
    def __init__(self, fps: float, smoothing_window: int = 8):
        self.fps = fps if fps and fps > 0 else 30

        self.coordinates = defaultdict(
            lambda: deque(maxlen=int(self.fps))
        )

        # tracker_id -> recent speed values
        self.speed_history = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )

        # tracker_id -> latest smoothed speed
        self.latest_speeds = {}

        # speeds confirmed only after line crossing
        self.confirmed_speeds = []

    def update(self, tracker_id: int, transformed_y: float):
        self.coordinates[tracker_id].append(transformed_y)

        if len(self.coordinates[tracker_id]) < self.fps / 2:
            return None

        y_start = self.coordinates[tracker_id][0]
        y_end = self.coordinates[tracker_id][-1]

        distance = abs(y_end - y_start)
        time_seconds = len(self.coordinates[tracker_id]) / self.fps

        if time_seconds <= 0:
            return None

        raw_speed = (distance / time_seconds) * 3.6

        self.speed_history[tracker_id].append(raw_speed)

        smoothed_speed = sum(self.speed_history[tracker_id]) / len(
            self.speed_history[tracker_id]
        )

        self.latest_speeds[tracker_id] = smoothed_speed

        return smoothed_speed

    def confirm_speed(self, tracker_id: int):
        if tracker_id in self.latest_speeds:
            self.confirmed_speeds.append(self.latest_speeds[tracker_id])

    def get_average_speed(self):
        if not self.confirmed_speeds:
            return 0.0

        return sum(self.confirmed_speeds) / len(self.confirmed_speeds)