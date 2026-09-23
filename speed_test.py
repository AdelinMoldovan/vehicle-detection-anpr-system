import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ========================= CONFIGURARE =========================

SPEED_THRESHOLD_KMH = 60.0

# Valoare orientativă. Trebuie calibrată pentru video-ul tău.
METERS_PER_PIXEL = 0.05

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35

# COCO:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

MAX_MISSED_FRAMES = 20
SPEED_WINDOW = 10
STATIONARY_SPEED_KMH = 2.0
MIN_TRACK_HITS_FOR_SPEED = 5


# ========================= TRACK INDIVIDUAL =========================

class VehicleTrack:
    def __init__(self, track_id, box, timestamp):
        self.track_id = track_id
        self.box = box

        self.positions = deque(maxlen=SPEED_WINDOW)

        self.missed_frames = 0
        self.hits = 0
        self.speed_kmh = 0.0

        self._add_position(box, timestamp)

    def _add_position(self, box, timestamp):
        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        self.positions.append(
            (center_x, center_y, timestamp)
        )

        self.hits += 1

    def update(self, box, timestamp):
        self.box = box
        self.missed_frames = 0

        self._add_position(
            box,
            timestamp
        )

        self._recompute_speed()

    def mark_missed(self):
        self.missed_frames += 1

    def _recompute_speed(self):
        if (
            len(self.positions) < 2
            or self.hits < MIN_TRACK_HITS_FOR_SPEED
        ):
            return

        x_old, y_old, t_old = self.positions[0]
        x_new, y_new, t_new = self.positions[-1]

        delta_time = t_new - t_old

        if delta_time <= 0:
            return

        pixel_distance = np.hypot(
            x_new - x_old,
            y_new - y_old
        )

        distance_meters = (
            pixel_distance
            * METERS_PER_PIXEL
        )

        speed_kmh = (
            distance_meters
            / delta_time
        ) * 3.6

        if speed_kmh < STATIONARY_SPEED_KMH:
            speed_kmh = 0.0

        self.speed_kmh = speed_kmh

    def is_speed_ready(self):
        return (
            self.hits
            >= MIN_TRACK_HITS_FOR_SPEED
        )


# ========================= TRACK MANAGER =========================

class TrackManager:
    def __init__(self):
        self.tracks = {}

    def update(self, detections, timestamp):
        seen_ids = set()

        for (
            track_id,
            x1,
            y1,
            x2,
            y2
        ) in detections:

            seen_ids.add(track_id)

            box = (
                x1,
                y1,
                x2,
                y2
            )

            if track_id in self.tracks:
                self.tracks[
                    track_id
                ].update(
                    box,
                    timestamp
                )
            else:
                self.tracks[
                    track_id
                ] = VehicleTrack(
                    track_id,
                    box,
                    timestamp
                )

        dead_ids = []

        for (
            track_id,
            track
        ) in self.tracks.items():

            if track_id not in seen_ids:
                track.mark_missed()

                if (
                    track.missed_frames
                    > MAX_MISSED_FRAMES
                ):
                    dead_ids.append(
                        track_id
                    )

        for track_id in dead_ids:
            del self.tracks[track_id]

        return self.tracks


# ========================= DESENARE =========================

def draw_track(frame, track):
    x1, y1, x2, y2 = [
        int(value)
        for value in track.box
    ]

    if track.missed_frames == 0:
        alpha = 1.0
    else:
        alpha = max(
            0.35,
            1.0
            - track.missed_frames
            / MAX_MISSED_FRAMES
        )

    if not track.is_speed_ready():
        color = (200, 200, 200)
        label = (
            f"ID {track.track_id}: "
            f"measuring..."
        )
    else:
        is_fast = (
            track.speed_kmh
            >= SPEED_THRESHOLD_KMH
        )

        color = (
            (0, 0, 255)
            if is_fast
            else (0, 255, 0)
        )

        label = (
            f"ID {track.track_id}: "
            f"{track.speed_kmh:.1f} km/h"
        )

    color = tuple(
        int(
            channel * alpha
            + 30 * (1 - alpha)
        )
        for channel in color
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    text_width, text_height = text_size

    cv2.rectangle(
        overlay,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 6, y1),
        color,
        -1
    )

    cv2.putText(
        overlay,
        label,
        (x1 + 3, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ========================= RULARE VIDEO =========================

def run(source, save_path=None):
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(
            f"Nu s-a putut deschide "
            f"sursa video: {source}"
        )
        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    ) or 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    writer = None

    if save_path:
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            fps,
            (width, height)
        )

    manager = TrackManager()
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_index / fps
        frame_index += 1

        results = model.track(
            frame,
            persist=True,
            classes=list(
                VEHICLE_CLASS_IDS
            ),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        detections = []
        result = results[0]

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            boxes = (
                result.boxes.xyxy
                .cpu()
                .tolist()
            )

            for (
                track_id,
                box
            ) in zip(ids, boxes):

                x1, y1, x2, y2 = box

                detections.append(
                    (
                        track_id,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

        tracks = manager.update(
            detections,
            timestamp
        )

        for track in tracks.values():
            if (
                track.missed_frames
                <= MAX_MISSED_FRAMES
            ):
                draw_track(
                    frame,
                    track
                )

        cv2.putText(
            frame,
            (
                f"Threshold: "
                f"{SPEED_THRESHOLD_KMH} km/h | "
                f"red = high speed | "
                f"green = normal | "
                f"{len(tracks)} tracked vehicles"
            ),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        if writer:
            writer.write(frame)
        else:
            display_frame = cv2.resize(
                frame,
                (1280, 720)
            )

            cv2.imshow(
                "Vehicle Speed Test",
                display_frame
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
                break

    cap.release()

    if writer:
        writer.release()

        print(
            f"Video procesat salvat la: "
            f"{save_path}"
        )

    cv2.destroyAllWindows()


# ========================= CALIBRARE =========================

def run_calibration_helper(source):
    cap = cv2.VideoCapture(source)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        print(
            "Nu s-a putut citi un "
            "cadru pentru calibrare."
        )
        return

    display_frame = cv2.resize(
        frame,
        (1280, 720)
    )

    points = []

    window_name = (
        "Calibration - click two points"
    )

    def on_click(
        event,
        x,
        y,
        flags,
        param
    ):
        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and len(points) < 2
        ):
            points.append((x, y))

            cv2.circle(
                display_frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.imshow(
                window_name,
                display_frame
            )

    cv2.imshow(
        window_name,
        display_frame
    )

    cv2.setMouseCallback(
        window_name,
        on_click
    )

    print(
        "Dă click pe două puncte "
        "cu distanță reală cunoscută."
    )

    print(
        "Apasă orice tastă după "
        "selectarea celor două puncte."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        print(
            "Calibrare anulată. "
            "Sunt necesare exact două puncte."
        )
        return

    pixel_distance = np.hypot(
        points[0][0] - points[1][0],
        points[0][1] - points[1][1]
    )

    print(
        f"Distanța dintre puncte: "
        f"{pixel_distance:.2f} px"
    )

    try:
        real_distance = float(
            input(
                "Introdu distanța reală "
                "în metri: "
            )
        )
    except ValueError:
        print(
            "Ai introdus o valoare invalidă."
        )
        return

    meters_per_pixel = (
        real_distance
        / pixel_distance
    )

    print("\n--- REZULTAT CALIBRARE ---")

    print(
        f"METERS_PER_PIXEL = "
        f"{meters_per_pixel:.6f}"
    )

    print(
        "Copiază valoarea în constanta "
        "METERS_PER_PIXEL."
    )


# ========================= MAIN =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Detectează vehicule într-un video "
            "și estimează viteza acestora."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Calea către fișierul video"
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Pornește instrumentul "
            "de calibrare"
        )
    )

    parser.add_argument(
        "--save",
        default=None,
        help=(
            "Calea unde se salvează "
            "videoul procesat"
        )
    )

    args = parser.parse_args()

    if args.calibrate:
        run_calibration_helper(
            args.source
        )
    else:
        run(
            args.source,
            save_path=args.save
        )