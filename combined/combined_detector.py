import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from combined.anpr_manager import ANPRManager
from combined.combined_visualizer import draw_combined_frame
from combined.result_manager import ResultManager
from speed.speed_estimator import ViewTransformer, SpeedEstimator


# ================================================================
# CONFIGURARE GENERALĂ
# ================================================================

VEHICLE_CLASS_IDS = [2, 5]  # 2 = car, 5 = bus

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

VEHICLE_CONFIDENCE = 0.40

# Schimbă aici limita afișată în proiect.
SPEED_LIMIT_KMH = 60.0

# Calibrarea existentă a drumului.
SOURCE = np.array([
    [500, 120],
    [800, 120],
    [1500, 900],
    [-250, 900]
])

TARGET_WIDTH = 20
TARGET_HEIGHT = 90

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
], dtype=np.float32)

LINE_START = sv.Point(960, 300)
LINE_END = sv.Point(330, 300)

# ANPR se execută doar la fiecare N cadre.
# 3 este un compromis între viteză și numărul de citiri.
ANPR_FRAME_INTERVAL = 1

# OCR doar în partea inferioară a imaginii originale.
OCR_ZONE_RATIO = 0.25

PLATE_CONFIDENCE = 0.20
PLATE_INFERENCE_SIZE = 1280

# Lungimea traseului vizual al vehiculului.
TRACE_SECONDS = 2

# Output-uri.
COMBINED_OUTPUT_DIR = Path("outputs/combined")
FINAL_VIDEO_PATH = COMBINED_OUTPUT_DIR / "final_video.mp4"
BIRD_EYE_VIDEO_PATH = COMBINED_OUTPUT_DIR / "bird_eye_view.mp4"
FINAL_CSV_PATH = COMBINED_OUTPUT_DIR / "final_results.csv"
STATISTICS_PATH = COMBINED_OUTPUT_DIR / "statistics.txt"


# ================================================================
# FUNCȚII AUXILIARE
# ================================================================

def scale_point_to_original(
    point,
    scale_x: float,
    scale_y: float
) -> Tuple[int, int]:
    """Transformă un punct din 1280x720 în rezoluția originală."""

    return (
        int(point[0] * scale_x),
        int(point[1] * scale_y)
    )


def scale_box_to_original(
    box,
    scale_x: float,
    scale_y: float
) -> Tuple[int, int, int, int]:
    """Transformă un box din 1280x720 în rezoluția originală."""

    x1, y1, x2, y2 = box

    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    )


def draw_original_polygon_and_line(
    frame,
    scale_x: float,
    scale_y: float
):
    """Desenează poligonul și linia pe cadrul original."""

    scaled_polygon = np.array([
        scale_point_to_original(point, scale_x, scale_y)
        for point in SOURCE
    ], dtype=np.int32)

    cv2.polylines(
        frame,
        [scaled_polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=max(2, int(3 * scale_x))
    )

    original_line_start = scale_point_to_original(
        (LINE_START.x, LINE_START.y),
        scale_x,
        scale_y
    )

    original_line_end = scale_point_to_original(
        (LINE_END.x, LINE_END.y),
        scale_x,
        scale_y
    )

    cv2.line(
        frame,
        original_line_start,
        original_line_end,
        (255, 255, 255),
        max(2, int(3 * scale_x))
    )

"""
def draw_vehicle_traces(
    frame,
    trace_history: Dict[int, deque]
):
    Desenează traseul fiecărui tracker pe cadrul original.

    for points in trace_history.values():
        if len(points) < 2:
            continue

        point_array = np.array(
            points,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [point_array],
            isClosed=False,
            color=(0, 255, 255),
            thickness=3
        )

"""
def draw_vehicle_traces(
    frame,
    trace_history,
    active_tracker_ids
):
    """Desenează traseele doar pentru vehiculele active în cadrul curent."""

    for tracker_id, points in trace_history.items():

        # Nu desenăm traseul dacă vehiculul nu mai este vizibil
        if tracker_id not in active_tracker_ids:
            continue

        if len(points) < 2:
            continue

        point_array = np.array(
            points,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [point_array],
            isClosed=False,
            color=(0, 255, 255),
            thickness=3
        )
def draw_bird_eye_view(
    transformed_points,
    tracker_ids,
    frame_size=(400, 500)
):
    bird_view = np.zeros(
        (frame_size[1], frame_size[0], 3),
        dtype=np.uint8
    )

    scale_x = frame_size[0] / TARGET_WIDTH
    scale_y = frame_size[1] / TARGET_HEIGHT

    for index, point in enumerate(transformed_points):
        x = int(point[0] * scale_x)
        y = int(point[1] * scale_y)

        x = max(0, min(frame_size[0] - 1, x))
        y = max(0, min(frame_size[1] - 1, y))

        cv2.circle(
            bird_view,
            (x, y),
            6,
            (0, 255, 0),
            -1
        )

        if index < len(tracker_ids):
            cv2.putText(
                bird_view,
                str(int(tracker_ids[index])),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    cv2.rectangle(
        bird_view,
        (0, 0),
        (frame_size[0] - 1, frame_size[1] - 1),
        (255, 255, 255),
        1
    )

    cv2.putText(
        bird_view,
        "Bird's Eye View",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return bird_view


def save_statistics(
    cars: int,
    buses: int,
    vehicles_in: int,
    vehicles_out: int,
    average_speed: float,
    result_manager: ResultManager
):
    vehicles = result_manager.get_all_vehicles()

    speeding_count = sum(
        1
        for vehicle in vehicles.values()
        if vehicle.get("speeding", False)
    )

    plates_count = sum(
        1
        for vehicle in vehicles.values()
        if vehicle.get("license_plate")
    )

    with open(
        STATISTICS_PATH,
        "w",
        encoding="utf-8"
    ) as statistics_file:
        statistics_file.write(f"cars={cars}\n")
        statistics_file.write(f"buses={buses}\n")
        statistics_file.write(f"vehicles_in={vehicles_in}\n")
        statistics_file.write(f"vehicles_out={vehicles_out}\n")
        statistics_file.write(
            f"average_speed={average_speed:.2f} km/h\n"
        )
        statistics_file.write(
            f"speed_limit={SPEED_LIMIT_KMH:.2f} km/h\n"
        )
        statistics_file.write(
            f"speeding_vehicles={speeding_count}\n"
        )
        statistics_file.write(
            f"recognized_license_plates={plates_count}\n"
        )


# ================================================================
# PIPELINE PRINCIPAL
# ================================================================

def run_combined_detection(
    video_path: str,
    vehicle_model_path: str,
    plate_model_path: str
) -> None:

    COMBINED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # MODELE ȘI MANAGERE
    # ------------------------------------------------------------

    vehicle_model = YOLO(
        vehicle_model_path
    )

    result_manager = ResultManager(
        speed_limit=SPEED_LIMIT_KMH
    )

    anpr_manager = ANPRManager(
        plate_model_path=plate_model_path,
        output_crop_dir="outputs/combined/crops",
        plate_confidence=PLATE_CONFIDENCE,
        inference_size=PLATE_INFERENCE_SIZE,
        frame_interval=ANPR_FRAME_INTERVAL,
        ocr_zone_ratio=OCR_ZONE_RATIO,
        padding_x=10,
        padding_y=6
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        print(
            f"Nu s-a putut deschide videoclipul: {video_path}"
        )
        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    original_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    scale_x = original_width / FRAME_WIDTH
    scale_y = original_height / FRAME_HEIGHT

    frame_delay = max(
        1,
        int(1000 / fps)
    )

    # ------------------------------------------------------------
    # OUTPUT VIDEO
    # ------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    final_writer = cv2.VideoWriter(
        str(FINAL_VIDEO_PATH),
        fourcc,
        fps,
        (original_width, original_height)
    )

    bird_writer = cv2.VideoWriter(
        str(BIRD_EYE_VIDEO_PATH),
        fourcc,
        fps,
        (400, 500)
    )

    if not final_writer.isOpened():
        print("Nu s-a putut crea final_video.mp4.")
        cap.release()
        return

    # ------------------------------------------------------------
    # SPEED + TRACKING
    # ------------------------------------------------------------

    tracker = sv.ByteTrack(
        frame_rate=fps
    )

    polygon_zone = sv.PolygonZone(
        polygon=SOURCE,
        frame_resolution_wh=(
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    )

    line_zone = sv.LineZone(
        start=LINE_START,
        end=LINE_END
    )

    view_transformer = ViewTransformer(
        source=SOURCE,
        target=TARGET
    )

    speed_estimator = SpeedEstimator(
        fps=fps,
        smoothing_window=12
    )

    # ------------------------------------------------------------
    # STARE
    # ------------------------------------------------------------

    number_of_cars = 0
    number_of_buses = 0

    manual_in_count = 0
    manual_out_count = 0

    line_counted_ids = set()
    vehicle_last_y = {}

    trace_history = defaultdict(
        lambda: deque(
            maxlen=max(
                2,
                int(fps * TRACE_SECONDS)
            )
        )
    )

    frame_index = 0
    paused = False
    quit_requested = False

    # ------------------------------------------------------------
    # LOOP VIDEO
    # ------------------------------------------------------------

    while True:
        ret, original_frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # Acesta rămâne cadrul calibrat pentru speed estimation.
        processing_frame = cv2.resize(
            original_frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # --------------------------------------------------------
        # VEHICLE DETECTION
        # --------------------------------------------------------

        vehicle_result = vehicle_model(
            processing_frame,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(
            vehicle_result
        )

        if detections.confidence is not None:
            confidence_mask = (
                detections.confidence
                > VEHICLE_CONFIDENCE
            )

            detections = detections[
                confidence_mask
            ]

        if detections.class_id is not None:
            class_mask = np.array([
                class_id in VEHICLE_CLASS_IDS
                for class_id
                in detections.class_id
            ])

            detections = detections[
                class_mask
            ]

        # Eliminăm detecțiile foarte mici.
        if len(detections) > 0:
            widths = (
                detections.xyxy[:, 2]
                - detections.xyxy[:, 0]
            )

            heights = (
                detections.xyxy[:, 3]
                - detections.xyxy[:, 1]
            )

            size_mask = (
                (widths > 35)
                & (heights > 35)
            )

            detections = detections[
                size_mask
            ]

        # Zona drumului.
        detections = detections[
            polygon_zone.trigger(
                detections
            )
        ]

        # Eliminăm detecțiile tăiate de marginile imaginii.
        if len(detections) > 0:
            x1_values = detections.xyxy[:, 0]
            x2_values = detections.xyxy[:, 2]

            edge_mask = (
                (x1_values > 15)
                & (
                    x2_values
                    < FRAME_WIDTH - 15
                )
            )

            detections = detections[
                edge_mask
            ]

        detections = tracker.update_with_detections(
            detections
        )

        line_zone.trigger(
            detections
        )

        anchor_points = (
            detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
        )

        transformed_points = (
            view_transformer.transform_points(
                anchor_points
            )
        )

        tracker_ids = []

        if detections.tracker_id is not None:
            tracker_ids = [
                int(tracker_id)
                for tracker_id
                in detections.tracker_id
            ]

        bird_view = draw_bird_eye_view(
            transformed_points=transformed_points,
            tracker_ids=tracker_ids
        )

        # --------------------------------------------------------
        # SPEED + TRACKED VEHICLES
        # --------------------------------------------------------

        tracked_vehicles_original: List[
            Tuple[
                int,
                str,
                float,
                float,
                float,
                float
            ]
        ] = []

        if (
            detections.tracker_id is not None
            and detections.class_id is not None
        ):
            for index, (
                tracker_id,
                class_id
            ) in enumerate(
                zip(
                    detections.tracker_id,
                    detections.class_id
                )
            ):
                tracker_id = int(
                    tracker_id
                )

                class_id = int(
                    class_id
                )

                vehicle_type = (
                    "car"
                    if class_id == 2
                    else "bus"
                )

                box_1280 = (
                    detections.xyxy[index]
                )

                original_box = scale_box_to_original(
                    box_1280,
                    scale_x,
                    scale_y
                )

                (
                    original_x1,
                    original_y1,
                    original_x2,
                    original_y2
                ) = original_box

                tracked_vehicles_original.append(
                    (
                        tracker_id,
                        vehicle_type,
                        original_x1,
                        original_y1,
                        original_x2,
                        original_y2
                    )
                )

                # Creăm înregistrarea comună.
                result_manager.ensure_vehicle(
                    tracker_id,
                    vehicle_type
                )

                speed_kmh = None

                if index < len(
                    transformed_points
                ):
                    transformed_y = float(
                        transformed_points[
                            index
                        ][1]
                    )

                    speed_kmh = (
                        speed_estimator.update(
                            tracker_id,
                            transformed_y
                        )
                    )

                result_manager.update_speed(
                    tracker_id=tracker_id,
                    vehicle_type=vehicle_type,
                    speed_kmh=speed_kmh
                )

                bottom_center_original = (
                    int(
                        (
                            original_x1
                            + original_x2
                        ) / 2
                    ),
                    int(original_y2)
                )

                trace_history[
                    tracker_id
                ].append(
                    bottom_center_original
                )

        # --------------------------------------------------------
        # ANPR PE ACELAȘI TRACKER_ID
        # --------------------------------------------------------

        anpr_results = anpr_manager.process_frame(
            original_frame=original_frame,
            frame_index=frame_index,
            tracked_vehicles=(
                tracked_vehicles_original
            )
        )

        for tracker_id, anpr_data in (
            anpr_results.items()
        ):
            vehicle_data = (
                result_manager.get_vehicle(
                    tracker_id
                )
            )

            vehicle_type = ""

            if vehicle_data:
                vehicle_type = (
                    vehicle_data.get(
                        "vehicle_type",
                        ""
                    )
                )

            result_manager.update_plate(
                tracker_id=tracker_id,
                vehicle_type=vehicle_type,
                license_plate=anpr_data.get(
                    "license_plate"
                ),
                confidence=anpr_data.get(
                    "plate_confidence",
                    0.0
                ),
                vote_count=anpr_data.get(
                    "vote_count",
                    0
                )
            )

        # --------------------------------------------------------
        # COUNTING IN / OUT
        # --------------------------------------------------------

        line_y = float(
            LINE_START.y
        )

        for index in range(
            len(detections)
        ):
            if (
                detections.tracker_id is None
                or detections.class_id is None
            ):
                break

            tracker_id = int(
                detections.tracker_id[
                    index
                ]
            )

            class_id = int(
                detections.class_id[
                    index
                ]
            )

            vehicle_type = (
                "car"
                if class_id == 2
                else "bus"
            )

            current_y = float(
                detections.xyxy[
                    index
                ][3]
            )

            if (
                tracker_id in vehicle_last_y
                and tracker_id
                not in line_counted_ids
            ):
                previous_y = vehicle_last_y[
                    tracker_id
                ]

                crossed_down = (
                    previous_y
                    < line_y
                    <= current_y
                )

                crossed_up = (
                    previous_y
                    > line_y
                    >= current_y
                )

                if crossed_down or crossed_up:
                    speed_value = (
                        speed_estimator
                        .latest_speeds
                        .get(
                            tracker_id
                        )
                    )

                    if (
                        speed_value is None
                        or speed_value <= 5
                    ):
                        vehicle_last_y[
                            tracker_id
                        ] = current_y
                        continue

                    direction = (
                        "IN"
                        if crossed_down
                        else "OUT"
                    )

                    if crossed_down:
                        manual_in_count += 1
                    else:
                        manual_out_count += 1

                    if class_id == 2:
                        number_of_cars += 1
                    elif class_id == 5:
                        number_of_buses += 1

                    line_counted_ids.add(
                        tracker_id
                    )

                    speed_estimator.confirm_speed(
                        tracker_id
                    )

                    result_manager.update_speed(
                        tracker_id=tracker_id,
                        vehicle_type=vehicle_type,
                        speed_kmh=speed_value
                    )

                    result_manager.update_direction(
                        tracker_id=tracker_id,
                        vehicle_type=vehicle_type,
                        direction=direction
                    )

            vehicle_last_y[
                tracker_id
            ] = current_y

        # --------------------------------------------------------
        # STATISTICI
        # --------------------------------------------------------

        average_speed = (
            speed_estimator.get_average_speed()
        )

        speeding_count = sum(
            1
            for vehicle
            in result_manager
            .get_all_vehicles()
            .values()
            if vehicle.get(
                "speeding",
                False
            )
        )

        statistics = {
            "cars": number_of_cars,
            "buses": number_of_buses,
            "in": manual_in_count,
            "out": manual_out_count,
            "average_speed": average_speed,
            "speeding": speeding_count
        }

        # --------------------------------------------------------
        # VIZUALIZARE FINALĂ
        # --------------------------------------------------------

        annotated_frame = (
            original_frame.copy()
        )

        draw_original_polygon_and_line(
            annotated_frame,
            scale_x,
            scale_y
        )

        """
        draw_vehicle_traces(
    annotated_frame,
    trace_history
)"""
        active_tracker_ids = set(tracker_ids)

        # Ștergem traseele vehiculelor care nu mai sunt active
        inactive_ids = [
            tracker_id
            for tracker_id in trace_history
            if tracker_id not in active_tracker_ids
        ]

        for tracker_id in inactive_ids:
            del trace_history[tracker_id]

        draw_vehicle_traces(
            annotated_frame,
            trace_history,
            active_tracker_ids
        )

        annotated_frame = draw_combined_frame(
            frame=annotated_frame,
            tracked_vehicles=(
                tracked_vehicles_original
            ),
            result_manager=result_manager,
            anpr_manager=anpr_manager,
            statistics=statistics,
            speed_limit=SPEED_LIMIT_KMH,
            paused=paused
        )

        final_writer.write(
            annotated_frame
        )

        bird_writer.write(
            bird_view
        )

        # Afișarea este micșorată, dar output-ul rămâne
        # la rezoluția originală.
        display_frame = cv2.resize(
            annotated_frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            )
        )

        cv2.imshow(
            "Combined Detection + Speed + ANPR",
            display_frame
        )

        cv2.imshow(
            "Bird's Eye View",
            bird_view
        )

        key = cv2.waitKey(
            0 if paused else frame_delay
        ) & 0xFF

        if key == ord("q"):
            quit_requested = True
            break

        if key == ord("p"):
            paused = not paused

        if paused:
            while True:
                pause_key = (
                    cv2.waitKey(0)
                    & 0xFF
                )

                if pause_key == ord("p"):
                    paused = False
                    break

                if pause_key == ord("n"):
                    # Avansăm un singur frame,
                    # dar rămânem în modul pauză.
                    break

                if pause_key == ord("q"):
                    quit_requested = True
                    break

            if quit_requested:
                break

    # ============================================================
    # FINALIZARE ȘI SALVARE
    # ============================================================

    cap.release()
    final_writer.release()
    bird_writer.release()
    cv2.destroyAllWindows()

    result_manager.save_csv(
        str(FINAL_CSV_PATH)
    )

    save_statistics(
        cars=number_of_cars,
        buses=number_of_buses,
        vehicles_in=manual_in_count,
        vehicles_out=manual_out_count,
        average_speed=(
            speed_estimator.get_average_speed()
        ),
        result_manager=result_manager
    )

    print(
        f"Video final salvat în: {FINAL_VIDEO_PATH}"
    )

    print(
        f"Bird's Eye View salvat în: {BIRD_EYE_VIDEO_PATH}"
    )

    print(
        f"CSV final salvat în: {FINAL_CSV_PATH}"
    )

    print(
        f"Statistici salvate în: {STATISTICS_PATH}"
    )