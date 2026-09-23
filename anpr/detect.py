import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

from anpr.ocr import PlateOCR
from anpr.plate_utils import (
    assign_plate_to_vehicle,
    box_to_json,
    clamp_box
)


# ========================== CONFIGURARE ==========================

VEHICLE_CLASS_IDS = [2, 5]  # 2 = car, 5 = bus

VEHICLE_CONFIDENCE = 0.30
PLATE_CONFIDENCE = 0.20

VEHICLE_INFERENCE_SIZE = 1280
PLATE_INFERENCE_SIZE = 1280

# Rulează detectorul plăcuțelor pe fiecare cadru.
PLATE_FRAME_INTERVAL = 1

# Începe OCR-ul relativ devreme.
OCR_ZONE_RATIO = 0.25

PLATE_PADDING_X = 10
PLATE_PADDING_Y = 6

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Minimum de voturi pentru ca rezultatul să fie considerat confirmat.
MIN_VOTES_FOR_CONFIRMATION = 2

# Limităm numărul de citiri salvate per vehicul.
MAX_VOTES_PER_VEHICLE = 30

# =================================================================


def get_best_plate(
    votes: List[Tuple[str, float]]
) -> Tuple[str | None, float, int]:
    """
    Alege textul care apare cel mai des.

    La egalitate, câștigă textul cu media confidence mai mare.

    Returnează:
        best_text,
        average_confidence,
        vote_count
    """

    if not votes:
        return None, 0.0, 0

    grouped = defaultdict(list)

    for text, confidence in votes:
        grouped[text].append(
            float(confidence)
        )

    ranked = []

    for text, confidences in grouped.items():
        vote_count = len(confidences)

        average_confidence = (
            sum(confidences)
            / vote_count
        )

        maximum_confidence = max(
            confidences
        )

        ranked.append(
            (
                vote_count,
                average_confidence,
                maximum_confidence,
                text
            )
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2]
        ),
        reverse=True
    )

    (
        vote_count,
        average_confidence,
        _,
        best_text
    ) = ranked[0]

    return (
        best_text,
        average_confidence,
        vote_count
    )


def draw_corner_border(
    frame,
    box,
    color=(0, 255, 0),
    thickness=7
):
    """
    Desenează colțurile vehiculului, precum în vizualizarea finală.
    """

    x1, y1, x2, y2 = map(
        int,
        box
    )

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    line_x = max(
        20,
        min(90, int(width * 0.25))
    )

    line_y = max(
        20,
        min(90, int(height * 0.25))
    )

    # Stânga sus
    cv2.line(
        frame,
        (x1, y1),
        (x1 + line_x, y1),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + line_y),
        color,
        thickness
    )

    # Dreapta sus
    cv2.line(
        frame,
        (x2, y1),
        (x2 - line_x, y1),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + line_y),
        color,
        thickness
    )

    # Stânga jos
    cv2.line(
        frame,
        (x1, y2),
        (x1 + line_x, y2),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - line_y),
        color,
        thickness
    )

    # Dreapta jos
    cv2.line(
        frame,
        (x2, y2),
        (x2 - line_x, y2),
        color,
        thickness
    )
    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - line_y),
        color,
        thickness
    )


def resize_plate_crop(
    crop,
    target_height: int = 100
):
    if crop is None or crop.size == 0:
        return None

    height, width = crop.shape[:2]

    if height <= 0 or width <= 0:
        return None

    scale = target_height / height

    target_width = max(
        1,
        int(width * scale)
    )

    return cv2.resize(
        crop,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC
    )


def draw_plate_overlay(
    frame,
    vehicle_box,
    plate_crop,
    plate_text
):
    """
    Afișează crop-ul mărit și numărul deasupra vehiculului.
    """

    if not plate_text:
        return

    enlarged_crop = resize_plate_crop(
        plate_crop,
        target_height=100
    )

    if enlarged_crop is None:
        return

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        vehicle_box
    )

    crop_height, crop_width = (
        enlarged_crop.shape[:2]
    )

    text_area_height = 62
    spacing = 15

    center_x = int(
        (x1 + x2) / 2
    )

    overlay_x1 = int(
        center_x - crop_width / 2
    )

    overlay_x1 = max(
        0,
        min(
            frame_width - crop_width,
            overlay_x1
        )
    )

    crop_y2 = y1 - spacing
    crop_y1 = crop_y2 - crop_height

    text_y1 = crop_y2
    text_y2 = text_y1 + text_area_height

    # Dacă nu este loc deasupra, mutăm sub vehicul.
    if crop_y1 < 0:
        crop_y1 = y2 + spacing
        crop_y2 = crop_y1 + crop_height
        text_y1 = crop_y2
        text_y2 = text_y1 + text_area_height

    if (
        crop_y1 < 0
        or text_y2 > frame_height
    ):
        return

    overlay_x2 = overlay_x1 + crop_width

    frame[
        crop_y1:crop_y2,
        overlay_x1:overlay_x2
    ] = enlarged_crop

    cv2.rectangle(
        frame,
        (overlay_x1, text_y1),
        (overlay_x2, text_y2),
        (255, 255, 255),
        -1
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.25
    thickness = 3

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        plate_text,
        font,
        font_scale,
        thickness
    )

    while (
        text_width > crop_width - 12
        and font_scale > 0.45
    ):
        font_scale -= 0.05

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            plate_text,
            font,
            font_scale,
            thickness
        )

    text_x = int(
        overlay_x1
        + (crop_width - text_width) / 2
    )

    text_y = int(
        text_y1
        + (
            text_area_height
            + text_height
        ) / 2
        - baseline
    )

    cv2.putText(
        frame,
        plate_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def write_raw_csv(
    rows: List[Dict],
    output_csv_path: str
) -> None:
    fieldnames = [
        "frame_nmr",
        "car_id",
        "car_bbox",
        "license_plate_bbox",
        "license_plate_bbox_score",
        "license_number",
        "license_number_score",
        "plate_crop_path"
    ]

    with open(
        output_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Rezultatele ANPR au fost salvate în: "
        f"{output_csv_path}"
    )


def run_detection(
    video_path: str,
    vehicle_model_path: str,
    plate_model_path: str,
    output_csv_path: str = "outputs/anpr/anpr_raw.csv"
) -> None:

    output_dir = Path(
        output_csv_path
    ).parent

    crop_dir = (
        output_dir
        / "crops"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Ștergem crop-urile rulării precedente.
    for crop_file in crop_dir.glob("*"):
        if crop_file.is_file():
            crop_file.unlink()

    vehicle_model = YOLO(
        vehicle_model_path
    )

    plate_model = YOLO(
        plate_model_path
    )

    ocr = PlateOCR(
        use_gpu=True,
        strict_romanian_format=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Nu s-a putut deschide videoclipul: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    frame_delay = max(
        1,
        int(1000 / fps)
    )

    rows: List[Dict] = []

    # car_id -> [(text, confidence), ...]
    plate_votes = defaultdict(list)

    # car_id -> cel mai bun crop pentru fiecare text
    best_crops = defaultdict(dict)

    # car_id -> ultima detecție a plăcuței
    last_plate_boxes = {}

    # car_id -> ultima box a vehiculului
    last_vehicle_boxes = {}

    frame_number = -1
    paused = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        frame_height, frame_width = (
            frame.shape[:2]
        )

        ocr_min_y = int(
            frame_height
            * OCR_ZONE_RATIO
        )

        annotated = frame.copy()

        # =====================================================
        # 1. VEHICLE DETECTION + BYTETRACK
        # =====================================================

        vehicle_results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASS_IDS,
            conf=VEHICLE_CONFIDENCE,
            imgsz=VEHICLE_INFERENCE_SIZE,
            verbose=False
        )

        tracked_vehicles: List[
            Tuple[
                int,
                float,
                float,
                float,
                float
            ]
        ] = []

        vehicle_result = vehicle_results[0]

        if (
            vehicle_result.boxes is not None
            and vehicle_result.boxes.id is not None
        ):
            tracker_ids = (
                vehicle_result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            vehicle_boxes = (
                vehicle_result.boxes.xyxy
                .cpu()
                .tolist()
            )

            for tracker_id, box in zip(
                tracker_ids,
                vehicle_boxes
            ):
                vx1, vy1, vx2, vy2 = box

                tracker_id = int(
                    tracker_id
                )

                tracked_vehicles.append(
                    (
                        tracker_id,
                        float(vx1),
                        float(vy1),
                        float(vx2),
                        float(vy2)
                    )
                )

                last_vehicle_boxes[
                    tracker_id
                ] = (
                    int(vx1),
                    int(vy1),
                    int(vx2),
                    int(vy2)
                )

        # =====================================================
        # 2. PLATE DETECTION + OCR LIVE
        # =====================================================

        if (
            frame_number
            % PLATE_FRAME_INTERVAL
            == 0
        ):
            plate_result = plate_model(
                frame,
                conf=PLATE_CONFIDENCE,
                imgsz=PLATE_INFERENCE_SIZE,
                verbose=False
            )[0]

            if plate_result.boxes is not None:
                for detected_plate in plate_result.boxes:

                    detection_confidence = float(
                        detected_plate.conf[0]
                    )

                    px1, py1, px2, py2 = (
                        detected_plate.xyxy[0]
                        .cpu()
                        .tolist()
                    )

                    # Evităm doar plăcuțele extrem de îndepărtate.
                    if py2 < ocr_min_y:
                        continue

                    car_id = assign_plate_to_vehicle(
                        (
                            px1,
                            py1,
                            px2,
                            py2
                        ),
                        tracked_vehicles
                    )

                    if car_id is None:
                        continue

                    car_id = int(
                        car_id
                    )

                    corresponding_vehicle = next(
                        (
                            vehicle
                            for vehicle
                            in tracked_vehicles
                            if vehicle[0] == car_id
                        ),
                        None
                    )

                    if corresponding_vehicle is None:
                        continue

                    (
                        _,
                        vx1,
                        vy1,
                        vx2,
                        vy2
                    ) = corresponding_vehicle

                    crop_x1, crop_y1, crop_x2, crop_y2 = (
                        clamp_box(
                            (
                                px1 - PLATE_PADDING_X,
                                py1 - PLATE_PADDING_Y,
                                px2 + PLATE_PADDING_X,
                                py2 + PLATE_PADDING_Y
                            ),
                            frame_width,
                            frame_height
                        )
                    )

                    if (
                        crop_x2 <= crop_x1
                        or crop_y2 <= crop_y1
                    ):
                        continue

                    plate_crop = frame[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2
                    ]

                    if plate_crop.size == 0:
                        continue

                    last_plate_boxes[
                        car_id
                    ] = (
                        int(px1),
                        int(py1),
                        int(px2),
                        int(py2)
                    )

                    plate_text, ocr_confidence = (
                        ocr.read_plate(
                            plate_crop
                        )
                    )

                    if plate_text:
                        votes = plate_votes[
                            car_id
                        ]

                        votes.append(
                            (
                                plate_text,
                                float(ocr_confidence)
                            )
                        )

                        # Nu lăsăm lista să crească nelimitat.
                        if (
                            len(votes)
                            > MAX_VOTES_PER_VEHICLE
                        ):
                            plate_votes[
                                car_id
                            ] = votes[
                                -MAX_VOTES_PER_VEHICLE:
                            ]

                        current_crop = (
                            best_crops[
                                car_id
                            ].get(
                                plate_text
                            )
                        )

                        if (
                            current_crop is None
                            or ocr_confidence
                            > current_crop[
                                "ocr_confidence"
                            ]
                        ):
                            crop_filename = (
                                f"car_{car_id}_"
                                f"{plate_text}.jpg"
                            )

                            crop_path = (
                                crop_dir
                                / crop_filename
                            )

                            cv2.imwrite(
                                str(crop_path),
                                plate_crop
                            )

                            best_crops[
                                car_id
                            ][
                                plate_text
                            ] = {
                                "crop": plate_crop.copy(),
                                "crop_path": str(
                                    crop_path
                                ).replace("\\", "/"),
                                "ocr_confidence": float(
                                    ocr_confidence
                                )
                            }

                    (
                        best_text,
                        average_confidence,
                        vote_count
                    ) = get_best_plate(
                        plate_votes.get(
                            car_id,
                            []
                        )
                    )

                    crop_path = ""

                    if best_text:
                        crop_data = (
                            best_crops[
                                car_id
                            ].get(
                                best_text
                            )
                        )

                        if crop_data:
                            crop_path = crop_data[
                                "crop_path"
                            ]

                    rows.append({
                        "frame_nmr": frame_number,
                        "car_id": car_id,
                        "car_bbox": box_to_json(
                            (
                                vx1,
                                vy1,
                                vx2,
                                vy2
                            )
                        ),
                        "license_plate_bbox": box_to_json(
                            (
                                px1,
                                py1,
                                px2,
                                py2
                            )
                        ),
                        "license_plate_bbox_score": (
                            detection_confidence
                        ),
                        "license_number": (
                            best_text or ""
                        ),
                        "license_number_score": (
                            average_confidence
                            if best_text
                            else 0.0
                        ),
                        "plate_crop_path": crop_path
                    })

        # =====================================================
        # 3. VIZUALIZARE LIVE
        # =====================================================

        for (
            tracker_id,
            vx1,
            vy1,
            vx2,
            vy2
        ) in tracked_vehicles:

            vehicle_box = (
                vx1,
                vy1,
                vx2,
                vy2
            )

            draw_corner_border(
                annotated,
                vehicle_box,
                color=(0, 255, 0),
                thickness=7
            )

            (
                best_text,
                average_confidence,
                vote_count
            ) = get_best_plate(
                plate_votes.get(
                    tracker_id,
                    []
                )
            )

            if best_text:
                crop_data = (
                    best_crops[
                        tracker_id
                    ].get(
                        best_text
                    )
                )

                if crop_data:
                    draw_plate_overlay(
                        frame=annotated,
                        vehicle_box=vehicle_box,
                        plate_crop=crop_data[
                            "crop"
                        ],
                        plate_text=best_text
                    )

            plate_box = last_plate_boxes.get(
                tracker_id
            )

            if plate_box:
                px1, py1, px2, py2 = plate_box

                cv2.rectangle(
                    annotated,
                    (px1, py1),
                    (px2, py2),
                    (0, 0, 255),
                    3
                )

        confirmed_plates = sum(
            1
            for tracker_id in plate_votes
            if get_best_plate(
                plate_votes[tracker_id]
            )[2] >= MIN_VOTES_FOR_CONFIRMATION
        )

        status_text = (
            f"Tracked vehicles: "
            f"{len(tracked_vehicles)} | "
            f"Confirmed plates: "
            f"{confirmed_plates}"
        )

        cv2.putText(
            annotated,
            status_text,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            (
                "P = Pause/Resume | "
                "N = Next | "
                "Q = Save & Quit"
            ),
            (30, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        display = cv2.resize(
            annotated,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )

        cv2.imshow(
            "ANPR Live Detection",
            display
        )

        key = cv2.waitKey(
            0 if paused else frame_delay
        ) & 0xFF

        if key == ord("q"):
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
                    break

                if pause_key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()

                    write_raw_csv(
                        rows,
                        output_csv_path
                    )

                    return

    cap.release()
    cv2.destroyAllWindows()

    write_raw_csv(
        rows,
        output_csv_path
    )